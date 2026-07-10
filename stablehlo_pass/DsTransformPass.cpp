//===-- DsTransformPass.cpp -----------------------------------------------===//
//
// MLIR pass that rewrites float tensor arithmetic into double-single (DS)
// arithmetic, inlined as native StableHLO ops so XLA can fuse them.
//
// Analogous to double_single_ray/skeleton/Skeleton.cpp but operating on
// StableHLO IR instead of LLVM IR.
//
// DS representation: each float tensor Value is split into two f32 tensor
// Values (hi, lo), tracked in dsMap.  The pair is recombined at function
// exit.
//
// Transformation table (mirrors Skeleton.cpp):
//   stablehlo.add       → ds_add  (two_sum sequences)
//   stablehlo.subtract  → ds_sub
//   stablehlo.multiply  → ds_mul  (two_prod + two_sum sequences)
//   stablehlo.divide    → ds_div  (ported from double_binary32_div)
//   stablehlo.sqrt      → ds_sqrt (ported from double_binary32_sqrt)
//   stablehlo.negate    → ds_negate (negate both components)
//   stablehlo.abs       → ds_abs  (conditional negate, ported from
//                         double-single-lib's double_binary32_fabs)
//   stablehlo.compare   → ds_compare (hi-first, lo-tiebreak lexicographic,
//                         ported from double_binary32_compare)
//   stablehlo.select    → applies the same predicate to both components
//   stablehlo.maximum/minimum → built from ds_compare + select
//   func entry args     → split into (hi, lo) via emitFromFloat
//   func return values  → recombined via emitToFloat
//
// Ops in this table other than add/sub/mul/div/sqrt/negate/abs/compare/
// select/maximum/minimum, dot_general, and reduce are NOT DS-transformed:
// a DS-tracked operand flowing into one of them silently reverts to
// native f32 precision from that point on. Set DS_WARN_UNSUPPORTED=1 to
// have the pass report these to stderr as they're encountered (op name +
// location); default behavior (unset) is unchanged.
//
// negate/abs/compare/divide/sqrt ported from
// double_single_ray/llvm-accuracy-analysis-k-test/double-single-lib
// (double_binary32_neg/fabs/compare/div/sqrt) -- see that library for the
// authoritative reference and handoff.md for port notes. The library's
// __two_mul (FMA-based) is not used as-is for divide/sqrt -- see
// emitDsDiv's comment for why, and what's substituted instead.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
// mlir/Pass/PassPlugin.h is not installed in this MLIR build; provide the
// plugin ABI inline (mirrors the header exactly).
#include "llvm/Support/Compiler.h"
namespace mlir {
#define MLIR_PLUGIN_API_VERSION 1
struct PassPluginLibraryInfo {
    uint32_t APIVersion;
    const char *PluginName;
    const char *PluginVersion;
    void (*RegisterPassesCallback)();
};
} // namespace mlir
#define MLIR_PLUGIN_API_EXPORT LLVM_ATTRIBUTE_WEAK
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdlib>
#include <string>
#include <utility>

using namespace mlir;

namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

// Returns true if the value is an f32 or f64 tensor we should transform.
static bool isFloatTensor(Value v) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t) return false;
    return t.getElementType().isF32() || t.getElementType().isF64();
}

// Returns the f32 tensor type with the same shape as the input type.
static RankedTensorType toF32Type(RankedTensorType t) {
    return RankedTensorType::get(t.getShape(), Float32Type::get(t.getContext()));
}

// ── DS arithmetic helpers ─────────────────────────────────────────────────────
//
// All helpers take an OpBuilder positioned after the op being replaced and
// return (hi, lo) pairs.  They expand directly to stablehlo ops — no custom
// calls — so XLA sees and can fuse the arithmetic.

// two_sum(a, b) → (s, e)
//   s  = a + b
//   bb = s - a
//   e  = (a - (s - bb)) + (b - bb)
static std::pair<Value, Value> emitTwoSum(OpBuilder& bld, Location loc,
                                           Value a, Value b) {
    Value s  = bld.create<stablehlo::AddOp>(loc, a, b);
    Value bb = bld.create<stablehlo::SubtractOp>(loc, s, a);
    Value e  = bld.create<stablehlo::AddOp>(loc,
                   bld.create<stablehlo::SubtractOp>(loc, a,
                       bld.create<stablehlo::SubtractOp>(loc, s, bb)),
                   bld.create<stablehlo::SubtractOp>(loc, b, bb));
    return {s, e};
}

// fast_two_sum(a, b) → (hi, lo).  REQUIRES |a| >= |b| -- Dekker's fast
// form (1 add + 2 subs, vs. two_sum's general 1 add + 4 subs). Ported
// from double-single-lib's __fast_two_sum; used by divide/sqrt's final
// combining steps, where the precondition holds by construction (the
// leading term dominates the correction term).
//   h = a + b
//   t = h - a
//   l = b - t
static std::pair<Value, Value> emitFastTwoSum(OpBuilder& bld, Location loc,
                                               Value a, Value b) {
    Value h = bld.create<stablehlo::AddOp>(loc, a, b);
    Value t = bld.create<stablehlo::SubtractOp>(loc, h, a);
    Value l = bld.create<stablehlo::SubtractOp>(loc, b, t);
    return {h, l};
}

// emitSplat: broadcast a scalar float constant to the shape of ref_val.
static Value emitSplat(OpBuilder& bld, Location loc, Value ref_val, float scalar) {
    auto ty = cast<RankedTensorType>(ref_val.getType());
    APFloat val(scalar);  // double → convert to f32 precision below
    bool lossy = false;
    val.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &lossy);
    auto attr = DenseElementsAttr::get(ty, val);
    return bld.create<stablehlo::ConstantOp>(loc, attr);
}

// emitSplit: Veltkamp split of a into (hi, lo) using constant 4097.
//   t  = 4097 * a
//   hi = t - (t - a)
//   lo = a - hi
static std::pair<Value, Value> emitSplit(OpBuilder& bld, Location loc, Value a) {
    Value c  = emitSplat(bld, loc, a, 4097.0f);
    Value t  = bld.create<stablehlo::MulOp>(loc, c, a);
    Value hi = bld.create<stablehlo::SubtractOp>(loc, t,
                   bld.create<stablehlo::SubtractOp>(loc, t, a));
    Value lo = bld.create<stablehlo::SubtractOp>(loc, a, hi);
    return {hi, lo};
}

// two_prod(a, b) → (p, e)  using the Veltkamp split (constant 4097)
//   p = a * b
//   e = ((a_hi*b_hi - p) + a_hi*b_lo + a_lo*b_hi) + a_lo*b_lo
static std::pair<Value, Value> emitTwoProd(OpBuilder& bld, Location loc,
                                            Value a, Value b) {
    Value p = bld.create<stablehlo::MulOp>(loc, a, b);
    auto [a_hi, a_lo] = emitSplit(bld, loc, a);
    auto [b_hi, b_lo] = emitSplit(bld, loc, b);
    Value e = bld.create<stablehlo::AddOp>(loc,
                  bld.create<stablehlo::AddOp>(loc,
                      bld.create<stablehlo::AddOp>(loc,
                          bld.create<stablehlo::SubtractOp>(loc,
                              bld.create<stablehlo::MulOp>(loc, a_hi, b_hi), p),
                          bld.create<stablehlo::MulOp>(loc, a_hi, b_lo)),
                      bld.create<stablehlo::MulOp>(loc, a_lo, b_hi)),
                  bld.create<stablehlo::MulOp>(loc, a_lo, b_lo));
    return {p, e};
}

// ds_add((a_hi, a_lo), (b_hi, b_lo)) → (out_hi, out_lo)
static std::pair<Value, Value> emitDsAdd(OpBuilder& bld, Location loc,
                                          Value a_hi, Value a_lo,
                                          Value b_hi, Value b_lo) {
    auto [s1, e1] = emitTwoSum(bld, loc, a_hi, b_hi);
    auto [s2, e2] = emitTwoSum(bld, loc, a_lo, b_lo);
    auto [t1, t2] = emitTwoSum(bld, loc, s1,
                        bld.create<stablehlo::AddOp>(loc, s2, e1));
    Value out_lo  = bld.create<stablehlo::AddOp>(loc, t2, e2);
    return {t1, out_lo};
}

// ds_sub((a_hi, a_lo), (b_hi, b_lo)) → (out_hi, out_lo)
// Negate b then delegate to ds_add.
static std::pair<Value, Value> emitDsSub(OpBuilder& bld, Location loc,
                                          Value a_hi, Value a_lo,
                                          Value b_hi, Value b_lo) {
    Value neg_hi = bld.create<stablehlo::NegOp>(loc, b_hi);
    Value neg_lo = bld.create<stablehlo::NegOp>(loc, b_lo);
    return emitDsAdd(bld, loc, a_hi, a_lo, neg_hi, neg_lo);
}

// ds_mul((a_hi, a_lo), (b_hi, b_lo)) → (out_hi, out_lo)
static std::pair<Value, Value> emitDsMul(OpBuilder& bld, Location loc,
                                          Value a_hi, Value a_lo,
                                          Value b_hi, Value b_lo) {
    auto [p1, e1]  = emitTwoProd(bld, loc, a_hi, b_hi);
    Value cross    = bld.create<stablehlo::AddOp>(loc,
                         bld.create<stablehlo::MulOp>(loc, a_hi, b_lo),
                         bld.create<stablehlo::MulOp>(loc, a_lo, b_hi));
    auto [s, e2]   = emitTwoSum(bld, loc, p1, cross);
    Value out_lo   = bld.create<stablehlo::AddOp>(loc,
                         bld.create<stablehlo::AddOp>(loc, e1, e2),
                         bld.create<stablehlo::MulOp>(loc, a_lo, b_lo));
    return {s, out_lo};
}

// ds_div((a_hi, a_lo), (b_hi, b_lo)) → (out_hi, out_lo)
// Ported exactly from double-single-lib's double_binary32_div:
//   t1 = a_hi / b_hi
//   (t2, t3) = two_mul(b_hi, t1)
//   t4 = b_lo * t1
//   t5 = a_hi - t2        [Sterbenz: exact -- t2 is close enough to a_hi
//                           that this subtraction introduces no rounding]
//   t6 = a_lo - t4
//   t7 = t5 + t6
//   t8 = t7 / b_hi
//   (out_hi, out_lo) = fast_two_sum(t1, t8)
//
// KNOWN LIBRARY LIMITATION, kept as-is (not "fixed") per this project's
// port-the-reference-exactly policy: t3, the TwoProd/two_mul residual of
// b_hi*t1, is computed above and then never used again -- t6 subtracts
// t4 (the b_lo*t1 contribution), not t3. Re-deriving the correction
// algebraically: for a/b = t1 + delta, the exact identity
// a_hi - t1*b_hi = t5 - t3 shows the correction sum should be
// `t5 - t3 + a_lo - t4`, not `t5 + a_lo - t4` as written here. This is
// confirmed to be a genuine omission and not an intentional design
// choice: the library's OWN sibling routine for the "divide a DS pair by
// a plain scalar" case, __double_binary_div_double_by_single (see
// emitDsDivByScalar below, used by sqrt), computes the analogous term as
// `al - t3` -- i.e. it correctly keeps the residual that this routine
// drops. IMPORTANT -- this is not an adversarial-input-only corner case:
// t3 captures t1's own single-precision rounding error (t1 = a_hi/b_hi is
// only correctly-rounded, not exact), which the exact identity above
// shows the correction needs regardless of b_lo. Empirically (500k
// random trials, host-side check, both b_lo=0 and b_lo!=0), omitting t3
// costs up to ~1.16e-7 relative error (~2^-23, f32-ULP level) on the
// combined (hi, lo) result for EVERY division through this routine,
// including plain lo=0/lo=0 cases with no cancellation at all -- not the
// ~2^-48-class accuracy a double-word division algorithm of this shape
// is designed to reach. In other words: this port's emitDsDiv provides
// roughly one correctly-rounded division's worth of precision, not the
// deep double-word accuracy the other DS ops (add/sub/mul/sqrt) achieve.
// See test_ds_divide.py's DS_DIV_REL_ERR_BOUND for the bound derived
// from this finding, used uniformly (not just for a divisor-lo!=0 case).
//
// Deviation from the library, deliberate (unlike the t3 omission above,
// which is kept as-is): the library's __two_mul uses a real hardware FMA
// (__builtin_fmaf(a, b, -h)) for its TwoProduct, which cannot be
// expressed as separate StableHLO ops -- by the time a stablehlo.multiply's
// result is available, it is already rounded, so a following
// stablehlo.subtract cannot recover the *unrounded* residual the way a
// true single-rounding FMA instruction does. Substituting the existing
// emitTwoProd (Veltkamp-split-based) here: it computes the same
// error-free product decomposition (p, e) via a mechanism StableHLO can
// actually express, and is already relied on by emitDsMul above. This
// project's own FMA-safety finding (README) also confirms this backend
// does not silently contract separate multiply+subtract StableHLO ops
// into fma.rn.f32, so there's no risk of the Veltkamp sequence being
// "helpfully" corrupted into something resembling the library's FMA path
// (which would round differently) -- see also the note on Sterbenz-style
// subtractions being an elevated-simplifier-risk sequence, same category
// as the TwoSum residual chain Experiment 2b already checks.
static std::pair<Value, Value> emitDsDiv(OpBuilder& bld, Location loc,
                                          Value a_hi, Value a_lo,
                                          Value b_hi, Value b_lo) {
    Value t1 = bld.create<stablehlo::DivOp>(loc, a_hi, b_hi);
    auto [t2, t3] = emitTwoProd(bld, loc, b_hi, t1);
    Value t4 = bld.create<stablehlo::MulOp>(loc, b_lo, t1);
    Value t5 = bld.create<stablehlo::SubtractOp>(loc, a_hi, t2);
    Value t6 = bld.create<stablehlo::SubtractOp>(loc, a_lo, t4);
    Value t7 = bld.create<stablehlo::AddOp>(loc, t5, t6);
    Value t8 = bld.create<stablehlo::DivOp>(loc, t7, b_hi);
    return emitFastTwoSum(bld, loc, t1, t8);
}

// __double_binary_div_double_by_single(a_hi, a_lo, b) → (out_hi, out_lo)
// Ported exactly from double-single-lib's __double_binary_div_double_by_single
// -- divides a DS pair by a plain (non-DS-tracked) scalar. Used only by
// emitDsSqrt's internal refinement step, matching what double_binary32_sqrt
// actually calls.
//   t1 = a_hi / b
//   (t2, t3) = two_mul(b, t1)
//   t4 = a_hi - t2        [Sterbenz: exact]
//   t5 = a_lo - t3
//   t6 = t4 + t5
//   t7 = t6 / b
//   (out_hi, out_lo) = fast_two_sum(t1, t7)
//
// IMPORTANT: this is NOT the same sequence as calling emitDsDiv(a_hi,
// a_lo, b, 0) with a zero lo operand, despite both dividing by something
// with an effectively-zero low channel -- double_binary32_div (what
// emitDsDiv ports, see below) drops the TwoProd/two_mul residual (t3)
// from its correction sum entirely, while THIS routine correctly keeps
// it (t5 = a_lo - t3). Confirmed by reading both routines side by side
// in the reference source: this is a genuine difference between the two
// library functions, not two equivalent formulations of the same thing.
// A DS/DS emitDsDiv-based substitution here would silently inherit
// emitDsDiv's t3-omission (see its comment for the full finding and why
// divide keeps that omission, as a literal port of an authoritative but
// evidently imperfect reference) -- sqrt does not need to inherit it,
// since double_binary32_sqrt calls the *other*, more accurate helper.
static std::pair<Value, Value> emitDsDivByScalar(OpBuilder& bld, Location loc,
                                                   Value a_hi, Value a_lo,
                                                   Value b) {
    Value t1 = bld.create<stablehlo::DivOp>(loc, a_hi, b);
    auto [t2, t3] = emitTwoProd(bld, loc, b, t1);
    Value t4 = bld.create<stablehlo::SubtractOp>(loc, a_hi, t2);
    Value t5 = bld.create<stablehlo::SubtractOp>(loc, a_lo, t3);
    Value t6 = bld.create<stablehlo::AddOp>(loc, t4, t5);
    Value t7 = bld.create<stablehlo::DivOp>(loc, t6, b);
    return emitFastTwoSum(bld, loc, t1, t7);
}

// ds_sqrt((a_hi, a_lo)) → (out_hi, out_lo)
// Ported exactly from double-single-lib's double_binary32_sqrt:
//   t1 = sqrtf(a_hi)
//   (t2, t3) = __double_binary_div_double_by_single(a_hi, a_lo, t1)
//   (t4, t5) = two_sum(t1, t2)
//   t6 = t5 + t3
//   t7 = 0.5 * t4
//   t8 = 0.5 * t6
//   (out_hi, out_lo) = fast_two_sum(t7, t8)
//
// Edge semantics not special-cased, matching the library: a_hi < 0 makes
// t1 = sqrtf(a_hi) = NaN, which propagates through every subsequent step
// (NaN produces NaN under every op used here) -- so ds_sqrt of a negative
// value is (NaN, NaN), not an error. a_hi == 0 makes t1 = sqrtf(0) = 0
// exactly, which then feeds emitDsDivByScalar as b = 0: its first
// division, a_hi/b = 0/0, is IEEE-754 NaN immediately -- a different,
// more direct mechanism than divide's own b_hi==0 case (which needs
// a_hi != 0 to reach the two_prod(0, inf) indeterminate form; see
// emitDsDiv), but the same outcome: sqrt of an exact-zero DS pair is
// (NaN, NaN), reported here rather than silently special-cased to a
// clean zero.
static std::pair<Value, Value> emitDsSqrt(OpBuilder& bld, Location loc,
                                           Value a_hi, Value a_lo) {
    Value t1 = bld.create<stablehlo::SqrtOp>(loc, a_hi);
    auto [t2, t3] = emitDsDivByScalar(bld, loc, a_hi, a_lo, t1);
    auto [t4, t5] = emitTwoSum(bld, loc, t1, t2);
    Value t6 = bld.create<stablehlo::AddOp>(loc, t5, t3);
    Value half = emitSplat(bld, loc, t4, 0.5f);
    Value t7 = bld.create<stablehlo::MulOp>(loc, half, t4);
    Value t8 = bld.create<stablehlo::MulOp>(loc, half, t6);
    return emitFastTwoSum(bld, loc, t7, t8);
}

// ds_abs((a_hi, a_lo)) → (out_hi, out_lo)
// Ported from double-single-lib's double_binary32_fabs. The condition is
// hi >= -lo (equivalently hi + lo >= 0), NOT sign(hi) alone: this
// correctly handles the case hi == 0.0 with lo < 0, where the represented
// value is actually negative but hi's own sign says otherwise. Do not
// "simplify" this to a per-component abs or a hi-only sign test -- lo's
// sign is correlated with hi's, and both of those alternatives are wrong.
static std::pair<Value, Value> emitDsAbs(OpBuilder& bld, Location loc,
                                          Value a_hi, Value a_lo) {
    Value neg_lo  = bld.create<stablehlo::NegOp>(loc, a_lo);
    Value cond    = bld.create<stablehlo::CompareOp>(loc, a_hi, neg_lo,
                        stablehlo::ComparisonDirection::GE);
    Value neg_hi  = bld.create<stablehlo::NegOp>(loc, a_hi);
    Value out_hi  = bld.create<stablehlo::SelectOp>(loc, cond, a_hi, neg_hi);
    Value out_lo  = bld.create<stablehlo::SelectOp>(loc, cond, a_lo, neg_lo);
    return {out_hi, out_lo};
}

// ds_compare((a_hi, a_lo), (b_hi, b_lo), direction) → boolean tensor
// Ported from double-single-lib's double_binary32_compare: lexicographic,
// hi compared first, lo breaks ties. NaN handling is not special-cased --
// it falls out for free from stablehlo.compare's own IEEE-754 NaN
// semantics on the underlying hi/hi and lo/lo comparisons (ordered
// comparisons with NaN are false, NE with NaN is true), exactly matching
// what the library computes by hand via explicit (x == x) checks.
//
// For LE/GE, the hi-component comparison must use the *strict* form (LT
// for LE, GT for GE, not the requested non-strict one) so a hi-tie doesn't
// short-circuit the OR before the lo tiebreak is consulted; the lo
// comparison uses the actual requested direction. LT/GT are already
// strict, so they're their own "strict form". EQ/NE are handled directly
// rather than through this hi/lo-OR pattern, matching the library's
// explicit per-case structure.
static Value emitDsCompare(OpBuilder& bld, Location loc,
                            Value a_hi, Value a_lo, Value b_hi, Value b_lo,
                            stablehlo::ComparisonDirection dir) {
    using CD = stablehlo::ComparisonDirection;
    Value hiEq = bld.create<stablehlo::CompareOp>(loc, a_hi, b_hi, CD::EQ);

    if (dir == CD::EQ) {
        Value loEq = bld.create<stablehlo::CompareOp>(loc, a_lo, b_lo, CD::EQ);
        return bld.create<stablehlo::AndOp>(loc, hiEq, loEq);
    }
    if (dir == CD::NE) {
        Value hiNe = bld.create<stablehlo::CompareOp>(loc, a_hi, b_hi, CD::NE);
        Value loNe = bld.create<stablehlo::CompareOp>(loc, a_lo, b_lo, CD::NE);
        return bld.create<stablehlo::OrOp>(loc, hiNe,
                   bld.create<stablehlo::AndOp>(loc, hiEq, loNe));
    }

    CD strictHiDir = (dir == CD::LE) ? CD::LT : (dir == CD::GE) ? CD::GT : dir;
    Value hiCmp = bld.create<stablehlo::CompareOp>(loc, a_hi, b_hi, strictHiDir);
    Value loCmp = bld.create<stablehlo::CompareOp>(loc, a_lo, b_lo, dir);
    return bld.create<stablehlo::OrOp>(loc, hiCmp,
               bld.create<stablehlo::AndOp>(loc, hiEq, loCmp));
}

// Split an f32/f64 tensor into a DS (hi, lo) pair.
//   For f64: hi = float(v),  lo = float(v - double(hi))
//   For f32: hi = v,         lo = 0  (already exact)
static std::pair<Value, Value> emitFromFloat(OpBuilder& bld, Location loc,
                                              Value v) {
    auto ty    = cast<RankedTensorType>(v.getType());
    auto f32Ty = toF32Type(ty);

    if (ty.getElementType().isF32()) {
        auto zeroAttr = DenseElementsAttr::get(f32Ty,
            APFloat(APFloat::IEEEsingle(), 0u));
        Value lo = bld.create<stablehlo::ConstantOp>(loc, zeroAttr);
        return {v, lo};
    }

    // f64 path
    Value hi        = bld.create<stablehlo::ConvertOp>(loc, f32Ty, v);
    Value hi_as_f64 = bld.create<stablehlo::ConvertOp>(loc, ty, hi);
    Value diff      = bld.create<stablehlo::SubtractOp>(loc, v, hi_as_f64);
    Value lo        = bld.create<stablehlo::ConvertOp>(loc, f32Ty, diff);
    return {hi, lo};
}

// Recombine a DS (hi, lo) pair back into a single tensor of targetType.
//   result = cast(hi, targetType) + cast(lo, targetType)
static Value emitToFloat(OpBuilder& bld, Location loc,
                          Value hi, Value lo, Type targetType) {
    Value hi_cast = bld.create<stablehlo::ConvertOp>(loc, targetType, hi);
    Value lo_cast = bld.create<stablehlo::ConvertOp>(loc, targetType, lo);
    return bld.create<stablehlo::AddOp>(loc, hi_cast, lo_cast);
}

// ── Pass ─────────────────────────────────────────────────────────────────────

struct DsTransformPass
    : public PassWrapper<DsTransformPass, OperationPass<func::FuncOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DsTransformPass)

    StringRef getArgument() const override { return "ds-transform"; }
    StringRef getDescription() const override {
        return "Rewrite float tensor arithmetic to double-single (DS) arithmetic";
    }

    // Maps each original float Value → its DS pair (hi, lo).
    llvm::DenseMap<Value, std::pair<Value, Value>> dsMap;

    // DS_RETURN_PAIRS=1 (opt-in, off by default): see the func.return
    // handler in processOps() for what this changes. Read directly from
    // the environment (mirroring DS_BYPASS/DS_TEST_PASSTHROUGH/DS_PASS_MODE
    // in ds_pjrt_plugin.cpp) rather than as an MLIR pass option, since
    // mlir-ds-opt is always spawned as a child process that inherits the
    // caller's environment (ds_pjrt_plugin.cpp's run_command() passes
    // `environ` straight through to posix_spawn) -- no plumbing through
    // the pass-pipeline string is needed.
    bool returnPairs = false;

    // DS_WARN_UNSUPPORTED=1 (opt-in, off by default): see the fallback
    // check at the end of processOps()'s op loop. Same getenv-based
    // mechanism as returnPairs above, for the same reason.
    bool warnUnsupported = false;

    // ── Entry: split function arguments into DS pairs ─────────────────────
    void convertFuncArgs(func::FuncOp func) {
        OpBuilder b(&func.getBody().front().front());
        for (auto arg : func.getArguments()) {
            if (!isFloatTensor(arg)) continue;
            auto [hi, lo] = emitFromFloat(b, func.getLoc(), arg);
            dsMap[arg] = {hi, lo};
        }
    }

    // ── Core: walk ops and replace float arithmetic ───────────────────────
    void processOps(func::FuncOp func) {
        // Snapshot ops first — same trick as Skeleton.cpp — so newly inserted
        // ops are not re-visited.
        SmallVector<Operation*> origOps;
        func.walk([&](Operation* op) { origOps.push_back(op); });

        SmallVector<Operation*> toErase;

        for (auto* op : origOps) {
            OpBuilder b(op);
            Location loc = op->getLoc();

            // ── stablehlo.constant ────────────────────────────────────────
            // Add constant tensors to dsMap as (constant, 0) so downstream
            // arithmetic ops (e.g. a + 1.0) can be DS-transformed.
            // The original op is kept — hi is the constant's own result.
            //
            // Insert AFTER the constant, not before: for f64 constants,
            // emitFromFloat creates ConvertOps that reference this op's
            // result, so they must appear after it to satisfy SSA dominance.
            if (auto constOp = dyn_cast<stablehlo::ConstantOp>(op)) {
                if (!isFloatTensor(constOp.getResult())) continue;
                OpBuilder bPost(op->getContext());
                bPost.setInsertionPointAfter(op);
                auto [hi, lo] = emitFromFloat(bPost, loc, constOp.getResult());
                dsMap[constOp.getResult()] = {hi, lo};
                continue;
            }

            // ── stablehlo.broadcast_in_dim ────────────────────────────────
            // JAX often lowers scalar constants as const + broadcast before
            // arithmetic. Propagate the DS pair through the broadcast by
            // cloning it for both hi and lo.
            if (auto bcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
                if (!isFloatTensor(bcastOp.getResult())) continue;
                if (!dsMap.count(bcastOp.getOperand())) continue;

                auto [in_hi, in_lo] = dsMap[bcastOp.getOperand()];

                // For f64 inputs the cloned op inherits the f64 result type,
                // but operands are f32 — update the result type to match.
                auto bcastF32Ty = toF32Type(
                    cast<RankedTensorType>(bcastOp.getResult().getType()));

                auto* hi_clone = b.clone(*op);
                hi_clone->setOperand(0, in_hi);
                hi_clone->getResult(0).setType(bcastF32Ty);
                auto* lo_clone = b.clone(*op);
                lo_clone->setOperand(0, in_lo);
                lo_clone->getResult(0).setType(bcastF32Ty);

                dsMap[bcastOp.getResult()] = {hi_clone->getResult(0),
                                              lo_clone->getResult(0)};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.add ─────────────────────────────────────────────
            if (auto addOp = dyn_cast<stablehlo::AddOp>(op)) {
                if (!isFloatTensor(addOp.getResult())) continue;
                if (!dsMap.count(addOp.getLhs()) ||
                    !dsMap.count(addOp.getRhs())) continue;

                auto [a_hi, a_lo] = dsMap[addOp.getLhs()];
                auto [b_hi, b_lo] = dsMap[addOp.getRhs()];
                auto [r_hi, r_lo] = emitDsAdd(b, loc, a_hi, a_lo, b_hi, b_lo);
                dsMap[addOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.subtract ────────────────────────────────────────
            if (auto subOp = dyn_cast<stablehlo::SubtractOp>(op)) {
                if (!isFloatTensor(subOp.getResult())) continue;
                if (!dsMap.count(subOp.getLhs()) ||
                    !dsMap.count(subOp.getRhs())) continue;

                auto [a_hi, a_lo] = dsMap[subOp.getLhs()];
                auto [b_hi, b_lo] = dsMap[subOp.getRhs()];
                auto [r_hi, r_lo] = emitDsSub(b, loc, a_hi, a_lo, b_hi, b_lo);
                dsMap[subOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.multiply ────────────────────────────────────────
            if (auto mulOp = dyn_cast<stablehlo::MulOp>(op)) {
                if (!isFloatTensor(mulOp.getResult())) continue;
                if (!dsMap.count(mulOp.getLhs()) ||
                    !dsMap.count(mulOp.getRhs())) continue;

                auto [a_hi, a_lo] = dsMap[mulOp.getLhs()];
                auto [b_hi, b_lo] = dsMap[mulOp.getRhs()];
                auto [r_hi, r_lo] = emitDsMul(b, loc, a_hi, a_lo, b_hi, b_lo);
                dsMap[mulOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.divide ──────────────────────────────────────────
            // Ported from double-single-lib's double_binary32_div -- see
            // emitDsDiv for the exact sequence and the FMA/two_mul
            // substitution note. No special-casing for b_hi == 0 or
            // negative/NaN inputs -- the library doesn't special-case them
            // either. Note this is NOT the same as plain a/b's IEEE
            // propagation: b_hi == 0 makes t1 = a_hi/b_hi an infinity,
            // and the algorithm's own two_prod(b_hi, t1) step then hits
            // the IEEE-754 0*inf indeterminate form, so division by
            // exactly zero yields NaN here, not a clean +inf. Confirmed
            // this is a property of the reference library's own
            // structure, not introduced by this port -- see
            // tests/test_ds_divide.py's edge-semantics tests.
            if (auto divOp = dyn_cast<stablehlo::DivOp>(op)) {
                if (!isFloatTensor(divOp.getResult())) continue;
                if (!dsMap.count(divOp.getLhs()) ||
                    !dsMap.count(divOp.getRhs())) continue;

                auto [a_hi, a_lo] = dsMap[divOp.getLhs()];
                auto [b_hi, b_lo] = dsMap[divOp.getRhs()];
                auto [r_hi, r_lo] = emitDsDiv(b, loc, a_hi, a_lo, b_hi, b_lo);
                dsMap[divOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.sqrt ─────────────────────────────────────────────
            // Ported from double-single-lib's double_binary32_sqrt -- see
            // emitDsSqrt for the exact sequence, emitDsDivByScalar for its
            // scalar-divide refinement step (a separate, more accurate
            // helper than emitDsDiv -- see emitDsDivByScalar's comment for
            // why), and edge-semantics notes (negative input -> (NaN, NaN);
            // exact-zero input -> (NaN, NaN) via an immediate 0/0, not a
            // clean zero -- neither is special-cased, matching the library).
            if (auto sqrtOp = dyn_cast<stablehlo::SqrtOp>(op)) {
                if (!isFloatTensor(sqrtOp.getResult())) continue;
                if (!dsMap.count(sqrtOp.getOperand())) continue;

                auto [a_hi, a_lo] = dsMap[sqrtOp.getOperand()];
                auto [r_hi, r_lo] = emitDsSqrt(b, loc, a_hi, a_lo);
                dsMap[sqrtOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.negate ──────────────────────────────────────────
            if (auto negOp = dyn_cast<stablehlo::NegOp>(op)) {
                if (!isFloatTensor(negOp.getResult())) continue;
                if (!dsMap.count(negOp.getOperand())) continue;

                auto [a_hi, a_lo] = dsMap[negOp.getOperand()];
                Value r_hi = b.create<stablehlo::NegOp>(loc, a_hi);
                Value r_lo = b.create<stablehlo::NegOp>(loc, a_lo);
                dsMap[negOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.abs ─────────────────────────────────────────────
            if (auto absOp = dyn_cast<stablehlo::AbsOp>(op)) {
                if (!isFloatTensor(absOp.getResult())) continue;
                if (!dsMap.count(absOp.getOperand())) continue;

                auto [a_hi, a_lo] = dsMap[absOp.getOperand()];
                auto [r_hi, r_lo] = emitDsAbs(b, loc, a_hi, a_lo);
                dsMap[absOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.compare ─────────────────────────────────────────
            // Result is a boolean tensor, not a float one -- it is never
            // entered into dsMap; instead its uses are rewired directly to
            // the computed boolean (e.g. a following stablehlo.select).
            // dsMap only ever contains float-tensor keys, so checking
            // dsMap.count() on both operands is sufficient to confirm they
            // are DS-tracked floats -- no separate isFloatTensor check on
            // the operands is needed.
            if (auto cmpOp = dyn_cast<stablehlo::CompareOp>(op)) {
                if (!dsMap.count(cmpOp.getLhs()) ||
                    !dsMap.count(cmpOp.getRhs())) continue;

                auto [a_hi, a_lo] = dsMap[cmpOp.getLhs()];
                auto [b_hi, b_lo] = dsMap[cmpOp.getRhs()];
                Value result = emitDsCompare(b, loc, a_hi, a_lo, b_hi, b_lo,
                                              cmpOp.getComparisonDirection());
                cmpOp.getResult().replaceAllUsesWith(result);
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.select ──────────────────────────────────────────
            // The predicate itself is a plain boolean tensor (possibly
            // produced by the ds_compare handler above, possibly not) --
            // not DS-tracked; the same predicate applies to both hi and lo.
            if (auto selOp = dyn_cast<stablehlo::SelectOp>(op)) {
                if (!isFloatTensor(selOp.getResult())) continue;
                if (!dsMap.count(selOp.getOnTrue()) ||
                    !dsMap.count(selOp.getOnFalse())) continue;

                Value pred = selOp.getPred();
                auto [t_hi, t_lo] = dsMap[selOp.getOnTrue()];
                auto [f_hi, f_lo] = dsMap[selOp.getOnFalse()];
                Value r_hi = b.create<stablehlo::SelectOp>(loc, pred, t_hi, f_hi);
                Value r_lo = b.create<stablehlo::SelectOp>(loc, pred, t_lo, f_lo);
                dsMap[selOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.maximum / stablehlo.minimum ─────────────────────
            // Not in double-single-lib (it has no max/min routine) -- built
            // from ds_compare + select per the task's own instruction, since
            // that composition is straightforward and introduces no new
            // rounding behavior beyond what compare/select already have.
            if (auto maxOp = dyn_cast<stablehlo::MaxOp>(op)) {
                if (!isFloatTensor(maxOp.getResult())) continue;
                if (!dsMap.count(maxOp.getLhs()) ||
                    !dsMap.count(maxOp.getRhs())) continue;

                auto [a_hi, a_lo] = dsMap[maxOp.getLhs()];
                auto [b_hi, b_lo] = dsMap[maxOp.getRhs()];
                Value cond = emitDsCompare(b, loc, a_hi, a_lo, b_hi, b_lo,
                                            stablehlo::ComparisonDirection::GE);
                Value r_hi = b.create<stablehlo::SelectOp>(loc, cond, a_hi, b_hi);
                Value r_lo = b.create<stablehlo::SelectOp>(loc, cond, a_lo, b_lo);
                dsMap[maxOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            if (auto minOp = dyn_cast<stablehlo::MinOp>(op)) {
                if (!isFloatTensor(minOp.getResult())) continue;
                if (!dsMap.count(minOp.getLhs()) ||
                    !dsMap.count(minOp.getRhs())) continue;

                auto [a_hi, a_lo] = dsMap[minOp.getLhs()];
                auto [b_hi, b_lo] = dsMap[minOp.getRhs()];
                Value cond = emitDsCompare(b, loc, a_hi, a_lo, b_hi, b_lo,
                                            stablehlo::ComparisonDirection::LE);
                Value r_hi = b.create<stablehlo::SelectOp>(loc, cond, a_hi, b_hi);
                Value r_lo = b.create<stablehlo::SelectOp>(loc, cond, a_lo, b_lo);
                dsMap[minOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.dot_general ────────────────────────────────────
            // 4-matmul DS decomposition. Given DS pairs (a_hi, a_lo) and
            // (b_hi, b_lo) from dsMap:
            //
            //   p    = dot(a_hi, b_hi)          primary result
            //   e1   = dot(a_hi, b_lo)          hi × lo_b correction
            //   e2   = dot(a_lo, b_hi)          lo_a × hi correction
            //   e3   = dot(a_lo, b_lo)          lo_a × lo_b (small but exact)
            //   corr = e1 + e2 + e3
            //   (out_hi, out_lo) = two_sum(p, corr)
            //
            // Precision note: each sub-dot accumulates in f32 (no two_prod per
            // element), but lo-channel cross terms are fully accounted for.
            // XLA sees 4 dot_generals it can dispatch to cuBLAS independently.
            if (auto dotOp = dyn_cast<stablehlo::DotGeneralOp>(op)) {
                if (!isFloatTensor(dotOp.getResult())) continue;
                if (!dsMap.count(dotOp.getLhs()) ||
                    !dsMap.count(dotOp.getRhs())) continue;

                auto [a_hi, a_lo] = dsMap[dotOp.getLhs()];
                auto [b_hi, b_lo] = dsMap[dotOp.getRhs()];

                // Clone the original op to inherit all attributes (dim numbers,
                // precision config, algorithm, etc.) then swap operands.
                // For f64 inputs the clone inherits the f64 result type; update
                // it to f32 since operands are now the f32 hi/lo channels.
                auto dotF32ResultTy = toF32Type(
                    cast<RankedTensorType>(dotOp.getResult().getType()));
                auto makeDot = [&](Value lhs, Value rhs) -> Value {
                    auto* cloned = b.clone(*op);
                    cloned->setOperand(0, lhs);
                    cloned->setOperand(1, rhs);
                    cloned->getResult(0).setType(dotF32ResultTy);
                    return cloned->getResult(0);
                };

                Value p    = makeDot(a_hi, b_hi);
                Value e1   = makeDot(a_hi, b_lo);
                Value e2   = makeDot(a_lo, b_hi);
                Value e3   = makeDot(a_lo, b_lo);
                Value corr = b.create<stablehlo::AddOp>(loc,
                                 b.create<stablehlo::AddOp>(loc, e1, e2), e3);
                auto [out_hi, out_lo] = emitTwoSum(b, loc, p, corr);

                dsMap[dotOp.getResult()] = {out_hi, out_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.reduce ──────────────────────────────────────────
            // Transform single-input reductions whose input was DS-expanded.
            // Replaces reduce(input) with reduce(input_hi, input_lo) using a
            // DS accumulation body so rounding errors are carried forward.
            if (auto redOp = dyn_cast<stablehlo::ReduceOp>(op)) {
                if (redOp.getInputs().size() != 1) continue;

                Value inp     = redOp.getInputs()[0];
                Value initVal = redOp.getInitValues()[0];
                if (!isFloatTensor(inp)) continue;
                if (!dsMap.count(inp))   continue;

                // Inspect body — must contain exactly one transformable op.
                Block& body = redOp.getBody().front();
                Operation* bodyOp = nullptr;
                for (Operation& bop : body.without_terminator()) {
                    if (bodyOp) { bodyOp = nullptr; break; }
                    bodyOp = &bop;
                }
                if (!bodyOp) continue;
                bool isAdd = isa<stablehlo::AddOp>(bodyOp);
                bool isSub = isa<stablehlo::SubtractOp>(bodyOp);
                bool isMul = isa<stablehlo::MulOp>(bodyOp);
                if (!isAdd && !isSub && !isMul) continue;

                auto [inp_hi, inp_lo] = dsMap[inp];

                // Convert init value to a DS pair: (initVal, 0.0).
                auto [init_hi, init_lo] = emitFromFloat(b, loc, initVal);

                // Result type of each DS component = f32 with same shape as
                // the original result.  For f32 inputs this is a no-op; for f64
                // inputs the accumulation is in f32 and we must use f32 types
                // for both the result and the reduction body block arguments.
                Type resultTy = toF32Type(
                    cast<RankedTensorType>(redOp.getResult(0).getType()));
                // Body arguments are rank-0 scalar tensors — always f32.
                Type scalarTy = toF32Type(
                    cast<RankedTensorType>(initVal.getType()));

                // New reduce: 2 inputs (hi, lo), 2 inits, same dimensions.
                auto newReduce = b.create<stablehlo::ReduceOp>(
                    loc,
                    TypeRange{resultTy, resultTy},
                    ValueRange{inp_hi, inp_lo},
                    ValueRange{init_hi, init_lo},
                    redOp.getDimensions());

                // Build DS reduction body.
                // Block arg order: [acc_hi, acc_lo, elem_hi, elem_lo].
                Block* newBody = new Block();
                newReduce.getBody().push_back(newBody);

                auto acc_hi  = newBody->addArgument(scalarTy, loc);
                auto acc_lo  = newBody->addArgument(scalarTy, loc);
                auto elem_hi = newBody->addArgument(scalarTy, loc);
                auto elem_lo = newBody->addArgument(scalarTy, loc);

                OpBuilder bodyBld = OpBuilder::atBlockEnd(newBody);
                Value res_hi, res_lo;
                if (isAdd)
                    std::tie(res_hi, res_lo) =
                        emitDsAdd(bodyBld, loc, acc_hi, acc_lo, elem_hi, elem_lo);
                else if (isSub)
                    std::tie(res_hi, res_lo) =
                        emitDsSub(bodyBld, loc, acc_hi, acc_lo, elem_hi, elem_lo);
                else
                    std::tie(res_hi, res_lo) =
                        emitDsMul(bodyBld, loc, acc_hi, acc_lo, elem_hi, elem_lo);
                bodyBld.create<stablehlo::ReturnOp>(loc, ValueRange{res_hi, res_lo});

                // Map original result so downstream ops and func.return see it.
                dsMap[redOp.getResult(0)] = {newReduce.getResult(0),
                                             newReduce.getResult(1)};
                toErase.push_back(op);
                continue;
            }

            // ── func.return: recombine DS pairs back to original type ─────
            //
            // Default (returnPairs == false): every dsMap-tracked operand is
            // recombined to its original f32/f64 type via emitToFloat --
            // this branch is byte-for-byte the pre-existing behavior,
            // unconditionally, so DS_RETURN_PAIRS=0/unset cannot change it.
            //
            // DS_RETURN_PAIRS=1: if the *same* dsMap-tracked SSA value is
            // returned twice in one func.return (e.g. `return %s, %s` for a
            // DS-tracked %s), the first occurrence is replaced with its raw
            // `hi` component and the second with its raw `lo` component --
            // skipping emitToFloat's recombination entirely for that pair
            // of operands -- so the caller can recombine in f64 on the host
            // instead of losing precision to an f32 return (see
            // ds_reeval/exp3_pair_accuracy.py and
            // ds_reeval/test_return_pairs_structural.py).
            //
            // This only ever substitutes a value of the *same type* as the
            // operand it replaces (guarded by `orig.getType() ==
            // hi.getType()`, i.e. the operand's declared return type must
            // already be f32, which hi/lo always are) -- so the FuncOp's
            // result-type signature never needs to change, and there is no
            // risk of emitting ill-typed IR or changing the function's
            // arity as JAX originally traced it. f64-typed returns (hi/lo
            // are f32 but the declared return type is f64) always fall
            // back to normal recombination below, since substituting an
            // f32 value for an f64 result would be ill-typed; extending
            // DS_RETURN_PAIRS to that case would additionally require
            // updating func.getFunctionType(), which is out of scope here.
            // A value returned only once, or a third+ occurrence of the
            // same value, is unaffected either way (falls through to
            // ordinary recombination) -- this deliberately only special-
            // cases the exact doubled-return pattern above.
            if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
                OpBuilder rb(retOp);
                llvm::DenseMap<Value, int> seen;
                for (auto& operand : retOp->getOpOperands()) {
                    Value orig = operand.get();
                    if (!dsMap.count(orig)) continue;
                    auto [hi, lo] = dsMap[orig];

                    bool substitutedPair = false;
                    if (returnPairs && orig.getType() == hi.getType()) {
                        int idx = seen[orig];
                        seen[orig] = idx + 1;
                        if (idx == 0) { operand.set(hi); substitutedPair = true; }
                        else if (idx == 1) { operand.set(lo); substitutedPair = true; }
                    }
                    if (!substitutedPair) {
                        Value combined = emitToFloat(rb, loc, hi, lo, orig.getType());
                        operand.set(combined);
                    }
                }
                continue;
            }

            // ── DS_WARN_UNSUPPORTED=1 diagnostic ──────────────────────────
            // Reached only if none of the handlers above matched this op.
            // If it produces a float-typed result from a DS-tracked
            // operand, DS precision silently reverts to native f32 from
            // this point on -- report it if the human opted in. No
            // "continue" needed: this is the last check in the loop body.
            if (warnUnsupported) {
                bool hasFloatResult = false;
                for (Value result : op->getResults()) {
                    if (isFloatTensor(result)) { hasFloatResult = true; break; }
                }
                bool hasDsTrackedOperand = false;
                for (Value operand : op->getOperands()) {
                    if (dsMap.count(operand)) { hasDsTrackedOperand = true; break; }
                }
                if (hasFloatResult && hasDsTrackedOperand) {
                    llvm::errs() << "[ds-transform] WARNING: unsupported op '"
                                 << op->getName() << "' at " << op->getLoc()
                                 << " consumes a DS-tracked operand but is not "
                                 << "transformed -- downstream DS precision "
                                 << "degrades to native f32 from this point.\n";
                }
            }
        }

        // Erase replaced ops in reverse order (same as Skeleton.cpp)
        for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
            if ((*it)->use_empty())
                (*it)->erase();
    }

    // ── Pass entry point ──────────────────────────────────────────────────
    void runOnOperation() override {
        func::FuncOp func = getOperation();
        dsMap.clear();
        const char* rp = std::getenv("DS_RETURN_PAIRS");
        returnPairs = rp && std::string(rp) == "1";
        const char* wu = std::getenv("DS_WARN_UNSUPPORTED");
        warnUnsupported = wu && std::string(wu) == "1";
        convertFuncArgs(func);
        processOps(func);
    }
};

} // namespace

void registerDsTransformPass() {
    mlir::PassRegistration<DsTransformPass>();
}

// ── Plugin registration ───────────────────────────────────────────────────────

extern "C" MLIR_PLUGIN_API_EXPORT ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "DsTransformPass",
        "v0.1",
        []() {
            ::mlir::PassRegistration<DsTransformPass>();
        }
    };
}

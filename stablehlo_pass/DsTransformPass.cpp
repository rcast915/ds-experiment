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
//   stablehlo.exponential → ds_exp (ported from double-single-libm's expds)
//   stablehlo.log       → ds_log  (ported from double-single-libm's logds)
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
// Ops in this table other than add/sub/mul/div/sqrt/exp/log/negate/abs/
// compare/select/maximum/minimum, dot_general, and reduce are NOT
// DS-transformed:
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
// emitDsDiv's comment for why, and what's substituted instead. divide's
// emitDsDiv also includes one deliberate correction beyond a literal
// port (a dropped TwoProd residual term, confirmed to be a genuine bug
// in the reference rather than a design choice) -- see its comment.
//
// exp/log ported from double-single-libm (expds in exp.c, logds in log.c);
// their lookup tables are compiled in from libmds/exp_table.h and
// libmds/log_table.h, copied unmodified from that library. See emitDsExp
// and emitDsLog for how the library's branches, bit manipulation, and
// table reads are expressed as StableHLO ops.
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

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

// double-single-libm's table headers only need this typedef (libmds.h).
typedef float __libmds_binary32_t;
#include "libmds/exp_table.h"
#include "libmds/log_table.h"

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

// True if v is a constant, or a convert/broadcast of one -- i.e. something
// XLA's constant folding will turn into a plain constant.
static bool isConstantLike(Value v) {
    while (Operation* def = v.getDefiningOp()) {
        if (isa<stablehlo::ConstantOp>(def)) return true;
        if (!isa<stablehlo::ConvertOp>(def) &&
            !isa<stablehlo::BroadcastInDimOp>(def)) return false;
        v = def->getOperand(0);
    }
    return false;
}

// two_sum(a, b) → (s, e)
//   s  = a + b
//   bb = s - a
//   e  = (a - (s - bb)) + (b - bb)
//
// If a is a constant and b is not, the operands are swapped first. TwoSum
// is symmetric and error-free, so (s, e) is the same either way, but
// `(b + C) - C` is a shape XLA's algebraic simplifier may reassociate to
// `b + (C - C)`, which would replace bb by b and lose the rounding error
// this sequence exists to capture. Subtracting the non-constant operand
// keeps the sequence out of that pattern.
static std::pair<Value, Value> emitTwoSum(OpBuilder& bld, Location loc,
                                           Value a, Value b) {
    if (isConstantLike(a) && !isConstantLike(b)) std::swap(a, b);
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
// Based on double-single-lib's double_binary32_div, WITH ONE DELIBERATE
// CORRECTION (see below) to a confirmed bug in the reference:
//   t1 = a_hi / b_hi
//   (t2, t3) = two_mul(b_hi, t1)
//   t4 = b_lo * t1
//   t5 = a_hi - t2        [Sterbenz: exact -- t2 is close enough to a_hi
//                           that this subtraction introduces no rounding]
//   t6 = a_lo - t4
//   t7 = (t5 + t6) - t3   [see note: library's literal t7 = t5 + t6 drops t3]
//   t8 = t7 / b_hi
//   (out_hi, out_lo) = fast_two_sum(t1, t8)
//
// DEVIATION FROM THE LIBRARY, deliberate: as literally written,
// double_binary32_div computes t7 = t5 + t6, never using t3 (the
// TwoProd/two_mul residual of b_hi*t1) again after computing it.
// Re-deriving the correction algebraically: for a/b = t1 + delta, the
// exact identity a_hi - t1*b_hi = t5 - t3 shows the correction sum needs
// `t5 - t3 + a_lo - t4`, not `t5 + a_lo - t4`. Confirmed this is a
// genuine omission and not an intentional design choice: the library's
// OWN sibling routine for "divide a DS pair by a plain scalar",
// __double_binary_div_double_by_single (see emitDsDivByScalar below,
// used by sqrt), computes the analogous term as `al - t3` -- i.e. it
// correctly keeps the residual this routine drops. And it is not an
// adversarial-input-only corner case: t3 captures t1's own
// single-precision rounding error (t1 = a_hi/b_hi is only
// correctly-rounded, not exact), so it matters regardless of b_lo --
// empirically, the literal library formula costs up to ~1.16e-7 relative
// error (~2^-23, f32-ULP level) on EVERY division through this routine,
// including plain lo=0/lo=0 cases with no cancellation at all, instead of
// the ~2^-48-class accuracy this algorithm's structure is designed to
// reach. Including the `- t3` term above restores that accuracy
// (confirmed empirically: worst case ~1.7e-14 over 500k random trials,
// vs. add/sub/mul/sqrt's established double-word precision class) at the
// cost of one extra stablehlo.subtract op. This correction was an
// explicit choice, not the port-the-reference-exactly default this
// project otherwise follows -- flagging it here since it's the one place
// this port intentionally diverges from what the source literally says
// beyond the unavoidable FMA substitution below.
//
// Deviation from the library, deliberate (separately, and unavoidable):
// the library's __two_mul uses a real hardware FMA
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
// (which would round differently). Sterbenz-style subtractions are an
// elevated-simplifier-risk sequence, same category as the TwoSum residual
// chain.
static std::pair<Value, Value> emitDsDiv(OpBuilder& bld, Location loc,
                                          Value a_hi, Value a_lo,
                                          Value b_hi, Value b_lo) {
    Value t1 = bld.create<stablehlo::DivOp>(loc, a_hi, b_hi);
    auto [t2, t3] = emitTwoProd(bld, loc, b_hi, t1);
    Value t4 = bld.create<stablehlo::MulOp>(loc, b_lo, t1);
    Value t5 = bld.create<stablehlo::SubtractOp>(loc, a_hi, t2);
    Value t6 = bld.create<stablehlo::SubtractOp>(loc, a_lo, t4);
    Value t6sum = bld.create<stablehlo::AddOp>(loc, t5, t6);
    Value t7 = bld.create<stablehlo::SubtractOp>(loc, t6sum, t3);  // correction: see comment above
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
// NOTE: kept as its own function rather than calling emitDsDiv(a_hi,
// a_lo, b, 0) with a zero lo operand, for two reasons. First, fidelity:
// __double_binary_div_double_by_single is what double_binary32_sqrt
// actually calls in the reference source, not double_binary32_div with a
// zero lo -- porting the function that's actually called is more
// faithful even where the two happen to be numerically close. Second,
// even after emitDsDiv's own `-t3` correction (see its comment) makes
// the two algebraically equivalent when b_lo=0 -- both reduce to
// `t5 + a_lo - t3` as real numbers -- they are NOT bit-identical:
// emitDsDiv computes it as `(t5 + a_lo) - t3` (add then subtract) while
// this routine computes `t5 + (a_lo - t3)` (subtract then add), and
// floating-point addition/subtraction is not associative, so the two
// orderings can round differently in the last bit. Keeping this as a
// direct, separate port avoids relying on that coincidental equivalence.
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

// ── exp / log (ported from double-single-libm) ───────────────────────────────
//
// The helpers below mirror double-single-libm's libmds_internal.h building
// blocks one-for-one (the _2_xy suffix is the library's own naming: result
// is a double-word, operands are single (1) or double (2) words). They are
// kept separate from emitDsAdd/emitDsMul above, which implement different
// sequences, so that emitDsExp/emitDsLog evaluate exactly what expds/logds
// evaluate.
//
// Constant operands follow the same rule as emitTwoSum: a sequence of the
// form `(x + C) - C` is never emitted, since XLA may fold it to x. Where
// the library has one (the shifter trick's `k = ssxh - SHIFTER`, and a
// fast_two_sum whose leading operand is a polynomial coefficient), an
// equivalent that computes the same value is used and noted at that step.
//
// Three library mechanisms have no direct StableHLO equivalent:
//   - __libmds_mul_2_11 uses a hardware FMA. emitTwoProd (Veltkamp split)
//     is substituted, for the reason given in emitDsDiv's comment. Both
//     produce the exact product error, and every product in exp/log has
//     operands of bounded magnitude, so the split cannot overflow.
//   - Data-dependent branches (special cases, log's zh >= sqrt(2) step)
//     become stablehlo.select: both arms are computed for every element
//     and the special-case result overrides the main path at the end.
//   - Float/int reinterpretation and table reads become
//     stablehlo.bitcast_convert, integer shift/and ops on i32 tensors, and
//     stablehlo.gather from a constant 1-D table.

// __libmds_add_2_12: a + (bh, bl)
static std::pair<Value, Value> emitAdd212(OpBuilder& bld, Location loc,
                                           Value a, Value bh, Value bl) {
    auto [t1h, t1l] = emitTwoSum(bld, loc, a, bh);
    Value t2 = bld.create<stablehlo::AddOp>(loc, t1l, bl);
    return emitFastTwoSum(bld, loc, t1h, t2);
}

// __libmds_add_2_22: (ah, al) + (bh, bl)
static std::pair<Value, Value> emitAdd222(OpBuilder& bld, Location loc,
                                           Value ah, Value al,
                                           Value bh, Value bl) {
    auto [t1h, t1l] = emitTwoSum(bld, loc, ah, bh);
    Value t2 = bld.create<stablehlo::AddOp>(loc, al, bl);
    Value t3 = bld.create<stablehlo::AddOp>(loc, t1l, t2);
    return emitFastTwoSum(bld, loc, t1h, t3);
}

// __libmds_mul_2_12: a * (bh, bl)
static std::pair<Value, Value> emitMul212(OpBuilder& bld, Location loc,
                                           Value a, Value bh, Value bl) {
    auto [t1h, t1l] = emitTwoProd(bld, loc, a, bh);
    Value t2 = bld.create<stablehlo::MulOp>(loc, a, bl);
    Value t3 = bld.create<stablehlo::AddOp>(loc, t1l, t2);
    return emitFastTwoSum(bld, loc, t1h, t3);
}

// __libmds_mul_2_22: (ah, al) * (bh, bl)
static std::pair<Value, Value> emitMul222(OpBuilder& bld, Location loc,
                                           Value ah, Value al,
                                           Value bh, Value bl) {
    auto [t1h, t1l] = emitTwoProd(bld, loc, ah, bh);
    Value t2 = bld.create<stablehlo::MulOp>(loc, ah, bl);
    Value t3 = bld.create<stablehlo::MulOp>(loc, al, bh);
    Value t4 = bld.create<stablehlo::AddOp>(loc, t2, t3);
    Value t5 = bld.create<stablehlo::AddOp>(loc, t4, t1l);
    return emitTwoSum(bld, loc, t1h, t5);
}

// The i32 tensor type with the same shape as a float tensor value.
static RankedTensorType toI32Type(Value ref) {
    auto ty = cast<RankedTensorType>(ref.getType());
    return RankedTensorType::get(ty.getShape(),
                                 IntegerType::get(ty.getContext(), 32));
}

// emitISplat: an i32 constant with every element equal to `scalar`.
static Value emitISplat(OpBuilder& bld, Location loc, RankedTensorType i32Ty,
                        int32_t scalar) {
    auto attr = DenseElementsAttr::get(
        i32Ty, APInt(32, static_cast<uint64_t>(static_cast<int64_t>(scalar)),
                     /*isSigned=*/true));
    return bld.create<stablehlo::ConstantOp>(loc, attr);
}

// __libmds_two_pow_normal: 2^k as f32, for an i32 tensor k in the normal
// exponent range, built as the bit pattern (127 + k) << 23.
static Value emitTwoPow(OpBuilder& bld, Location loc, Value k, Type f32Ty) {
    auto i32Ty = cast<RankedTensorType>(k.getType());
    Value biased = bld.create<stablehlo::AddOp>(loc,
                       emitISplat(bld, loc, i32Ty, 127), k);
    Value bits = bld.create<stablehlo::ShiftLeftOp>(loc, biased,
                     emitISplat(bld, loc, i32Ty, 23));
    return bld.create<stablehlo::BitcastConvertOp>(loc, f32Ty, bits);
}

// The biased exponent field of an f32 tensor, minus the bias, as i32.
static Value emitExponentField(OpBuilder& bld, Location loc, Value x) {
    auto i32Ty = toI32Type(x);
    Value bits = bld.create<stablehlo::BitcastConvertOp>(loc, i32Ty, x);
    Value shifted = bld.create<stablehlo::ShiftRightLogicalOp>(loc, bits,
                        emitISplat(bld, loc, i32Ty, 23));
    Value field = bld.create<stablehlo::AndOp>(loc, shifted,
                      emitISplat(bld, loc, i32Ty, 0xff));
    return bld.create<stablehlo::SubtractOp>(loc, field,
               emitISplat(bld, loc, i32Ty, 127));
}

// Split an i32 tensor n into (n >> 1, n - (n >> 1)), so that 2^n can be
// applied as two multiplications by normal powers of two.
static std::pair<Value, Value> emitHalveExponent(OpBuilder& bld, Location loc,
                                                  Value n) {
    auto i32Ty = cast<RankedTensorType>(n.getType());
    Value n1 = bld.create<stablehlo::ShiftRightArithmeticOp>(loc, n,
                   emitISplat(bld, loc, i32Ty, 1));
    Value n2 = bld.create<stablehlo::SubtractOp>(loc, n, n1);
    return {n1, n2};
}

// table[idx], elementwise: `idx` is an i32 tensor of any static shape and
// the result is an f32 tensor of the same shape. Emitted as a
// stablehlo.gather of size-1 slices from a rank-1 constant, with the
// index-vector dimension implied (index_vector_dim == rank(idx)), so no
// reshape of idx is needed.
static Value emitTableLookup(OpBuilder& bld, Location loc,
                              ArrayRef<float> table, Value idx, Type f32Ty) {
    auto idxTy = cast<RankedTensorType>(idx.getType());
    const int64_t tableShape[] = {static_cast<int64_t>(table.size())};
    auto tableTy = RankedTensorType::get(
        tableShape, Float32Type::get(idxTy.getContext()));
    Value tableCst = bld.create<stablehlo::ConstantOp>(
        loc, DenseElementsAttr::get(tableTy, table));
    const int64_t dim0[] = {0};
    const int64_t sliceSizes[] = {1};
    ArrayRef<int64_t> none;
    auto dimNumbers = stablehlo::GatherDimensionNumbersAttr::get(
        idxTy.getContext(),
        /*offsetDims=*/none,
        /*collapsedSliceDims=*/dim0,
        /*operandBatchingDims=*/none,
        /*startIndicesBatchingDims=*/none,
        /*startIndexMap=*/dim0,
        /*indexVectorDim=*/idxTy.getRank());
    return bld.create<stablehlo::GatherOp>(
        loc, f32Ty, tableCst, idx, dimNumbers,
        bld.getDenseI64ArrayAttr(sliceSizes), bld.getBoolAttr(false));
}

// ds_exp((x_hi, x_lo)) → (out_hi, out_lo)
// Ported from double-single-libm's expds (exp.c); the step comments use
// that file's variable names. Outline:
//   k    = nearestint(x_hi * 2^8/log(2))            [shifter trick]
//   n    = floor(k / 2^8),  idx = k - n * 2^8        [0 <= idx <= 255]
//   r    = x - k * 2^-8 * log(2)                     [log(2) in 6 chunks]
//   p    = polynomial approximating e^rh             [degree 4]
//   w    = table[idx] * p * (1 + rl)                 [table = 2^(idx/2^8)]
//   out  = 2^n * w                                   [two scalings]
//
// Special cases, in the library's priority order, each returning the same
// value in both components: NaN → NaN, +Inf → +Inf, -Inf → 0, sure
// overflow → +Inf, sure complete underflow → 0. The library tests xh, xl,
// and xh + xl for NaN separately; the sum alone is tested here, since it
// is NaN whenever either word is.
static std::pair<Value, Value> emitDsExp(OpBuilder& bld, Location loc,
                                          Value xh, Value xl) {
    using CD = stablehlo::ComparisonDirection;
    Type f32Ty = xh.getType();
    auto i32Ty = toI32Type(xh);
    auto F = [&](float v) { return emitSplat(bld, loc, xh, v); };
    auto I = [&](int32_t v) { return emitISplat(bld, loc, i32Ty, v); };
    auto add = [&](Value a, Value b) -> Value {
        return bld.create<stablehlo::AddOp>(loc, a, b); };
    auto sub = [&](Value a, Value b) -> Value {
        return bld.create<stablehlo::SubtractOp>(loc, a, b); };
    auto mul = [&](Value a, Value b) -> Value {
        return bld.create<stablehlo::MulOp>(loc, a, b); };
    auto cmp = [&](Value a, Value b, CD dir) -> Value {
        return bld.create<stablehlo::CompareOp>(loc, a, b, dir); };
    auto lor = [&](Value a, Value b) -> Value {
        return bld.create<stablehlo::OrOp>(loc, a, b); };
    auto land = [&](Value a, Value b) -> Value {
        return bld.create<stablehlo::AndOp>(loc, a, b); };
    auto sel = [&](Value c, Value a, Value b) -> Value {
        return bld.create<stablehlo::SelectOp>(loc, c, a, b); };

    // kint = nearestint(xh * 2^8/log(2)): the low 18 bits of the shifted
    // sum's significand, sign-extended. The library also forms the same
    // integer as a float, k = ssxh - SHIFTER; here k is converted from
    // kint instead, which is exact and gives the identical value.
    Value ssxh = add(mul(xh, F(0x1.715476p8f)), F(0x1.8p23f));
    Value ssxhbits = bld.create<stablehlo::BitcastConvertOp>(loc, i32Ty, ssxh);
    Value kint = bld.create<stablehlo::ShiftRightArithmeticOp>(loc,
                     bld.create<stablehlo::ShiftLeftOp>(loc, ssxhbits, I(14)),
                     I(14));
    Value k   = bld.create<stablehlo::ConvertOp>(loc, f32Ty, kint);
    Value n   = bld.create<stablehlo::ShiftRightArithmeticOp>(loc, kint, I(8));
    Value idx = sub(kint, bld.create<stablehlo::ShiftLeftOp>(loc, n, I(8)));

    // rh + rl ~= xh - k * 2^-8 * log(2) + xl. Each chunk of -2^-8 * log(2)
    // has 8 significant bits, so every k * chunk product is exact.
    static const float chunks[6] = {
        -0x1.62p-9f, -0x1.c8p-18f, -0x1.8p-28f,
        0x1.06p-37f, -0x1.dp-48f, 0x1.0cp-57f};
    Value rth = add(xh, mul(k, F(chunks[0])));   // rt1, Sterbenz
    Value rtl[5];                                // rt2l .. rt6l
    for (int i = 1; i < 6; ++i) {
        auto [h, l] = emitTwoSum(bld, loc, rth, mul(k, F(chunks[i])));
        rth = h;
        rtl[i - 1] = l;
    }
    Value rt7 = add(add(add(add(add(rtl[4], rtl[3]), rtl[2]), rtl[1]),
                        rtl[0]), xl);
    auto [rh, rl] = emitTwoSum(bld, loc, rth, rt7);

    // ph + pl ~= e^rh. The library's first step is fast_two_sum(c2, q);
    // emitTwoSum returns the same pair without subtracting the constant.
    Value q = mul(rh, add(F(0x1.555558p-3f), mul(rh, F(0x1.55555cp-5f))));
    auto [pt1h, pt1l] = emitTwoSum(bld, loc, F(0x1p-1f), q);
    auto [m2h, m2l]   = emitMul212(bld, loc, rh, pt1h, pt1l);
    auto [pt2h, pt2l] = emitAdd212(bld, loc, F(1.0f), m2h, m2l);
    auto [m1h, m1l]   = emitMul212(bld, loc, rh, pt2h, pt2l);
    auto [ph, pl]     = emitAdd212(bld, loc, F(1.0f), m1h, m1l);

    // th + tl = 2^(idx * 2^-8)
    Value th = emitTableLookup(bld, loc, __expds_table_hi, idx, f32Ty);
    Value tl = emitTableLookup(bld, loc, __expds_table_lo, idx, f32Ty);

    // wh + wl = (th + tl) * ((ph + pl) + (ph + pl) * rl)
    auto [qh, ql] = emitMul212(bld, loc, rl, ph, pl);
    auto [zh, zl] = emitAdd222(bld, loc, ph, pl, qh, ql);
    auto [wh, wl] = emitMul222(bld, loc, th, tl, zh, zl);

    // 2^n * (wh + wl), as s1 * (s2 * w) so that neither power of two
    // leaves the normal range.
    auto [n1, n2] = emitHalveExponent(bld, loc, n);
    Value s1 = emitTwoPow(bld, loc, n1, f32Ty);
    Value s2 = emitTwoPow(bld, loc, n2, f32Ty);
    Value resh = mul(s1, mul(s2, wh));
    Value resl = mul(s1, mul(s2, wl));

    // Special cases.
    Value sum   = add(xh, xl);
    Value omega = F(0x1.fffffep127f);
    Value zero  = F(0.0f);
    Value isNaN    = cmp(sum, sum, CD::NE);
    Value isPosInf = cmp(sum, omega, CD::GT);
    Value isNegInf = cmp(sum, F(-0x1.fffffep127f), CD::LT);
    Value ovHi = F(0x1.62e43p6f);
    Value unHi = F(-0x1.9fe36ap6f);
    Value overflow = lor(cmp(xh, ovHi, CD::GT),
                         land(cmp(xh, ovHi, CD::EQ),
                              cmp(xl, F(-0x1.25c612p-22f), CD::GT)));
    Value underflow = lor(cmp(xh, unHi, CD::LT),
                          land(cmp(xh, unHi, CD::EQ),
                               cmp(xl, F(0x1.d32c42p-18f), CD::LT)));
    Value passThrough = lor(isNaN, isPosInf);    // result is xh + xl itself
    Value special = sel(passThrough, sum,
                        sel(isNegInf, zero,
                            sel(overflow,
                                F(std::numeric_limits<float>::infinity()),
                                zero)));
    Value isSpecial = lor(lor(passThrough, isNegInf),
                          lor(overflow, underflow));
    return {sel(isSpecial, special, resh), sel(isSpecial, special, resl)};
}

// ds_log((x_hi, x_lo)) → (out_hi, out_lo)
// Ported from double-single-libm's logds (log.c); the step comments use
// that file's variable names. Outline:
//   E      = floor(log2(x_hi)), then z = x * 2^-E, halved (and E
//            incremented) if z_hi >= sqrt(2), so sqrt(2)/2 < z_hi < sqrt(2)
//   i      = nearestint(2^8 * z_hi) - 181              [0 <= i <= 181]
//   r      = w[i] * z - 1,  w[i] ~= 1/z_hi             [|r| <= 2^-8.496]
//   out    = E * log(2) + (-log(w[i])) + p(r)          [p ~= log(1 + r)]
//
// Special cases, in the library's priority order: NaN → NaN, +Inf → +Inf,
// -Inf → NaN, x_hi < 0 → NaN (each in both components), x_hi == 0 →
// (-Inf, 0).
//
// DEVIATION FROM THE LIBRARY, deliberate: for x_hi == 0, logds's comment
// says "Return -inf" but its code computes 1.0f / (0.0f * 0.0f), which is
// +Inf. log(0) is -Inf, so -Inf is what this emits.
//
// Two mechanical differences that do not change any computed value: E is
// converted to float with stablehlo.convert instead of the library's
// bit-pattern trick (both are exact for |E| <= 150), and the table index
// is clamped to 181, since inputs that take a special case still evaluate
// the main path here and could otherwise index past the 182-entry tables.
static std::pair<Value, Value> emitDsLog(OpBuilder& bld, Location loc,
                                          Value xh, Value xl) {
    using CD = stablehlo::ComparisonDirection;
    Type f32Ty = xh.getType();
    auto i32Ty = toI32Type(xh);
    auto F = [&](float v) { return emitSplat(bld, loc, xh, v); };
    auto I = [&](int32_t v) { return emitISplat(bld, loc, i32Ty, v); };
    auto add = [&](Value a, Value b) -> Value {
        return bld.create<stablehlo::AddOp>(loc, a, b); };
    auto mul = [&](Value a, Value b) -> Value {
        return bld.create<stablehlo::MulOp>(loc, a, b); };
    auto cmp = [&](Value a, Value b, CD dir) -> Value {
        return bld.create<stablehlo::CompareOp>(loc, a, b, dir); };
    auto lor = [&](Value a, Value b) -> Value {
        return bld.create<stablehlo::OrOp>(loc, a, b); };
    auto sel = [&](Value c, Value a, Value b) -> Value {
        return bld.create<stablehlo::SelectOp>(loc, c, a, b); };

    // E = floor(log2(xh)) (__libmds_logb_finite_non_zero). xh may be
    // subnormal, so xh is first scaled by 2^-E0 using its raw exponent
    // field E0, and the exponent of the scaled value is added back.
    Value e0 = emitExponentField(bld, loc, xh);
    auto [g1, g2] = emitHalveExponent(bld, loc,
                        bld.create<stablehlo::NegOp>(loc, e0));
    Value ssx = mul(emitTwoPow(bld, loc, g2, f32Ty),
                    mul(emitTwoPow(bld, loc, g1, f32Ty), xh));
    Value E = add(e0, emitExponentField(bld, loc, ssx));

    // zh + zl = 2^-E * (xh + xl), with 1 <= zh < 2.
    auto [f1, f2] = emitHalveExponent(bld, loc,
                        bld.create<stablehlo::NegOp>(loc, E));
    Value s1 = emitTwoPow(bld, loc, f1, f32Ty);
    Value s2 = emitTwoPow(bld, loc, f2, f32Ty);
    Value zh = mul(s1, mul(s2, xh));
    Value zl = mul(s1, mul(s2, xl));

    // If zh >= sqrt(2): halve z and increment E.
    Value half = F(0.5f);
    Value upper = cmp(zh, F(0x1.6a09e8p0f), CD::GE);
    zh = sel(upper, mul(zh, half), zh);
    zl = sel(upper, mul(zl, half), zl);
    E  = sel(upper, add(E, I(1)), E);

    // i = nearestint(2^8 * zh) - 181, read from the low byte of the
    // shifted sum's bit pattern.
    Value sh  = add(mul(F(0x1.0p8f), zh), F(0x1.7ffe96p23f));
    Value shb = bld.create<stablehlo::BitcastConvertOp>(loc, i32Ty, sh);
    Value i   = bld.create<stablehlo::MinOp>(loc,
                    bld.create<stablehlo::AndOp>(loc, shb, I(0xff)), I(181));

    // rh + rl = w * (zh + zl) - 1
    Value w = emitTableLookup(bld, loc, __logds_table_rcpr_z, i, f32Ty);
    auto [wzh, wzl] = emitMul212(bld, loc, w, zh, zl);
    auto [rh, rl]   = emitAdd212(bld, loc, F(-1.0f), wzh, wzl);

    // th + tl ~= -log(w)
    Value th = emitTableLookup(bld, loc, __logds_table_m_log_w_hi, i, f32Ty);
    Value tl = emitTableLookup(bld, loc, __logds_table_m_log_w_lo, i, f32Ty);

    // elh + ell = E * log(2), with log(2) in three chunks whose products
    // with e are exact.
    Value e = bld.create<stablehlo::ConvertOp>(loc, f32Ty, E);
    Value elrh = mul(e, F(0x1.62e4p-1f));
    Value elrm = mul(e, F(0x1.7f7cp-20f));
    Value elrl = mul(e, F(0x1.1cf8p-36f));
    auto [telh, tell] = emitFastTwoSum(bld, loc, elrh, elrm);
    auto [elh, ell]   = emitFastTwoSum(bld, loc, telh, add(tell, elrl));

    // ph + pl ~= log(1 + rh + rl): degree-5 Horner scheme in
    // double-single, zero constant term, double-single coefficients for
    // degrees 1 and 2.
    auto [m4h, m4l] = emitMul212(bld, loc, F(0x1.970e1cp-3f), rh, rl);
    auto [q4h, q4l] = emitAdd212(bld, loc, F(-0x1.000068p-2f), m4h, m4l);
    auto [m3h, m3l] = emitMul222(bld, loc, rh, rl, q4h, q4l);
    auto [q3h, q3l] = emitAdd212(bld, loc, F(0x1.555556p-2f), m3h, m3l);
    auto [m2h, m2l] = emitMul222(bld, loc, rh, rl, q3h, q3l);
    auto [q2h, q2l] = emitAdd222(bld, loc, F(-0x1p-1f), F(0x1.95ep-39f),
                                 m2h, m2l);
    auto [m1h, m1l] = emitMul222(bld, loc, rh, rl, q2h, q2l);
    auto [q1h, q1l] = emitAdd222(bld, loc, F(0x1p0f), F(-0x1.8p-47f),
                                 m1h, m1l);
    auto [ph, pl]   = emitMul222(bld, loc, rh, rl, q1h, q1l);

    // (elh + ell) + ((th + tl) + (ph + pl))
    auto [gh, gl]     = emitAdd222(bld, loc, th, tl, ph, pl);
    auto [resh, resl] = emitAdd222(bld, loc, elh, ell, gh, gl);

    // Special cases.
    Value sum  = add(xh, xl);
    Value zero = F(0.0f);
    Value nan  = F(std::numeric_limits<float>::quiet_NaN());
    Value passThrough = lor(cmp(sum, sum, CD::NE),               // NaN
                            cmp(sum, F(0x1.fffffep127f), CD::GT)); // +Inf
    Value invalid = lor(cmp(sum, F(-0x1.fffffep127f), CD::LT),   // -Inf
                        cmp(xh, zero, CD::LT));                  // negative
    Value isZero  = cmp(xh, zero, CD::EQ);
    Value specialHi = sel(passThrough, sum,
                          sel(invalid, nan,
                              F(-std::numeric_limits<float>::infinity())));
    Value specialLo = sel(passThrough, sum, sel(invalid, nan, zero));
    Value isSpecial = lor(lor(passThrough, invalid), isZero);
    return {sel(isSpecial, specialHi, resh), sel(isSpecial, specialLo, resl)};
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
    //
    // hi is the narrowed value BEHIND an optimization barrier, and every
    // use of hi goes through that barrier. XLA folds
    // convert(convert(v, f32), f64) straight back to v -- narrow-then-widen
    // treated as an identity, which it is not unless v already fits in
    // f32. The fold is a peephole on the convert-feeds-convert adjacency
    // itself, so the barrier has to sit between the two converts; one
    // placed after the widening convert is too late.
    //
    // Two places widen hi back to f64, and both need the barrier:
    //   - here, for lo = v - f64(hi). Folded, this becomes v - v = 0.
    //   - emitToFloat at func.return, for f64(hi) + f64(lo). Folded, this
    //     becomes v + f64(lo): the result is off by lo, an f32-level error.
    //     Observed on GPU when only the first use was barriered: `x + 0.0`
    //     on f64 inputs came back accurate to 2^-24, and the small helper
    //     modules JAX compiles ahead of a jitted function (which the plugin
    //     also transforms) shifted their f64 outputs by lo before the
    //     user's function ever ran.
    Value hi_raw = bld.create<stablehlo::ConvertOp>(loc, f32Ty, v);
    Value hi = bld.create<stablehlo::OptimizationBarrierOp>(
        loc, TypeRange{f32Ty}, ValueRange{hi_raw})->getResult(0);
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
            // Based on double-single-lib's double_binary32_div -- see
            // emitDsDiv for the exact sequence, the FMA/two_mul
            // substitution note, and the deliberate `-t3` correction (one
            // intentional deviation from the literal library, fixing a
            // confirmed bug there). No special-casing for b_hi == 0 or
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

            // ── stablehlo.exponential ──────────────────────────────────────
            // Ported from double-single-libm's expds -- see emitDsExp for
            // the sequence and its special-case handling.
            if (auto expOp = dyn_cast<stablehlo::ExpOp>(op)) {
                if (!isFloatTensor(expOp.getResult())) continue;
                if (!dsMap.count(expOp.getOperand())) continue;

                auto [a_hi, a_lo] = dsMap[expOp.getOperand()];
                auto [r_hi, r_lo] = emitDsExp(b, loc, a_hi, a_lo);
                dsMap[expOp.getResult()] = {r_hi, r_lo};
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.log ──────────────────────────────────────────────
            // Ported from double-single-libm's logds -- see emitDsLog for
            // the sequence, its special-case handling, and the one
            // deliberate deviation (log(0) is -Inf here).
            if (auto logOp = dyn_cast<stablehlo::LogOp>(op)) {
                if (!isFloatTensor(logOp.getResult())) continue;
                if (!dsMap.count(logOp.getOperand())) continue;

                auto [a_hi, a_lo] = dsMap[logOp.getOperand()];
                auto [r_hi, r_lo] = emitDsLog(b, loc, a_hi, a_lo);
                dsMap[logOp.getResult()] = {r_hi, r_lo};
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
            // instead of losing precision to an f32 return (see the
            // pair-accuracy sections of tests/test_ds_divide.py and
            // tests/test_ds_sqrt.py).
            //
            // Two return-type cases, both handled without ever changing the
            // FuncOp's result-type signature or the function's arity as JAX
            // originally traced it (no risk of ill-typed IR either way):
            //   - f32-typed return (orig.getType() == hi.getType(), the
            //     common case, e.g. an f32-input reduction):
            //     substitute hi/lo directly -- they're already the right type.
            //   - f64-typed return (hi/lo are f32 but the declared return
            //     type is f64, e.g. f64-input probes): substitute
            //     convert(hi, f64) and
            //     convert(lo, f64) instead -- two separate f64-typed outputs,
            //     each an exact widening of one raw component, rather than
            //     their (lossy-at-output) sum. Added alongside the f64
            //     ingestion-path bisection that needed to observe a raw f64-
            //     sourced DS pair, which the f32-only version of this flag
            //     could not do (it always fell back to full recombination
            //     for f64 returns). This is a test-infrastructure extension
            //     of an existing opt-in diagnostic, not a change to how f64
            //     values are split, combined, or arithmetic'd anywhere else
            //     in this pass -- the DS_RETURN_PAIRS==false path above is
            //     untouched, byte-for-byte, exactly as before.
            //
            // A value returned only once, or a third+ occurrence of the
            // same value, is unaffected either way (falls through to
            // ordinary recombination) -- this deliberately only special-
            // cases the exact doubled-return pattern, for either type.
            //
            // FIXED BUG (present since DS_RETURN_PAIRS was first added,
            // predating the f64 extension above): substitution used to be
            // decided per-operand, in left-to-right scan order, purely
            // from "is this the 1st or 2nd time I've seen this value SO
            // FAR" -- with no check on how many times the value appears
            // in TOTAL. A value returned exactly ONCE therefore also hit
            // idx==0 on its only occurrence and got silently substituted
            // with hi alone, permanently dropping lo, instead of falling
            // through to ordinary recombination as documented. This was
            // invisible for f32 returns: hi alone and hi+lo rounded back
            // to f32 are typically bit-identical anyway (lo is f32-ULP-
            // scale relative to hi, so plain f32 addition rounds it away
            // regardless -- the same output-quantization effect
            // documented throughout this project, e.g. divide's Section
            // 3a note in test_ds_divide.py). It is NOT invisible for f64:
            // f64 has enough precision to represent lo's contribution
            // exactly, so a single f64 return silently losing lo is a
            // real, measurable correctness bug -- confirmed via a
            // dedicated null test showing a single, non-doubled f64
            // return changed value
            // depending solely on whether DS_RETURN_PAIRS=1 was set,
            // which the documented design says should be impossible.
            // Fixed by counting each value's TOTAL occurrences across the
            // whole return list first, and only ever substituting when
            // that total is exactly 2.
            if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
                OpBuilder rb(retOp);

                llvm::DenseMap<Value, int> totalCount;
                for (auto& operand : retOp->getOpOperands()) {
                    Value orig = operand.get();
                    if (dsMap.count(orig)) totalCount[orig]++;
                }

                llvm::DenseMap<Value, int> seen;
                for (auto& operand : retOp->getOpOperands()) {
                    Value orig = operand.get();
                    if (!dsMap.count(orig)) continue;
                    auto [hi, lo] = dsMap[orig];

                    bool isF32Return = (orig.getType() == hi.getType());
                    bool isF64Return = !isF32Return && isFloatTensor(orig) &&
                        cast<RankedTensorType>(orig.getType()).getElementType().isF64();

                    bool substitutedPair = false;
                    if (returnPairs && (isF32Return || isF64Return) && totalCount[orig] == 2) {
                        int idx = seen[orig];
                        seen[orig] = idx + 1;
                        Value part;
                        if (idx == 0) { part = hi; substitutedPair = true; }
                        else if (idx == 1) { part = lo; substitutedPair = true; }
                        if (substitutedPair) {
                            if (isF64Return) {
                                Value widened = rb.create<stablehlo::ConvertOp>(loc, orig.getType(), part);
                                operand.set(widened);
                            } else {
                                operand.set(part);
                            }
                        }
                    }
                    if (!substitutedPair) {
                        // A function argument returned untouched keeps its
                        // original value: splitting and recombining it
                        // would only round an f64 to DS precision.
                        if (isa<BlockArgument>(orig)) continue;
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

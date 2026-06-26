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
//   func entry args     → split into (hi, lo) via emitFromFloat
//   func return values  → recombined via emitToFloat
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
            if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
                OpBuilder rb(retOp);
                for (auto& operand : retOp->getOpOperands()) {
                    if (!dsMap.count(operand.get())) continue;
                    auto [hi, lo] = dsMap[operand.get()];
                    Value combined = emitToFloat(rb, loc, hi, lo,
                                               operand.get().getType());
                    operand.set(combined);
                }
                continue;
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

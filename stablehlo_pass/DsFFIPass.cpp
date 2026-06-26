//===-- DsFFIPass.cpp -----------------------------------------------------===//
//
// StableHLO pass that replaces float tensor arithmetic with XLA custom calls
// to DS FFI kernels. Analogous to Skeleton.cpp but targeting StableHLO.
//
// Mirrors DsTransformPass.cpp structure but instead of inlining DS math,
// emits stablehlo.custom_call ops that dispatch to external CUDA/CPU kernels.
//
// DS representation: each f32 Value is tracked as a packed tensor<...x2xf32>
// in dsMap where [...,0]=hi and [...,1]=lo.
//
// Transformation table (mirrors Skeleton.cpp):
//   func entry args     → custom_call @ds_from_f32
//   stablehlo.add       → custom_call @ds_add
//   stablehlo.subtract  → custom_call @ds_sub
//   stablehlo.multiply  → custom_call @ds_mul
//   func return values  → custom_call @ds_to_f32
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
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

using namespace mlir;

namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

static bool isF32Tensor(Value v) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    return t && t.getElementType().isF32();
}

// tensor<D0 x D1 x ... x f32> → tensor<D0 x D1 x ... x 2 x f32>
static RankedTensorType toDSType(RankedTensorType t) {
    SmallVector<int64_t> shape(t.getShape());
    shape.push_back(2);
    return RankedTensorType::get(shape, Float32Type::get(t.getContext()));
}

// tensor<D0 x D1 x ... x 2 x f32> → tensor<D0 x D1 x ... x f32>
static RankedTensorType fromDSType(RankedTensorType t) {
    SmallVector<int64_t> shape(t.getShape().drop_back());
    return RankedTensorType::get(shape, Float32Type::get(t.getContext()));
}

// XLA minor-to-major layout for rank R: [R-1, R-2, ..., 0] = row-major.
static DenseIntElementsAttr layoutAttr(MLIRContext* ctx, int64_t rank) {
    SmallVector<int64_t> layout;
    for (int64_t i = rank - 1; i >= 0; --i) layout.push_back(i);
    auto type = RankedTensorType::get({rank}, IndexType::get(ctx));
    return DenseIntElementsAttr::get(type, layout);
}

// Emit a stablehlo.custom_call matching the format JAX's ffi.ffi_call produces.
static Value emitCustomCall(OpBuilder& bld, Location loc,
                             StringRef target,
                             Type resultType,
                             ValueRange operands) {
    MLIRContext* ctx = bld.getContext();

    SmallVector<Attribute> opLayouts, resLayouts;
    for (auto op : operands) {
        auto t = cast<RankedTensorType>(op.getType());
        opLayouts.push_back(layoutAttr(ctx, t.getRank()));
    }
    resLayouts.push_back(layoutAttr(ctx, cast<RankedTensorType>(resultType).getRank()));

    auto callOp = bld.create<stablehlo::CustomCallOp>(
        loc,
        TypeRange{resultType},
        operands,
        bld.getStringAttr(target),
        nullptr,                        // has_side_effect
        bld.getStringAttr(""),          // backend_config
        nullptr,                        // api_version
        nullptr,                        // called_computations
        bld.getArrayAttr(opLayouts),    // operand_layouts
        bld.getArrayAttr(resLayouts),   // result_layouts
        nullptr                         // output_operand_aliases
    );
    return callOp.getResult(0);
}

// ── Pass ─────────────────────────────────────────────────────────────────────

struct DsFFIPass
    : public PassWrapper<DsFFIPass, OperationPass<func::FuncOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DsFFIPass)

    StringRef getArgument()    const override { return "ds-ffi-transform"; }
    StringRef getDescription() const override {
        return "Replace float arithmetic with DS FFI custom calls";
    }

    // Maps original f32 tensor → packed [...,2] f32 DS tensor.
    // Analogous to valueMap in Skeleton.cpp.
    llvm::DenseMap<Value, Value> dsMap;

    // ── Entry: wrap f32 args with ds_from_f32 ────────────────────────────
    void convertFuncArgs(func::FuncOp func) {
        OpBuilder b(&func.getBody().front().front());
        for (auto arg : func.getArguments()) {
            if (!isF32Tensor(arg)) continue;
            auto ty   = cast<RankedTensorType>(arg.getType());
            auto dsTy = toDSType(ty);
            Value packed = emitCustomCall(b, func.getLoc(),
                                          "ds_from_f32", dsTy, {arg});
            dsMap[arg] = packed;
        }
    }

    // ── Core: replace add/sub/mul with custom calls ───────────────────────
    void processOps(func::FuncOp func) {
        // Snapshot first — same trick as Skeleton.cpp
        SmallVector<Operation*> origOps;
        func.walk([&](Operation* op) { origOps.push_back(op); });

        SmallVector<Operation*> toErase;

        for (auto* op : origOps) {
            OpBuilder b(op);
            Location loc = op->getLoc();

            // ── stablehlo.add ─────────────────────────────────────────────
            if (auto addOp = dyn_cast<stablehlo::AddOp>(op)) {
                if (!isF32Tensor(addOp.getResult())) continue;
                if (!dsMap.count(addOp.getLhs()) ||
                    !dsMap.count(addOp.getRhs())) continue;
                Value a   = dsMap[addOp.getLhs()];
                Value bv  = dsMap[addOp.getRhs()];
                Value res = emitCustomCall(b, loc, "ds_add",
                                           a.getType(), {a, bv});
                dsMap[addOp.getResult()] = res;
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.subtract ────────────────────────────────────────
            if (auto subOp = dyn_cast<stablehlo::SubtractOp>(op)) {
                if (!isF32Tensor(subOp.getResult())) continue;
                if (!dsMap.count(subOp.getLhs()) ||
                    !dsMap.count(subOp.getRhs())) continue;
                Value a   = dsMap[subOp.getLhs()];
                Value bv  = dsMap[subOp.getRhs()];
                Value res = emitCustomCall(b, loc, "ds_sub",
                                           a.getType(), {a, bv});
                dsMap[subOp.getResult()] = res;
                toErase.push_back(op);
                continue;
            }

            // ── stablehlo.multiply ────────────────────────────────────────
            if (auto mulOp = dyn_cast<stablehlo::MulOp>(op)) {
                if (!isF32Tensor(mulOp.getResult())) continue;
                if (!dsMap.count(mulOp.getLhs()) ||
                    !dsMap.count(mulOp.getRhs())) continue;
                Value a   = dsMap[mulOp.getLhs()];
                Value bv  = dsMap[mulOp.getRhs()];
                Value res = emitCustomCall(b, loc, "ds_mul",
                                           a.getType(), {a, bv});
                dsMap[mulOp.getResult()] = res;
                toErase.push_back(op);
                continue;
            }

            // ── func.return: recombine DS back to f32 ─────────────────────
            if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
                OpBuilder rb(retOp);
                for (auto& operand : retOp->getOpOperands()) {
                    if (!dsMap.count(operand.get())) continue;
                    Value packed  = dsMap[operand.get()];
                    auto  origTy  = cast<RankedTensorType>(operand.get().getType());
                    Value result  = emitCustomCall(rb, loc,
                                                   "ds_to_f32", origTy, {packed});
                    operand.set(result);
                }
                continue;
            }
        }

        // Erase replaced ops in reverse — same as Skeleton.cpp
        for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
            if ((*it)->use_empty())
                (*it)->erase();
    }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        dsMap.clear();
        convertFuncArgs(func);
        processOps(func);
    }
};

} // namespace

void registerDsFFIPass() {
    mlir::PassRegistration<DsFFIPass>();
}

extern "C" MLIR_PLUGIN_API_EXPORT ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION, "DsFFIPass", "v0.1",
        []() { ::mlir::PassRegistration<DsFFIPass>(); }
    };
}

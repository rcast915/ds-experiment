#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/transforms/Passes.h"

extern void registerDsTransformPass();

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    // registerAllDialects() registers the func dialect itself but not its
    // inliner *extension* -- that's separate, opt-in registration (a
    // DialectInlinerInterface telling the generic `inline` pass how to
    // handle func.call/func.return). Without this, `inline` in a pipeline
    // string silently no-ops on func.call sites rather than erroring,
    // which is why adding `inline` to the ds-transform pipeline didn't
    // actually eliminate the call boundary that breaks dsMap tracking
    // across function calls (e.g. JAX's jnp.where outlining) until this
    // was added.
    mlir::func::registerInlinerExtension(registry);

    mlir::registerAllPasses();
    mlir::stablehlo::registerPasses();
    registerDsTransformPass();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "DS Transform Tool\n", registry));
}

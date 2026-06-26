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

    mlir::registerAllPasses();
    mlir::stablehlo::registerPasses();
    registerDsTransformPass();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "DS Transform Tool\n", registry));
}

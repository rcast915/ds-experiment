#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"

extern void registerDsFFIPass();

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    mlir::registerAllPasses();
    registerDsFFIPass();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "DS FFI Transform Tool\n", registry));
}

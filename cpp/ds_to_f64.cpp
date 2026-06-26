#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template<typename Dims>
static int64_t batchSize(const Dims& dims) {
  int64_t n = 1;
  for (size_t i = 0; i + 1 < dims.size(); ++i) n *= dims[i];
  return n;
}

// Input:  x  shape [N, 2]
// Output: y  shape [N]
ffi::Error DsToF64Impl(ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F64> y) {
  auto x_dims = x.dimensions();
  auto y_dims = y->dimensions();

  if (x_dims.back() != 2) {
    return ffi::Error::InvalidArgument("ds_to_f64 input last dim must be 2");
  }
  if (batchSize(x_dims) != batchSize(y_dims) * y_dims.back()) {
    return ffi::Error::InvalidArgument("ds_to_f64 shape mismatch");
  }

  const int64_t n = batchSize(x_dims);
  const float* in = x.typed_data();
  double* out = y->typed_data();

  for (int64_t i = 0; i < n; ++i) {
    float hi = in[2 * i + 0];
    float lo = in[2 * i + 1];
    out[i] = static_cast<double>(hi) + static_cast<double>(lo);
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DsToF64,
    DsToF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F64>>());

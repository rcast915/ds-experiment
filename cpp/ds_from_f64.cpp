#include <cstdint>
#include <cstring>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template<typename Dims>
static int64_t batchSize(const Dims& dims) {
  int64_t n = 1;
  for (size_t i = 0; i + 1 < dims.size(); ++i) n *= dims[i];
  return n;
}

// Input:  x  shape [N]
// Output: y  shape [N, 2]
ffi::Error DsFromF64Impl(ffi::Buffer<ffi::F64> x,
                         ffi::ResultBuffer<ffi::F32> y) {
  auto x_dims = x.dimensions();
  auto y_dims = y->dimensions();

  if (y_dims.back() != 2) {
    return ffi::Error::InvalidArgument("ds_from_f64 output last dim must be 2");
  }
  if (batchSize(y_dims) != batchSize(x_dims) * x_dims.back()) {
    return ffi::Error::InvalidArgument("ds_from_f64 shape mismatch");
  }

  const int64_t n = batchSize(x_dims) * x_dims.back();
  const double* in = x.typed_data();
  float* out = y->typed_data();

  for (int64_t i = 0; i < n; ++i) {
    float hi = static_cast<float>(in[i]);
    float lo = static_cast<float>(in[i] - static_cast<double>(hi));
    out[2 * i + 0] = hi;
    out[2 * i + 1] = lo;
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DsFromF64,
    DsFromF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F32>>());

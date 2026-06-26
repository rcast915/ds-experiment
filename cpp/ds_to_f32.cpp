#include <cstdint>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template<typename Dims>
static int64_t totalElements(const Dims& dims) {
  int64_t n = 1;
  for (auto d : dims) n *= d;
  return n;
}

// Input:  x  shape [...,2] f32
// Output: y  shape [...] f32  where y[...] = x[...,0] + x[...,1]
ffi::Error DsToF32Impl(ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y) {
  auto x_dims = x.dimensions();
  auto y_dims = y->dimensions();

  if (x_dims.back() != 2)
    return ffi::Error::InvalidArgument("ds_to_f32 input last dim must be 2");
  if (totalElements(x_dims) != totalElements(y_dims) * 2)
    return ffi::Error::InvalidArgument("ds_to_f32 shape mismatch");

  const int64_t n = totalElements(y_dims);
  const float* in = x.typed_data();
  float* out = y->typed_data();
  for (int64_t i = 0; i < n; ++i)
    out[i] = in[2*i+0] + in[2*i+1];  // hi + lo
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(DsToF32, DsToF32Impl,
  ffi::Ffi::Bind()
    .Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "ds_to_f32", "Host", DsToF32);

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

// Input:  x  shape [...] f32  (any shape)
// Output: y  shape [...,2] f32  where y[...,0]=hi=x, y[...,1]=lo=0
ffi::Error DsFromF32Impl(ffi::Buffer<ffi::F32> x,
                         ffi::ResultBuffer<ffi::F32> y) {
  auto x_dims = x.dimensions();
  auto y_dims = y->dimensions();

  if (y_dims.back() != 2)
    return ffi::Error::InvalidArgument("ds_from_f32 output last dim must be 2");
  if (totalElements(x_dims) * 2 != totalElements(y_dims))
    return ffi::Error::InvalidArgument("ds_from_f32 shape mismatch");

  const int64_t n = totalElements(x_dims);
  const float* in = x.typed_data();
  float* out = y->typed_data();
  for (int64_t i = 0; i < n; ++i) {
    out[2*i+0] = in[i];  // hi
    out[2*i+1] = 0.0f;   // lo
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(DsFromF32, DsFromF32Impl,
  ffi::Ffi::Bind()
    .Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "ds_from_f32", "Host", DsFromF32);

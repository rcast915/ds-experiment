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

inline void two_sum(float a, float b, float& s, float& e) {
  s = a + b;
  float bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

// Input:  a shape [N,2], b shape [N,2]
// Output: y shape [N,2]
ffi::Error DsSubImpl(ffi::Buffer<ffi::F32> a,
                     ffi::Buffer<ffi::F32> b,
                     ffi::ResultBuffer<ffi::F32> y) {
  auto a_dims = a.dimensions();
  auto b_dims = b.dimensions();
  auto y_dims = y->dimensions();

  if (a_dims.back() != 2 || b_dims.back() != 2) {
    return ffi::Error::InvalidArgument("ds.sub expects last dim == 2");
  }
  if (a_dims.size() != b_dims.size() || batchSize(a_dims) != batchSize(b_dims)) {
    return ffi::Error::InvalidArgument("ds.sub input shapes must match");
  }
  if (y_dims.back() != 2 || batchSize(y_dims) != batchSize(a_dims)) {
    return ffi::Error::InvalidArgument("ds.sub output shape mismatch");
  }

  const int64_t n = batchSize(a_dims);
  const float* ap = a.typed_data();
  const float* bp = b.typed_data();
  float* out = y->typed_data();

  for (int64_t i = 0; i < n; ++i) {
    float a_hi = ap[2 * i + 0];
    float a_lo = ap[2 * i + 1];
    float b_hi = bp[2 * i + 0];
    float b_lo = bp[2 * i + 1];

    float s1, e1;
    two_sum(a_hi, -b_hi, s1, e1);

    float s2, e2;
    two_sum(a_lo, -b_lo, s2, e2);

    float t1, t2;
    two_sum(s1, s2 + e1, t1, t2);

    out[2 * i + 0] = t1;
    out[2 * i + 1] = t2 + e2;
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DsSub,
    DsSubImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());

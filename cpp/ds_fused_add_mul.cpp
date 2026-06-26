#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

inline void two_sum(float a, float b, float& s, float& e) {
  s = a + b;
  float bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

inline void split(float a, float& hi, float& lo) {
  constexpr float c = 4097.0f;
  float t = c * a;
  hi = t - (t - a);
  lo = a - hi;
}

inline void two_prod(float a, float b, float& p, float& e) {
  p = a * b;
  float a_hi, a_lo, b_hi, b_lo;
  split(a, a_hi, a_lo);
  split(b, b_hi, b_lo);
  e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}

inline void ds_from_double(double x, float& hi, float& lo) {
  hi = static_cast<float>(x);
  lo = static_cast<float>(x - static_cast<double>(hi));
}

inline void ds_add_pair(float a_hi, float a_lo,
                        float b_hi, float b_lo,
                        float& out_hi, float& out_lo) {
  float s1, e1;
  two_sum(a_hi, b_hi, s1, e1);

  float s2, e2;
  two_sum(a_lo, b_lo, s2, e2);

  float t1, t2;
  two_sum(s1, s2 + e1, t1, t2);

  out_hi = t1;
  out_lo = t2 + e2;
}

inline void ds_mul_pair(float a_hi, float a_lo,
                        float b_hi, float b_lo,
                        float& out_hi, float& out_lo) {
  float p1, e1;
  two_prod(a_hi, b_hi, p1, e1);

  float cross = a_hi * b_lo + a_lo * b_hi;
  float s, e2;
  two_sum(p1, cross, s, e2);

  out_hi = s;
  out_lo = e1 + e2 + a_lo * b_lo;
}

// Input: x shape [N], y shape [N] as f64
// Output: out shape [N,2] as DS packed in f32
ffi::Error DsFusedAddMulImpl(ffi::Buffer<ffi::F64> x,
                             ffi::Buffer<ffi::F64> y,
                             ffi::ResultBuffer<ffi::F32> out) {
  auto x_dims = x.dimensions();
  auto y_dims = y.dimensions();
  auto out_dims = out->dimensions();

  if (x_dims.size() != 1 || y_dims.size() != 1) {
    return ffi::Error::InvalidArgument("ds.fused_add_mul expects rank-1 inputs");
  }
  if (x_dims[0] != y_dims[0]) {
    return ffi::Error::InvalidArgument("input lengths must match");
  }
  if (out_dims.size() != 2 || out_dims[0] != x_dims[0] || out_dims[1] != 2) {
    return ffi::Error::InvalidArgument("output shape must be [N,2]");
  }

  const int64_t n = x_dims[0];
  const double* xp = x.typed_data();
  const double* yp = y.typed_data();
  float* op = out->typed_data();

  for (int64_t i = 0; i < n; ++i) {
    float x_hi, x_lo, y_hi, y_lo;
    ds_from_double(xp[i], x_hi, x_lo);
    ds_from_double(yp[i], y_hi, y_lo);

    float sum_hi, sum_lo;
    ds_add_pair(x_hi, x_lo, y_hi, y_lo, sum_hi, sum_lo);

    float prod_hi, prod_lo;
    ds_mul_pair(sum_hi, sum_lo, y_hi, y_lo, prod_hi, prod_lo);

    op[2 * i + 0] = prod_hi;
    op[2 * i + 1] = prod_lo;
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DsFusedAddMul,
    DsFusedAddMulImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F32>>());

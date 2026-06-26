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

// Convert f64 -> DS
inline void ds_from_f64_scalar(double x, float& hi, float& lo) {
  hi = static_cast<float>(x);
  lo = static_cast<float>(x - static_cast<double>(hi));
}

// DS add
inline void ds_add_scalar(float a_hi, float a_lo,
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

// DS mul
inline void ds_mul_scalar(float a_hi, float a_lo,
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

// Convert DS -> f64
inline double ds_to_f64_scalar(float hi, float lo) {
  return static_cast<double>(hi) + static_cast<double>(lo);
}

// Compute: out[i] = (x[i] + y[i]) * y[i], using DS internally.
// Input:  x shape [N], y shape [N]
// Output: out shape [N]
ffi::Error DsFusedAxpyImpl(ffi::Buffer<ffi::F64> x,
                           ffi::Buffer<ffi::F64> y,
                           ffi::ResultBuffer<ffi::F64> out) {
  auto x_dims = x.dimensions();
  auto y_dims = y.dimensions();
  auto out_dims = out->dimensions();

  if (x_dims.size() != 1 || y_dims.size() != 1) {
    return ffi::Error::InvalidArgument("ds_fused_axpy expects rank-1 inputs");
  }
  if (x_dims[0] != y_dims[0]) {
    return ffi::Error::InvalidArgument("input lengths must match");
  }
  if (out_dims.size() != 1 || out_dims[0] != x_dims[0]) {
    return ffi::Error::InvalidArgument("output shape must be [N]");
  }

  const int64_t n = x_dims[0];
  const double* xp = x.typed_data();
  const double* yp = y.typed_data();
  double* op = out->typed_data();

  for (int64_t i = 0; i < n; ++i) {
    float x_hi, x_lo, y_hi, y_lo;
    ds_from_f64_scalar(xp[i], x_hi, x_lo);
    ds_from_f64_scalar(yp[i], y_hi, y_lo);

    float sum_hi, sum_lo;
    ds_add_scalar(x_hi, x_lo, y_hi, y_lo, sum_hi, sum_lo);

    float prod_hi, prod_lo;
    ds_mul_scalar(sum_hi, sum_lo, y_hi, y_lo, prod_hi, prod_lo);

    op[i] = ds_to_f64_scalar(prod_hi, prod_lo);
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DsFusedAxpy,
    DsFusedAxpyImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());

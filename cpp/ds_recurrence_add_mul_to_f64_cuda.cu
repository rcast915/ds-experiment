#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

__device__ inline void two_sum(float a, float b, float& s, float& e) {
  s = a + b;
  float bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

__device__ inline void split(float a, float& hi, float& lo) {
  constexpr float c = 4097.0f;
  float t = c * a;
  hi = t - (t - a);
  lo = a - hi;
}

__device__ inline void two_prod(float a, float b, float& p, float& e) {
  p = a * b;
  float a_hi, a_lo, b_hi, b_lo;
  split(a, a_hi, a_lo);
  split(b, b_hi, b_lo);
  e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}

__device__ inline void ds_from_double(double x, float& hi, float& lo) {
  hi = static_cast<float>(x);
  lo = static_cast<float>(x - static_cast<double>(hi));
}

__device__ inline void ds_add_pair(float a_hi, float a_lo,
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

__device__ inline void ds_mul_pair(float a_hi, float a_lo,
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

__device__ inline double ds_to_double(float hi, float lo) {
  return static_cast<double>(hi) + static_cast<double>(lo);
}

__global__ void DsRecurrenceAddMulToF64Kernel(const double* x,
                                              const double* y,
                                              double* out,
                                              int64_t n,
                                              int32_t k) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float z_hi, z_lo, y_hi, y_lo;
  ds_from_double(x[i], z_hi, z_lo);
  ds_from_double(y[i], y_hi, y_lo);

  for (int32_t iter = 0; iter < k; ++iter) {
    float sum_hi, sum_lo;
    ds_add_pair(z_hi, z_lo, y_hi, y_lo, sum_hi, sum_lo);

    float prod_hi, prod_lo;
    ds_mul_pair(sum_hi, sum_lo, y_hi, y_lo, prod_hi, prod_lo);

    z_hi = prod_hi;
    z_lo = prod_lo;
  }

  out[i] = ds_to_double(z_hi, z_lo);
}

// Temporary version: default stream + device sync for trustworthy timing.
ffi::Error DsRecurrenceAddMulToF64CudaImpl(
    ffi::Buffer<ffi::F64> x,
    ffi::Buffer<ffi::F64> y,
    int32_t k,
    ffi::ResultBuffer<ffi::F64> out) {
  auto x_dims = x.dimensions();
  auto y_dims = y.dimensions();
  auto out_dims = out->dimensions();

  if (x_dims.size() != 1 || y_dims.size() != 1) {
    return ffi::Error::InvalidArgument(
        "ds_recurrence_add_mul_to_f64_cuda expects rank-1 inputs");
  }
  if (x_dims[0] != y_dims[0]) {
    return ffi::Error::InvalidArgument("input lengths must match");
  }
  if (out_dims.size() != 1 || out_dims[0] != x_dims[0]) {
    return ffi::Error::InvalidArgument("output shape must be [N]");
  }
  if (k < 0) {
    return ffi::Error::InvalidArgument("k must be nonnegative");
  }

  int64_t n = x_dims[0];
  const double* xp = x.typed_data();
  const double* yp = y.typed_data();
  double* op = out->typed_data();

  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);

  DsRecurrenceAddMulToF64Kernel<<<blocks, threads>>>(xp, yp, op, n, k);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return ffi::Error::Internal(cudaGetErrorString(err));
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return ffi::Error::Internal(cudaGetErrorString(err));
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DsRecurrenceAddMulToF64Cuda,
    DsRecurrenceAddMulToF64CudaImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Attr<int32_t>("k")
        .Ret<ffi::Buffer<ffi::F64>>());

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

__device__ inline void ds_sub_pair(float a_hi, float a_lo,
                                   float b_hi, float b_lo,
                                   float& out_hi, float& out_lo) {
  float s1, e1;
  two_sum(a_hi, -b_hi, s1, e1);

  float s2, e2;
  two_sum(a_lo, -b_lo, s2, e2);

  float t1, t2;
  two_sum(s1, s2 + e1, t1, t2);

  out_hi = t1;
  out_lo = t2 + e2;
}

__global__ void DsFusedAddSubToF64Kernel(const double* x,
                                         const double* y,
                                         double* out,
                                         int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float x_hi, x_lo, y_hi, y_lo;
  ds_from_double(x[i], x_hi, x_lo);
  ds_from_double(y[i], y_hi, y_lo);

  float sum_hi, sum_lo;
  ds_add_pair(x_hi, x_lo, y_hi, y_lo, sum_hi, sum_lo);

  float res_hi, res_lo;
  ds_sub_pair(sum_hi, sum_lo, x_hi, x_lo, res_hi, res_lo);

  out[i] = static_cast<double>(res_hi) + static_cast<double>(res_lo);
}

// Temporary version using default stream, matching your current GPU path style.
ffi::Error DsFusedAddSubToF64CudaImpl(
    ffi::Buffer<ffi::F64> x,
    ffi::Buffer<ffi::F64> y,
    ffi::ResultBuffer<ffi::F64> out) {
  auto x_dims = x.dimensions();
  auto y_dims = y.dimensions();
  auto out_dims = out->dimensions();

  if (x_dims.size() != 1 || y_dims.size() != 1) {
    return ffi::Error::InvalidArgument(
        "ds_fused_add_sub_to_f64_cuda expects rank-1 inputs");
  }
  if (x_dims[0] != y_dims[0]) {
    return ffi::Error::InvalidArgument("input lengths must match");
  }
  if (out_dims.size() != 1 || out_dims[0] != x_dims[0]) {
    return ffi::Error::InvalidArgument("output shape must be [N]");
  }

  int64_t n = x_dims[0];
  const double* xp = x.typed_data();
  const double* yp = y.typed_data();
  double* op = out->typed_data();

  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);

  DsFusedAddSubToF64Kernel<<<blocks, threads>>>(xp, yp, op, n);

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
    DsFusedAddSubToF64Cuda,
    DsFusedAddSubToF64CudaImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());

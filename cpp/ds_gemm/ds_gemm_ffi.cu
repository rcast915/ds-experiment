// ds_gemm_ffi.cu -- XLA FFI handler for the double-single (DS) matmul.
//
// The pass emits, for an f64 matmul C = A B,
//   %c_hi, %c_lo = stablehlo.custom_call @ds_gemm(%a_hi, %a_lo, %b_hi, %b_lo)
//       {api_version = 4 (typed FFI), operand/result layouts row-major}
// with every operand and result an f32 matrix: the (hi, lo) words of A (M x K),
// B (K x N) and C (M x N). The proxy plugin (pjrt_plugin/ds_pjrt_plugin.cpp)
// loads this library and registers `DsGemm` under the target name "ds_gemm"
// through PJRT's FFI extension, so user code never has to.

#include <cstdint>
#include <limits>
#include <cuda_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "ds_gemm_kernel.cuh"

namespace ffi = xla::ffi;

static ffi::Error DsGemmImpl(cudaStream_t stream,
                             ffi::Buffer<ffi::F32> a_hi, ffi::Buffer<ffi::F32> a_lo,
                             ffi::Buffer<ffi::F32> b_hi, ffi::Buffer<ffi::F32> b_lo,
                             ffi::ResultBuffer<ffi::F32> c_hi,
                             ffi::ResultBuffer<ffi::F32> c_lo) {
    auto ad = a_hi.dimensions();
    auto bd = b_hi.dimensions();
    auto cd = c_hi->dimensions();
    if (ad.size() != 2 || bd.size() != 2 || cd.size() != 2) {
        return ffi::Error::InvalidArgument("ds_gemm: expects rank-2 operands and result");
    }
    const int64_t M = ad[0], K = ad[1], N = bd[1];
    if (bd[0] != K || cd[0] != M || cd[1] != N) {
        return ffi::Error::InvalidArgument("ds_gemm: shapes must be A[M,K] B[K,N] C[M,N]");
    }
    if (a_lo.dimensions().size() != 2 || a_lo.dimensions()[0] != M || a_lo.dimensions()[1] != K ||
        b_lo.dimensions().size() != 2 || b_lo.dimensions()[0] != K || b_lo.dimensions()[1] != N) {
        return ffi::Error::InvalidArgument("ds_gemm: lo words must match the hi words' shapes");
    }
    constexpr int64_t kMax = std::numeric_limits<int>::max();
    if (M > kMax || N > kMax || K > kMax) {
        return ffi::Error::InvalidArgument("ds_gemm: dimension too large");
    }
    if (M == 0 || N == 0) return ffi::Error::Success();
    if (K == 0) {
        cudaMemsetAsync(c_hi->typed_data(), 0, sizeof(float) * M * N, stream);
        cudaMemsetAsync(c_lo->typed_data(), 0, sizeof(float) * M * N, stream);
        return ffi::Error::Success();
    }

    cudaError_t err = ds_gemm_launch((int)M, (int)N, (int)K,
                                     a_hi.typed_data(), a_lo.typed_data(),
                                     b_hi.typed_data(), b_lo.typed_data(),
                                     nullptr, c_hi->typed_data(), c_lo->typed_data(),
                                     stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DsGemm, DsGemmImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()    // a_hi
        .Arg<ffi::Buffer<ffi::F32>>()    // a_lo
        .Arg<ffi::Buffer<ffi::F32>>()    // b_hi
        .Arg<ffi::Buffer<ffi::F32>>()    // b_lo
        .Ret<ffi::Buffer<ffi::F32>>()    // c_hi
        .Ret<ffi::Buffer<ffi::F32>>());  // c_lo

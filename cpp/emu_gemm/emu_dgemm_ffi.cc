// emu_dgemm_ffi.cc -- XLA FFI handler: f64 C = A B through cublasDgemm, with
// cuBLAS FP64 emulation (Ozaki, CUDA 13.0 Update 2+) switched on per call.
//
// Feasibility spike for routing f64 matmuls to emulation from inside a jitted
// JAX program (XLA itself only reaches emulation at native-FP64 speed; see
// tests/probe_xla_emulation.py). Host-only C++: no device code.
//
//   emu_dgemm(A[M,K], B[K,N]) -> C[M,N], all f64, row-major
//   attrs: emulate (int64, 0 = native DGEMM, 1 = emulated)
//          ws_mb   (int64, 0 = cuBLAS's own workspace, >0 = that many MiB
//                   taken from XLA's scratch allocator for this call)
//
// Emulation is set with cublasSetMathMode on this library's own handle, never
// through CUBLAS_EMULATE_DOUBLE_PRECISION, so XLA's cuBLAS calls are unaffected.
// One handle per (emulate, explicit workspace) pair, so a handle given XLA
// scratch is never left holding cuBLAS's default workspace or vice versa.
// Row-major A B equals column-major (B^T A^T)^T, so passing B first with
// m = N, n = M computes the row-major product without transposing copies.
// Runs on the stream XLA hands the call. Single-GPU spike: one handle pair.

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>

#include <cublas_v2.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static std::mutex g_mu;
static cublasHandle_t g_handle[4] = {};   // index: 2 * emulate + (ws_mb > 0)

static ffi::Error CublasError(const char* what, cublasStatus_t s) {
    return ffi::Error::Internal(std::string("emu_dgemm: ") + what + " failed, status " +
                                std::to_string(static_cast<int>(s)));
}

static ffi::Error EmuDgemmImpl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                               ffi::Buffer<ffi::F64> a, ffi::Buffer<ffi::F64> b,
                               ffi::ResultBuffer<ffi::F64> c,
                               int64_t emulate, int64_t ws_mb) {
    auto ad = a.dimensions(), bd = b.dimensions(), cd = c->dimensions();
    if (ad.size() != 2 || bd.size() != 2 || cd.size() != 2)
        return ffi::Error::InvalidArgument("emu_dgemm: expects rank-2 operands and result");
    const int64_t M = ad[0], K = ad[1], N = bd[1];
    if (bd[0] != K || cd[0] != M || cd[1] != N)
        return ffi::Error::InvalidArgument("emu_dgemm: shapes must be A[M,K] B[K,N] C[M,N]");
    constexpr int64_t kMax = std::numeric_limits<int>::max();
    if (M > kMax || N > kMax || K > kMax)
        return ffi::Error::InvalidArgument("emu_dgemm: dimension too large");
    if (M == 0 || N == 0 || K == 0)
        return ffi::Error::InvalidArgument("emu_dgemm: empty matmul not handled in the spike");

    std::lock_guard<std::mutex> lock(g_mu);
    const bool emu = emulate != 0;
    cublasHandle_t& h = g_handle[2 * emu + (ws_mb > 0)];
    if (!h) {
        cublasStatus_t s = cublasCreate(&h);
        if (s != CUBLAS_STATUS_SUCCESS) { h = nullptr; return CublasError("cublasCreate", s); }
        s = cublasSetMathMode(h, emu ? CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
                                       : CUBLAS_DEFAULT_MATH);
        if (s != CUBLAS_STATUS_SUCCESS) return CublasError("cublasSetMathMode", s);
    }
    cublasStatus_t s = cublasSetStream(h, stream);
    if (s != CUBLAS_STATUS_SUCCESS) return CublasError("cublasSetStream", s);
    if (ws_mb > 0) {
        const size_t bytes = static_cast<size_t>(ws_mb) << 20;
        auto ws = scratch.Allocate(bytes, 256);
        if (!ws.has_value())
            return ffi::Error::Internal("emu_dgemm: XLA scratch allocation of workspace failed");
        s = cublasSetWorkspace(h, *ws, bytes);
        if (s != CUBLAS_STATUS_SUCCESS) return CublasError("cublasSetWorkspace", s);
    }
    const double one = 1.0, zero = 0.0;
    s = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K,
                    &one, b.typed_data(), (int)N, a.typed_data(), (int)K,
                    &zero, c->typed_data(), (int)N);
    if (s != CUBLAS_STATUS_SUCCESS) return CublasError("cublasDgemm", s);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EmuDgemm, EmuDgemmImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::F64>>()    // A
        .Arg<ffi::Buffer<ffi::F64>>()    // B
        .Ret<ffi::Buffer<ffi::F64>>()    // C
        .Attr<int64_t>("emulate")
        .Attr<int64_t>("ws_mb"));

// ds_gemm_kernel.cuh -- the double-single (DS) matmul kernel.
//
// Used by the standalone benchmark (ds_gemm_bench.cu) and by the XLA FFI
// handler the pass calls for f64 matmuls (ds_gemm_ffi.cu).
//
// Classic register-blocked GEMM tiling (64x64 block tile, depth 8, 256
// threads, 4x4 outputs per thread). Every multiply-add is ds_mac() from
// ds_gemm_math.h; accumulators are renormalized after each depth-8 tile.

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "ds_gemm_math.h"

constexpr int BM = 64, BN = 64, BK = 8;   // block tile
constexpr int TM = 4,  TN = 4;            // outputs per thread
constexpr int THREADS = (BM / TM) * (BN / TN);   // 256

// f64 -> (hi, lo) f32 pair, elementwise.
__global__ void split_kernel(const double* __restrict__ x, float* __restrict__ hi,
                             float* __restrict__ lo, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        double v = x[i];
        float h = __double2float_rn(v);
        hi[i] = h;
        lo[i] = __double2float_rn(v - (double)h);
    }
}

// C (M x N, row-major) = A (M x K) * B (K x N), A and B given as (hi, lo) f32
// arrays, row-major. The result is written either as f64 (C64 != nullptr)
// or as a renormalized (hi, lo) f32 pair (Ch, Cl).
__global__ void __launch_bounds__(THREADS)
ds_gemm_kernel(int M, int N, int K,
               const float* __restrict__ Ah, const float* __restrict__ Al,
               const float* __restrict__ Bh, const float* __restrict__ Bl,
               double* __restrict__ C64,
               float* __restrict__ Ch, float* __restrict__ Cl) {
    // A tile stored transposed (k-major) so a thread's TM rows are contiguous.
    __shared__ float sAh[BK][BM], sAl[BK][BM];
    __shared__ float sBh[BK][BN], sBl[BK][BN];

    const int tid = threadIdx.x;
    const int tx  = tid % (BN / TN);   // 0..15, along N
    const int ty  = tid / (BN / TN);   // 0..15, along M
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    float acc_h[TM][TN], acc_l[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) { acc_h[i][j] = 0.f; acc_l[i][j] = 0.f; }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load the A tile (BM x BK) and B tile (BK x BN): 512 elements each,
        // two per thread, zero-filled outside the matrix.
        #pragma unroll
        for (int r = 0; r < (BM * BK) / THREADS; ++r) {
            int idx = tid + r * THREADS;          // 0..511
            int am = idx / BK, ak = idx % BK;     // A: row within tile, k within tile
            int gr = row0 + am, gk = k0 + ak;
            bool ok = (gr < M) && (gk < K);
            size_t g = (size_t)gr * K + gk;
            sAh[ak][am] = ok ? Ah[g] : 0.f;
            sAl[ak][am] = ok ? Al[g] : 0.f;

            int bk = idx / BN, bn = idx % BN;     // B: k within tile, col within tile
            int gk2 = k0 + bk, gc = col0 + bn;
            bool ok2 = (gk2 < K) && (gc < N);
            size_t g2 = (size_t)gk2 * N + gc;
            sBh[bk][bn] = ok2 ? Bh[g2] : 0.f;
            sBl[bk][bn] = ok2 ? Bl[g2] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ah[TM], al[TM], bh[TN], bl[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) { ah[i] = sAh[k][ty * TM + i]; al[i] = sAl[k][ty * TM + i]; }
            #pragma unroll
            for (int j = 0; j < TN; ++j) { bh[j] = sBh[k][tx * TN + j]; bl[j] = sBl[k][tx * TN + j]; }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    ds_mac(acc_h[i][j], acc_l[i][j], ah[i], al[i], bh[j], bl[j]);
        }

        // Keep the low word small: renormalize once per depth-8 tile.
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                ds_finish(acc_h[i][j], acc_l[i][j]);
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row0 + ty * TM + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = col0 + tx * TN + j;
            if (c >= N) continue;
            size_t o = (size_t)r * N + c;
            if (C64) {
                C64[o] = (double)acc_h[i][j] + (double)acc_l[i][j];
            } else {
                Ch[o] = acc_h[i][j];
                Cl[o] = acc_l[i][j];
            }
        }
    }
}

// Launch helper: C = A B on `stream`, inputs as (hi, lo) pairs.
inline cudaError_t ds_gemm_launch(int M, int N, int K,
                                  const float* Ah, const float* Al,
                                  const float* Bh, const float* Bl,
                                  double* C64, float* Ch, float* Cl,
                                  cudaStream_t stream = 0) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    ds_gemm_kernel<<<grid, THREADS, 0, stream>>>(M, N, K, Ah, Al, Bh, Bl, C64, Ch, Cl);
    return cudaGetLastError();
}

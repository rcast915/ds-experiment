// ds_gemm_bench.cu -- standalone double-single (DS) matmul kernel and benchmark.
//
// Question it answers: can a hand-written GPU kernel do C = A B for f64 inputs
// in double-single arithmetic (pairs of f32, with real FMA) faster than
// cuBLAS's native f64 DGEMM, at f64-class accuracy?
//
// For each size n (square n x n matrices of standard-normal f64 values):
//   DGEMM   cuBLAS f64 matmul                      (baseline, and the reference)
//   SGEMM   cuBLAS f32 matmul on f32-rounded inputs (speed and accuracy of plain f32)
//   DS      split f64 -> (hi, lo) f32 pairs, then the DS kernel below
// and reports median time, GFLOP/s (2n^3 / t) and relative error.
//
// Kernel: classic register-blocked GEMM tiling (64x64 block tile, depth 8,
// 256 threads, 4x4 outputs per thread), with every multiply-add done by
// ds_mac() from ds_gemm_math.h and the accumulators renormalized after each
// depth-8 tile. Inputs and accumulators are (hi, lo) f32 pairs; the result is
// written as f64 = hi + lo.
//
// Build and run (inside the container; needs a GPU at build time for -arch=native):
//   nvcc -O3 -std=c++17 -arch=native -o ds_gemm_bench ds_gemm_bench.cu -lcublas
//   ./ds_gemm_bench [n ...]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "ds_gemm_math.h"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
    exit(1); } } while (0)

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

// C (M x N, row-major, f64) = A (M x K) * B (K x N), A and B given as (hi, lo)
// f32 arrays, row-major.
__global__ void __launch_bounds__(THREADS)
ds_gemm_kernel(int M, int N, int K,
               const float* __restrict__ Ah, const float* __restrict__ Al,
               const float* __restrict__ Bh, const float* __restrict__ Bl,
               double* __restrict__ C) {
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
            if (c < N) C[(size_t)r * N + c] = (double)acc_h[i][j] + (double)acc_l[i][j];
        }
    }
}

// Median over `reps` timed calls of f(), after `warmup` untimed calls.
template <class F>
static double time_ms(F f, int warmup = 3, int reps = 10) {
    for (int i = 0; i < warmup; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(a));
        f();
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static double rel_frobenius(const std::vector<double>& x, const std::vector<double>& ref) {
    long double num = 0, den = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        long double d = (long double)x[i] - ref[i];
        num += d * d;
        den += (long double)ref[i] * ref[i];
    }
    return (double)sqrtl(num / den);
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096};

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Square n x n, standard-normal f64 inputs. Times are medians of 10 runs.\n");
    printf("Error = ||C - C_ref||_F / ||C_ref||_F with C_ref = cuBLAS DGEMM;\n");
    printf("DGEMM's own error is checked against long double on sampled entries.\n\n");

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));   // no TF32 for SGEMM

    printf("%6s | %9s %8s | %9s %8s %9s | %9s %9s %8s %9s | %8s %8s\n",
           "n", "DGEMM ms", "GFLOP/s", "SGEMM ms", "GFLOP/s", "err", "DS ms", "split ms",
           "GFLOP/s", "err", "DGEMM/DS", "DGEMMerr");
    printf("-------+--------------------+------------------------------+--------------------------------------+------------------\n");

    for (int n : sizes) {
        const size_t nn = (size_t)n * n;
        std::mt19937_64 rng(n);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<double> hA(nn), hB(nn);
        for (auto& v : hA) v = nd(rng);
        for (auto& v : hB) v = nd(rng);
        std::vector<float> fA(nn), fB(nn);
        for (size_t i = 0; i < nn; ++i) { fA[i] = (float)hA[i]; fB[i] = (float)hB[i]; }

        double *dA, *dB, *dC64, *dCds;
        float *dAh, *dAl, *dBh, *dBl, *fdA, *fdB, *fdC;
        CUDA_CHECK(cudaMalloc(&dA, nn * 8));   CUDA_CHECK(cudaMalloc(&dB, nn * 8));
        CUDA_CHECK(cudaMalloc(&dC64, nn * 8)); CUDA_CHECK(cudaMalloc(&dCds, nn * 8));
        CUDA_CHECK(cudaMalloc(&dAh, nn * 4));  CUDA_CHECK(cudaMalloc(&dAl, nn * 4));
        CUDA_CHECK(cudaMalloc(&dBh, nn * 4));  CUDA_CHECK(cudaMalloc(&dBl, nn * 4));
        CUDA_CHECK(cudaMalloc(&fdA, nn * 4));  CUDA_CHECK(cudaMalloc(&fdB, nn * 4));
        CUDA_CHECK(cudaMalloc(&fdC, nn * 4));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), nn * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), nn * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fdA, fA.data(), nn * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fdB, fB.data(), nn * 4, cudaMemcpyHostToDevice));

        // Row-major C = A B is column-major C^T = B^T A^T: pass B first.
        const double one = 1.0, zero = 0.0;
        const float onef = 1.f, zerof = 0.f;
        auto dgemm = [&] {
            CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                     &one, dB, n, dA, n, &zero, dC64, n));
        };
        auto sgemm = [&] {
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                     &onef, fdB, n, fdA, n, &zerof, fdC, n));
        };
        const int sthreads = 256;
        const unsigned sblocks = (unsigned)((nn + sthreads - 1) / sthreads);
        auto split = [&] {
            split_kernel<<<sblocks, sthreads>>>(dA, dAh, dAl, nn);
            split_kernel<<<sblocks, sthreads>>>(dB, dBh, dBl, nn);
        };
        dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
        auto dsgemm = [&] {
            ds_gemm_kernel<<<grid, THREADS>>>(n, n, n, dAh, dAl, dBh, dBl, dCds);
        };

        double t_d = time_ms(dgemm);
        double t_s = time_ms(sgemm);
        double t_split = time_ms(split);
        double t_ds = time_ms(dsgemm);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> c64(nn), cds(nn);
        std::vector<float> c32(nn);
        CUDA_CHECK(cudaMemcpy(c64.data(), dC64, nn * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cds.data(), dCds, nn * 8, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c32.data(), fdC, nn * 4, cudaMemcpyDeviceToHost));
        std::vector<double> c32d(c32.begin(), c32.end());

        // DGEMM against long double on sampled entries (is the reference good enough?).
        double dgemm_err = 0;
        std::uniform_int_distribution<int> pick(0, n - 1);
        long double num = 0, den = 0;
        for (int s = 0; s < 64; ++s) {
            int r = pick(rng), c = pick(rng);
            long double ref = 0;
            for (int k = 0; k < n; ++k) ref += (long double)hA[(size_t)r * n + k] * hB[(size_t)k * n + c];
            long double d = (long double)c64[(size_t)r * n + c] - ref;
            num += d * d; den += ref * ref;
        }
        dgemm_err = (double)sqrtl(num / den);

        double flops = 2.0 * n * (double)n * n;
        printf("%6d | %9.3f %8.0f | %9.3f %8.0f %9.2e | %9.3f %9.3f %8.0f %9.2e | %7.2fx %8.1e\n",
               n, t_d, flops / t_d / 1e6, t_s, flops / t_s / 1e6, rel_frobenius(c32d, c64),
               t_ds, t_split, flops / t_ds / 1e6, rel_frobenius(cds, c64),
               t_d / (t_ds + t_split), dgemm_err);

        cudaFree(dA); cudaFree(dB); cudaFree(dC64); cudaFree(dCds);
        cudaFree(dAh); cudaFree(dAl); cudaFree(dBh); cudaFree(dBl);
        cudaFree(fdA); cudaFree(fdB); cudaFree(fdC);
    }
    cublasDestroy(h);

    printf("\nDGEMM/DS > 1 means the DS kernel (split included) is faster than native f64.\n");
    printf("Go/no-go (L40S, n=2048): DGEMM/DS >= 1.5x with DS err < 1e-12.\n");
    return 0;
}

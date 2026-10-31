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

#include "ds_gemm_kernel.cuh"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
    exit(1); } } while (0)

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
        auto dsgemm = [&] {
            ds_gemm_launch(n, n, n, dAh, dAl, dBh, dBl, dCds, nullptr, nullptr);
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

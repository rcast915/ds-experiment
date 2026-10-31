// ds_gemm_math.h -- the per-term arithmetic of the double-single matmul.
//
// Shared by the GPU kernel (ds_gemm_bench.cu) and a CPU check
// (test_ds_gemm_math.cpp), so the exact same sequence is tested on both.
//
// A value is a pair of floats (hi, lo) with value hi + lo. One output element
// of C = A B is the dot product sum_k a_k b_k, accumulated with the
// compensated scheme of Ogita, Rump & Oishi ("Dot2"), extended to inputs that
// are themselves (hi, lo) pairs:
//
//   p = ah*bh,  e = exact rounding error of p        (TwoProduct, via FMA)
//   e += ah*bl + al*bh                               (input lo words; al*bl dropped)
//   (s, q) = TwoSum(acc_hi, p)                        exact
//   acc_hi = s;  acc_lo += q + e
//
// 12 f32 instructions per term (1 mul, 3 fma, 8 add/sub). Every operation is
// written with an explicitly rounded intrinsic so the compiler cannot contract
// a multiply and an add into an FMA, which would break the error-free TwoSum.
// The CPU build must use -ffp-contract=off for the same reason.

#pragma once

#if defined(__CUDACC__)
#define DSG_HD __host__ __device__ __forceinline__
#else
#define DSG_HD inline
#endif

#if defined(__CUDA_ARCH__)
#define DSG_ADD(a, b)    __fadd_rn((a), (b))
#define DSG_SUB(a, b)    __fsub_rn((a), (b))
#define DSG_MUL(a, b)    __fmul_rn((a), (b))
#define DSG_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#else
#include <cmath>
#define DSG_ADD(a, b)    ((a) + (b))
#define DSG_SUB(a, b)    ((a) - (b))
#define DSG_MUL(a, b)    ((a) * (b))
#define DSG_FMA(a, b, c) std::fma((a), (b), (c))
#endif

// acc += a * b, with acc = (ch, cl), a = (ah, al), b = (bh, bl).
DSG_HD void ds_mac(float& ch, float& cl, float ah, float al, float bh, float bl) {
    float p  = DSG_MUL(ah, bh);
    float e  = DSG_FMA(ah, bh, -p);
    e        = DSG_FMA(ah, bl, e);
    e        = DSG_FMA(al, bh, e);
    float s  = DSG_ADD(ch, p);
    float bb = DSG_SUB(s, ch);
    float q  = DSG_ADD(DSG_SUB(ch, DSG_SUB(s, bb)), DSG_SUB(p, bb));
    ch = s;
    cl = DSG_ADD(cl, DSG_ADD(q, e));
}

// Renormalize an accumulated pair so |lo| <= half an ulp of hi.
DSG_HD void ds_finish(float& h, float& l) {
    float s = DSG_ADD(h, l);
    float z = DSG_SUB(s, h);
    l = DSG_SUB(l, z);
    h = s;
}

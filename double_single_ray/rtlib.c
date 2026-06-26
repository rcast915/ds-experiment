#include <stdio.h>
#include <math.h>
#include "llvm-accuracy-analysis-k-test/double-single-lib/double-binary32.h"

#define N 2

void convert_double_to_ds(double_binary32_t *out, double val) {
    float hi = (float)val;
    float lo = (float)(val - (double)hi);
    if (!isfinite(hi) || !isfinite(lo))
        printf("WARNING: convert_double_to_ds overflow: val=%.3e hi=%.3e lo=%.3e\n", val, hi, lo);
    out->hi = hi;
    out->lo = lo;
}

double ds_to_double(const double_binary32_t* a) {
    return (double)a->hi + (double)a->lo;
}

void external_double_add(double_binary32_t *r, const double_binary32_t *a, const double_binary32_t *b) {
    double_binary32_add(r, a, b);
}

void external_double_sub(double_binary32_t *r, const double_binary32_t *a, const double_binary32_t *b) {
    double_binary32_sub(r, a, b);
}

void external_double_mul(double_binary32_t *r, const double_binary32_t *a, const double_binary32_t *b) {
    double_binary32_mul(r, a, b);
}

void external_double_div(double_binary32_t *r, const double_binary32_t *a, const double_binary32_t *b) {
    double_binary32_div(r, a, b);
}

void external_double_neg(double_binary32_t *r, const double_binary32_t *a) {
    double_binary32_neg(r, a);
}

void external_double_sqrt(double_binary32_t *r, const double_binary32_t *a) {
    double_binary32_sqrt(r, a);
}

void external_double_fabs(double_binary32_t *r, const double_binary32_t *a) {
    double_binary32_fabs(r, a);
}

void external_double_fmadd(double_binary32_t* result,
                            const double_binary32_t *a,
                            const double_binary32_t *b,
                            const double_binary32_t *c) {
    double_binary32_t tmp;
    double_binary32_mul(&tmp, a, b);
    double_binary32_add(result, &tmp, c);
}

void print_ds_value(double_binary32_t ds) {
    printf("DS = %.8e + %.8e = %.17e\n", ds.hi, ds.lo, (double)ds.hi + (double)ds.lo);
}

#ifndef RTLIB_NO_MAIN
int main() {
    double A[N][N] = {
        {123.456, 78.9},
        {-34.5, 910.11}
    };
    double B[N][N] = {
        {11.2, -13.4},
        {0.05, 8.0}
    };

    printf("Matrix multiplication using DS arithmetic:\n");

    double C[N][N] = {0};
    double_binary32_t C_ds[N][N];
    double_binary32_t zero = {0.0f, 0.0f};

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C_ds[i][j] = zero;

    // Standard double multiply
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];

    // DS multiply
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double_binary32_t a_ds, b_ds, product;
                convert_double_to_ds(&a_ds, A[i][k]);
                convert_double_to_ds(&b_ds, B[k][j]);
                external_double_mul(&product, &a_ds, &b_ds);
                double_binary32_t tmp = C_ds[i][j];
                external_double_add(&C_ds[i][j], &tmp, &product);
            }
        }
    }

    double total_abs = 0.0, total_rel = 0.0;
    int count = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double d  = C[i][j];
            double ds = ds_to_double(&C_ds[i][j]);
            double ad = fabs(d - ds);
            total_abs += ad;
            total_rel += (fabs(d) > 1e-300) ? ad / fabs(d) : 0.0;
            count++;
        }
    }
    printf("\nAvg absolute diff: %.2e\n", total_abs / count);
    printf("Avg relative diff: %.2e\n", total_rel / count);
    return 0;
}
#endif // RTLIB_NO_MAIN

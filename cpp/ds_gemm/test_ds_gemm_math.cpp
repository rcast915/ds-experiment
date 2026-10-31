// CPU check of ds_gemm_math.h: DS dot products of random f64 vectors,
// compared with a long-double reference. Build and run:
//   g++ -O2 -ffp-contract=off -o /tmp/t test_ds_gemm_math.cpp && /tmp/t
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include "ds_gemm_math.h"

int main() {
    std::mt19937_64 rng(1);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int ns[] = {256, 1024, 2048, 4096};
    for (int n : ns) {
        double worst_rel = 0, worst_cond = 0, worst_f32 = 0;
        for (int trial = 0; trial < 300; ++trial) {
            std::vector<double> a(n), b(n);
            for (int i = 0; i < n; ++i) { a[i] = nd(rng); b[i] = nd(rng); }
            long double ref = 0, absum = 0;
            float ch = 0, cl = 0, f32 = 0;
            for (int i = 0; i < n; ++i) {
                ref   += (long double)a[i] * b[i];
                absum += fabsl((long double)a[i] * b[i]);
                float ah = (float)a[i], al = (float)(a[i] - ah);
                float bh = (float)b[i], bl = (float)(b[i] - bh);
                ds_mac(ch, cl, ah, al, bh, bl);
                f32 = f32 + (float)a[i] * (float)b[i];
            }
            ds_finish(ch, cl);
            long double got = (long double)ch + cl;
            double err = (double)fabsl(got - ref);
            worst_rel  = fmax(worst_rel,  err / (double)fabsl(ref));
            worst_cond = fmax(worst_cond, err / (double)absum);
            worst_f32  = fmax(worst_f32,  (double)fabsl((long double)f32 - ref) / (double)absum);
        }
        printf("n=%5d  DS err/|dot| %.2e   DS err/sum|ab| %.2e   (plain f32 err/sum|ab| %.2e)\n",
               n, worst_rel, worst_cond, worst_f32);
    }
}

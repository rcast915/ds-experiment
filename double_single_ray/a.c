#include <stdio.h>
#include <math.h>

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
static void check(const char* label, double got, double expected) {
    double err = fabs(got - expected) / (fabs(expected) > 1e-300 ? fabs(expected) : 1.0);
    printf("  %-30s got=%.10e  expected=%.10e  relerr=%.2e  %s\n",
           label, got, expected, err, err < 1e-5 ? "OK" : "FAIL");
}

// -----------------------------------------------------------------------
// 1. Scalar arithmetic  (fadd / fsub / fmul / fdiv)
// -----------------------------------------------------------------------
static void test_scalar() {
    printf("\n[1] Scalar arithmetic\n");
    double a = 1.0 / 3.0;
    double b = 1.0 / 7.0;
    check("a + b", a + b, 1.0/3.0 + 1.0/7.0);
    check("a - b", a - b, 1.0/3.0 - 1.0/7.0);
    check("a * b", a * b, (1.0/3.0) * (1.0/7.0));
    check("a / b", a / b, (1.0/3.0) / (1.0/7.0));
}

// -----------------------------------------------------------------------
// 2. Unary negation  (fneg)
// -----------------------------------------------------------------------
static void test_neg() {
    printf("\n[2] Negation\n");
    double x = 42.5;
    double y = -x;
    check("-42.5", y, -42.5);
}

// -----------------------------------------------------------------------
// 3. Math intrinsics  (sqrt, fabs)
// -----------------------------------------------------------------------
static void test_intrinsics() {
    printf("\n[3] Math intrinsics\n");
    double x = 2.0;
    check("sqrt(2)",  sqrt(x),  1.41421356237310);
    check("fabs(-3)", fabs(-3.0), 3.0);
}

// -----------------------------------------------------------------------
// 4. Integer <-> double conversions  (sitofp, fptosi)
// -----------------------------------------------------------------------
static void test_int_conversions() {
    printf("\n[4] Int <-> double conversions\n");
    int   n  = 12345;
    double d = (double)n;           // sitofp
    int   m  = (int)d;              // fptosi
    printf("  int->double->int: %d -> %.1f -> %d  %s\n",
           n, d, m, m == n ? "OK" : "FAIL");

    double big = 1e15 + 3.0;
    long long ll = (long long)big;  // fptosi i64
    printf("  double->i64: %.0f -> %lld  %s\n",
           big, ll, ll == 1000000000000003LL ? "OK" : "FAIL");
}

// -----------------------------------------------------------------------
// 5. Float <-> double conversions  (fpext, fptrunc)
// -----------------------------------------------------------------------
static void test_fp_conversions() {
    printf("\n[5] Float <-> double conversions\n");
    float  f = 3.14f;
    double d = (double)f;           // fpext
    float  g = (float)d;            // fptrunc
    printf("  float->double->float: %.6f -> %.6f -> %.6f  %s\n",
           f, d, g, fabsf(f - g) < 1e-6f ? "OK" : "FAIL");
}

// -----------------------------------------------------------------------
// 6. Comparison  (fcmp)
// -----------------------------------------------------------------------
static void test_compare() {
    printf("\n[6] Comparisons\n");
    double a = 1.5, b = 2.5;
    printf("  1.5 < 2.5 : %s\n", a < b  ? "true  OK" : "false FAIL");
    printf("  2.5 > 1.5 : %s\n", b > a  ? "true  OK" : "false FAIL");
    printf("  1.5 == 1.5: %s\n", a == a ? "true  OK" : "false FAIL");
    printf("  1.5 != 2.5: %s\n", a != b ? "true  OK" : "false FAIL");
}

// -----------------------------------------------------------------------
// 7. Array initialisation and element access
//    (memcpy from global, memset zero, GEP, load, store)
// -----------------------------------------------------------------------
static void test_arrays() {
    printf("\n[7] Array init and access\n");
    double A[4] = {1.1, 2.2, 3.3, 4.4};   // memcpy from constant
    double B[4] = {0.0, 0.0, 0.0, 0.0};   // memset zero
    for (int i = 0; i < 4; i++) B[i] = A[i] * 2.0;
    check("B[0] = 1.1*2", B[0], 2.2);
    check("B[3] = 4.4*2", B[3], 8.8);
}

// -----------------------------------------------------------------------
// 8. 2-D array (matrix)  (multi-dim GEP)
// -----------------------------------------------------------------------
static void test_matrix_2d() {
    printf("\n[8] 2D matrix element access\n");
    double M[3][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    check("M[1][2]", M[1][2], 6.0);
    check("M[2][0]", M[2][0], 7.0);
}

// -----------------------------------------------------------------------
// 9. Matrix multiplication  (exercises fmuladd / loop with PHI)
// -----------------------------------------------------------------------
static void test_matmul() {
    printf("\n[9] Matrix multiply (2x2)\n");
    double A[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    double B[2][2] = {{5.0, 6.0}, {7.0, 8.0}};
    double C[2][2] = {{0.0, 0.0}, {0.0, 0.0}};

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                C[i][j] += A[i][k] * B[k][j];

    // Expected: [[19,22],[43,50]]
    check("C[0][0]", C[0][0], 19.0);
    check("C[0][1]", C[0][1], 22.0);
    check("C[1][0]", C[1][0], 43.0);
    check("C[1][1]", C[1][1], 50.0);
}

// -----------------------------------------------------------------------
// 10. Accumulation loop  (PHI node carrying a double across iterations)
// -----------------------------------------------------------------------
static void test_loop_accumulate() {
    printf("\n[10] Loop accumulation (PHI node)\n");
    double sum = 0.0;
    for (int i = 1; i <= 100; i++)
        sum += (double)i;
    check("sum 1..100", sum, 5050.0);
}

// -----------------------------------------------------------------------
// 11. Chained operations
// -----------------------------------------------------------------------
static void test_chained() {
    printf("\n[11] Chained operations\n");
    double x = 2.0;
    double y = sqrt(x * x + 1.0) - 1.0;  // sqrt, mul, add, sub
    check("sqrt(5)-1", y, sqrt(5.0) - 1.0);
}

// -----------------------------------------------------------------------
int main() {
    printf("=== DS Pass Test Suite ===\n");
    test_scalar();
    test_neg();
    test_intrinsics();
    test_int_conversions();
    test_fp_conversions();
    test_compare();
    test_arrays();
    test_matrix_2d();
    test_matmul();
    test_loop_accumulate();
    test_chained();
    printf("\nDone.\n");
    return 0;
}

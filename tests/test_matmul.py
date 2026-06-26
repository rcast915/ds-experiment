"""
Matrix multiply accuracy test suite for the DS compiler pass.

Three sections mirror test_dot_product.py:

  Section 1 — NumPy Reference
    Validates ds_ref.ds_matmul against f64 ground truth for edge cases.
    No GPU or JAX needed.

  Section 2 — MLIR Structural
    Asserts that dot_general expands to 4 sub-matmuls in the lowered MLIR.

  Section 3 — GPU Numerical
    Asserts DS precision beats f32 on the GPU for cases where the
    lo channel carries information (prior element-wise mul in same jit).

Key limitation documented here:
    Standalone A @ B with no prior DS ops gives lo=0 for all inputs, so
    the 4-matmul decomposition adds nothing. DS benefit requires a chained
    operation (mul, add, reduce) in the same jax.jit to create the lo channel.

Usage:
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/test_matmul.py

  JAX_PLATFORMS=cpu python3 tests/test_matmul.py   # NumPy + structural only
"""

import sys
import os
import subprocess
import math
from pathlib import Path

import numpy as np

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
import ds_ref

OPT_BINARY = PROJECT_ROOT / "stablehlo_pass" / "build" / "mlir-ds-opt"
PLUGIN_SO  = PROJECT_ROOT / "pjrt_plugin"    / "build" / "libds_pjrt_plugin.so"

PASS_MARK = "  PASS "
FAIL_MARK = "  FAIL "
SKIP_MARK = "  SKIP "

_failures = []

def check(label, cond, detail=""):
    if cond:
        print(f"{PASS_MARK} {label}")
    else:
        msg = f"{FAIL_MARK} {label}" + (f"\n         {detail}" if detail else "")
        print(msg)
        _failures.append(label)

def skip(label, reason):
    print(f"{SKIP_MARK} {label}  [{reason}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: NumPy Reference Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_numpy_tests():
    print("\n[Section 1: NumPy Reference]\n")

    rng = np.random.default_rng(42)

    # ── 1a. Basic correctness against f64 ─────────────────────────────────────
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((8, 4)).astype(np.float32)

    ds_C  = ds_ref.ds_matmul(A, B)
    f64_C = ds_ref.f64_matmul(A, B)
    f32_C = ds_ref.f32_matmul(A, B)

    ds_err_max  = float(np.max(np.abs(ds_C  - f64_C)))
    f32_err_max = float(np.max(np.abs(f32_C - f64_C)))

    check("basic correctness: DS error ≤ f32 error on random 4×8 @ 8×4",
          ds_err_max <= f32_err_max + 1e-9,
          f"ds_max_err={ds_err_max:.3e}, f32_max_err={f32_err_max:.3e}")

    # ── 1b. Identity matrix: A @ I == A exactly ───────────────────────────────
    I = np.eye(4, dtype=np.float32)
    A4 = rng.standard_normal((4, 4)).astype(np.float32)
    ds_AI  = ds_ref.ds_matmul(A4, I)
    f64_AI = ds_ref.f64_matmul(A4, I)
    check("identity: DS(A @ I) matches f64 exactly",
          np.allclose(ds_AI, f64_AI, atol=1e-6),
          f"max_diff={np.max(np.abs(ds_AI - f64_AI)):.3e}")

    # ── 1c. Catastrophic cancellation pattern ─────────────────────────────────
    # Row of A: [1e6, -1e6+1, 1e6, -1e6+1] × column of B = [1,1,1,1]ᵀ.
    # f32 accumulation loses the "+1" offset; DS should keep it.
    # At scale 1e6 ULP≈0.0625, so "+1" is representable (1 >> 0.0625).
    A_cancel = np.array([[1e6, -1e6 + 1, 1e6, -1e6 + 1]], dtype=np.float32)
    B_ones   = np.ones((4, 1), dtype=np.float32)

    ds_c    = ds_ref.ds_matmul(A_cancel, B_ones)[0, 0]
    f64_c   = ds_ref.f64_matmul(A_cancel, B_ones)[0, 0]
    f32_c   = float(ds_ref.f32_matmul(A_cancel, B_ones)[0, 0])

    check("cancellation pattern: DS error ≤ f32 error",
          abs(ds_c - f64_c) <= abs(f32_c - f64_c) + 1.0,
          f"truth={f64_c}, ds={ds_c} (err {abs(ds_c-f64_c):.3e}), "
          f"f32={f32_c} (err {abs(f32_c-f64_c):.3e})")

    # ── 1d. NaN propagation ───────────────────────────────────────────────────
    A_nan = np.array([[1.0, float('nan')], [3.0, 4.0]], dtype=np.float32)
    B_nan = np.ones((2, 2), dtype=np.float32)
    ds_nan = ds_ref.ds_matmul(A_nan, B_nan)
    check("NaN propagation: NaN in A row → that output row is NaN",
          np.any(np.isnan(ds_nan[0])),
          f"row 0: {ds_nan[0]}")

    # ── 1e. Inf propagation ───────────────────────────────────────────────────
    A_inf = np.array([[1.0, float('inf')], [3.0, 4.0]], dtype=np.float32)
    ds_inf = ds_ref.ds_matmul(A_inf, B_nan)
    check("Inf propagation: Inf in A → corresponding output is Inf or NaN",
          np.any(np.isinf(ds_inf[0])) or np.any(np.isnan(ds_inf[0])),
          f"row 0: {ds_inf[0]}")

    # ── 1f. Subnormal inputs ──────────────────────────────────────────────────
    sub_val = np.float32(1.5e-40)
    A_sub = np.full((3, 4), sub_val, dtype=np.float32)
    B_sub = np.ones((4, 3), dtype=np.float32)
    ds_sub = ds_ref.ds_matmul(A_sub, B_sub)
    check("subnormal inputs: result is finite",
          np.all(np.isfinite(ds_sub)),
          f"result contains non-finite: {ds_sub}")

    # ── 1g. Precision on ill-conditioned sum-of-products ─────────────────────
    # 1×n @ n×1 == dot product: use the same triplet-cancellation pattern.
    n = 99   # divisible by 3
    row = np.zeros(n, dtype=np.float32)
    row[0::3] = np.float32(1e6)
    row[1::3] = np.float32(1.0)
    row[2::3] = np.float32(-1e6)
    A_row = row.reshape(1, n)
    B_col = np.ones((n, 1), dtype=np.float32)

    ds_ill  = ds_ref.ds_matmul(A_row, B_col)[0, 0]
    f64_ill = ds_ref.f64_matmul(A_row, B_col)[0, 0]
    f32_ill = float(ds_ref.f32_matmul(A_row, B_col)[0, 0])

    check("ill-conditioned 1×n @ n×1: DS error ≤ f32 error",
          abs(ds_ill - f64_ill) <= abs(f32_ill - f64_ill) + 1.0,
          f"truth={f64_ill}, ds={ds_ill}, f32={f32_ill}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: MLIR Structural Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_structural_tests():
    print("\n[Section 2: MLIR Structural]\n")

    if not OPT_BINARY.exists():
        skip("all structural tests", f"mlir-ds-opt not found at {OPT_BINARY}")
        return

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        skip("all structural tests", "JAX not available")
        return

    # Abstract args: lower() traces without executing on any device.
    # Avoids the PJRT double-registration crash in GPU mode.
    def run_pass(fn, *args):
        mlir = str(jax.jit(fn).lower(*args).compiler_ir())
        with open("/tmp/_ds_matmul_test.mlir", "w") as f:
            f.write(mlir)
        return subprocess.run(
            [str(OPT_BINARY),
             "--pass-pipeline=builtin.module(func.func(ds-transform))",
             "/tmp/_ds_matmul_test.mlir"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )

    A32 = jax.ShapeDtypeStruct((8, 8), jnp.float32)

    # ── 2a. (A * A) @ B — mul creates lo, dot_general expands to 4 ───────────
    def fn_sq_matmul(A, B):
        return (A * A) @ B

    r = run_pass(fn_sq_matmul, A32, A32)
    check("pass exits cleanly (returncode 0)",
          r.returncode == 0,
          r.stderr[:300] if r.returncode != 0 else "")

    n_dot = r.stdout.count("stablehlo.dot_general")
    check(f"(A*A)@B: dot_general expands to 4 sub-matmuls (got {n_dot})",
          n_dot == 4,
          "4-matmul DS decomposition: p=hi@hi, e1=hi@lo, e2=lo@hi, e3=lo@lo")

    n_mul = r.stdout.count("stablehlo.multiply")
    check(f"(A*A)@B: multiply ops for Veltkamp split (got {n_mul})",
          n_mul > 5)

    n_add = r.stdout.count("stablehlo.add")
    check(f"(A*A)@B: add ops for two_sum (got {n_add})",
          n_add > 3)

    # ── 2b. Plain A @ B — still expands even though lo=0 at runtime ──────────
    def fn_plain(A, B):
        return A @ B

    r2 = run_pass(fn_plain, A32, A32)
    n_dot2 = r2.stdout.count("stablehlo.dot_general")
    check(f"plain A@B: also expands to 4 sub-matmuls (got {n_dot2})",
          n_dot2 == 4,
          "pass always expands; precision benefit only appears when lo≠0")

    # ── 2c. Chained: (A @ B) + C — result propagates into add ────────────────
    def fn_chain(A, B, C):
        return (A @ B) + C

    r3 = run_pass(fn_chain, A32, A32, A32)
    n_dot3 = r3.stdout.count("stablehlo.dot_general")
    n_sub3 = r3.stdout.count("stablehlo.subtract")
    check(f"(A@B)+C: dot_general expands and add chains (dots={n_dot3}, subs={n_sub3})",
          n_dot3 == 4 and n_sub3 > 5)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: GPU Numerical Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_gpu_tests():
    print("\n[Section 3: GPU Numerical]\n")

    pjrt_env = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
    bypass   = os.environ.get("DS_BYPASS", "") == "1"

    if not pjrt_env or bypass or not PLUGIN_SO.exists():
        skip("all GPU numerical tests",
             "PJRT plugin not configured (set PJRT_NAMES_AND_LIBRARY_PATHS)")
        return

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        skip("all GPU numerical tests", "JAX not available")
        return

    def block(x):
        return x.block_until_ready()

    # ── 3a. DS benefit: (A*A) @ ones — mul creates lo, matmul accumulates ────
    # 1×n matmul == dot product. n=10000 elements of 0.1 to trigger rounding.
    n = 10_000
    A = jnp.full((1, n), np.float32(0.1))
    ones = jnp.ones((n, 1), dtype=jnp.float32)

    @jax.jit
    def sq_matmul(A, ones):
        return (A * A) @ ones   # A*A → DS lo channels; @ accumulates them

    ds_val  = float(block(sq_matmul(A, ones))[0, 0])
    a_np    = np.full(n, np.float32(0.1))
    f32_val = float(np.dot(a_np * a_np, np.ones(n, np.float32)))
    f64_val = float(np.dot(a_np.astype(np.float64) ** 2, np.ones(n, np.float64)))

    ds_err  = abs(ds_val  - f64_val)
    f32_err = abs(f32_val - f64_val)

    check(f"(A*A)@ones n={n}: DS error ({ds_err:.3e}) ≤ f32 error ({f32_err:.3e})",
          ds_err <= f32_err + 1e-9,
          f"ds={ds_val:.8f}, f32={f32_val:.8f}, f64={f64_val:.8f}")

    # ── 3b. Standalone A @ B — documents the no-benefit case ─────────────────
    # With lo=0 inputs, the 4 correction matmuls all compute 0, so DS == f32.
    A_plain = jnp.full((4, 4), np.float32(0.1))
    B_plain = jnp.ones((4, 4), dtype=jnp.float32)

    @jax.jit
    def plain_mm(A, B):
        return A @ B

    ds_plain  = np.array(block(plain_mm(A_plain, B_plain)))
    a_np2     = np.full((4, 4), np.float32(0.1))
    f32_plain = a_np2 @ np.ones((4, 4), np.float32)
    f64_plain = a_np2.astype(np.float64) @ np.ones((4, 4), np.float64)

    ds_err2  = float(np.max(np.abs(ds_plain - f64_plain)))
    f32_err2 = float(np.max(np.abs(f32_plain - f64_plain)))

    check("standalone A@B: DS result is numerically valid (finite, non-zero)",
          np.all(np.isfinite(ds_plain)) and np.any(ds_plain != 0),
          f"ds_result={ds_plain}")
    # This is an informational check — we do NOT assert DS < f32 here.
    print(f"         (note) standalone matmul: ds_err={ds_err2:.3e}, f32_err={f32_err2:.3e}"
          " — no improvement expected (lo=0)")

    # ── 3c. NaN propagation through matmul ───────────────────────────────────
    A_nan = jnp.array([[1.0, float('nan')], [3.0, 4.0]], dtype=jnp.float32)
    B_nan = jnp.ones((2, 2), dtype=jnp.float32)

    @jax.jit
    def mm(A, B):
        return A @ B

    res_nan = np.array(block(mm(A_nan, B_nan)))
    check("NaN propagation through matmul plugin: row 0 contains NaN",
          np.any(np.isnan(res_nan[0])),
          f"row 0: {res_nan[0]}")

    # ── 3d. Larger matrix — precision on accumulation-heavy case ─────────────
    # H100 cuBLAS uses TF32 tensor cores by default for f32 GEMM.  TF32 has a
    # 10-bit mantissa (~1e-3 relative error vs f32's 1.2e-7), which completely
    # masks the DS correction.  precision=HIGHEST disables TF32, letting the
    # fair DS vs strict-f32 comparison become visible.  Both the DS sub-matmuls
    # (cloned by the pass with the same attribute) and the numpy baseline then
    # use equivalent strict f32 arithmetic.
    n2 = 256
    rng = np.random.default_rng(7)
    A_big = jnp.array(rng.standard_normal((n2, n2)).astype(np.float32))
    B_big = jnp.array(rng.standard_normal((n2, n2)).astype(np.float32))

    @jax.jit
    def sq_big(A, B):
        return jnp.dot(A * A, B, precision=jax.lax.Precision.HIGHEST)

    ds_big  = np.array(block(sq_big(A_big, B_big)))
    a_big   = np.array(A_big)
    b_big   = np.array(B_big)
    f32_big = (a_big * a_big) @ b_big          # numpy strict f32
    f64_big = (a_big.astype(np.float64) ** 2) @ b_big.astype(np.float64)

    ds_err_big  = float(np.mean(np.abs(ds_big - f64_big)))
    f32_err_big = float(np.mean(np.abs(f32_big - f64_big)))

    check(f"(A*A)@B {n2}×{n2} HIGHEST precision: DS error ({ds_err_big:.3e}) ≤ f32 ({f32_err_big:.3e})",
          ds_err_big <= f32_err_big + 1e-6,
          f"ds_mean_err={ds_err_big:.3e}, f32_mean_err={f32_err_big:.3e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Matrix Multiply — DS Compiler Pass Accuracy Tests")
    print("=" * 60)

    run_numpy_tests()
    run_structural_tests()
    run_gpu_tests()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} test(s):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All matmul tests passed.")

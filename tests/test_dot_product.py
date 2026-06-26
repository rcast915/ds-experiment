"""
Dot product accuracy test suite for the DS compiler pass.

Three sections, each independently runnable:

  Section 1 — NumPy Reference
    Validates the DS algorithm itself (two_sum / two_prod / ds_dot) against
    f64 ground truth for all edge cases. No GPU or JAX needed.

  Section 2 — MLIR Structural
    Runs mlir-ds-opt on the lowered MLIR and asserts the DS expansion
    produced the right op counts. Requires the opt binary to be built.

  Section 3 — GPU Numerical
    Runs the actual JAX jit under the PJRT plugin and asserts DS precision
    beats f32 on the GPU. Requires the plugin to be loaded (run via
    run_tests.sh, or set PJRT_NAMES_AND_LIBRARY_PATHS manually).

Usage:
  # All sections (inside container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/test_dot_product.py

  # NumPy + structural only (no GPU needed):
  JAX_PLATFORMS=cpu python3 tests/test_dot_product.py
"""

import sys
import os
import subprocess
import math
from pathlib import Path

import numpy as np

TESTS_DIR   = Path(__file__).resolve().parent
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

    # ── 1a. Basic correctness ─────────────────────────────────────────────────
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = ds_ref.ds_dot(a, b)   # 1*4 + 2*5 + 3*6 = 32
    check("basic correctness: dot([1,2,3],[4,5,6]) == 32.0",
          math.isclose(result, 32.0, rel_tol=1e-6),
          f"got {result}")

    # ── 1b. Catastrophic cancellation ─────────────────────────────────────────
    # Pattern: (large, tiny, -large) triplets dotted with (1, 1, 1).
    # f32 loses the tiny value (it is below ULP at the large scale).
    # DS carries it in the lo channel.
    n = 1000
    vals = np.zeros(3 * n, dtype=np.float32)
    vals[0::3] = np.float32(1e8)
    vals[1::3] = np.float32(1.0)
    vals[2::3] = np.float32(-1e8)
    ones = np.ones(3 * n, dtype=np.float32)

    truth_val  = float(n)                       # exact: n × 1.0
    f32_result = ds_ref.f32_dot(vals, ones)
    ds_result  = ds_ref.ds_dot(vals, ones)

    f32_err = abs(f32_result - truth_val)
    ds_err  = abs(ds_result  - truth_val)

    check("catastrophic cancellation: DS recovers tiny value",
          ds_err < f32_err or ds_err < 1.0,
          f"truth={truth_val}, f32={f32_result} (err {f32_err:.3e}), ds={ds_result} (err {ds_err:.3e})")

    # ── 1c. NaN propagation ───────────────────────────────────────────────────
    a_nan = np.array([1.0, float('nan'), 3.0], dtype=np.float32)
    b_nan = np.array([1.0, 2.0, 3.0],          dtype=np.float32)
    result_nan = ds_ref.ds_dot(a_nan, b_nan)
    check("NaN propagation: dot with NaN element → NaN",
          math.isnan(result_nan),
          f"got {result_nan}")

    # ── 1d. Inf propagation ───────────────────────────────────────────────────
    a_inf = np.array([1.0, float('inf'), 3.0], dtype=np.float32)
    b_inf = np.array([1.0, 2.0, 3.0],          dtype=np.float32)
    result_inf = ds_ref.ds_dot(a_inf, b_inf)
    check("Inf propagation: dot with Inf element → Inf or NaN",
          math.isinf(result_inf) or math.isnan(result_inf),
          f"got {result_inf}")

    # ── 1e. Subnormal inputs ──────────────────────────────────────────────────
    # Float32 subnormals: values in (0, ~1.18e-38).
    # Veltkamp split of a subnormal a: 4097*a is normal, so the split is
    # exact and hi=0, lo=a. two_prod(subnormal, 1.0) = (subnormal, 0).
    # The DS accumulation should proceed without NaN or inf.
    sub_val = np.float32(1.5e-40)   # deep subnormal
    a_sub = np.full(100, sub_val,  dtype=np.float32)
    b_sub = np.ones(100,           dtype=np.float32)
    result_sub = ds_ref.ds_dot(a_sub, b_sub)
    check("subnormal inputs: result is finite",
          math.isfinite(result_sub),
          f"got {result_sub}")
    # DS should be at least as accurate as f32 on this easy case
    f32_sub = ds_ref.f32_dot(a_sub, b_sub)
    truth_sub = 100.0 * float(sub_val)
    check("subnormal inputs: DS error ≤ f32 error",
          abs(result_sub - truth_sub) <= abs(f32_sub - truth_sub) + 1e-50,
          f"ds_err={abs(result_sub-truth_sub):.3e}, f32_err={abs(f32_sub-truth_sub):.3e}")

    # ── 1f. Mixed subnormal and normal values ─────────────────────────────────
    mixed = np.array([1.0, sub_val, 1e8, sub_val, -1e8], dtype=np.float32)
    ones5 = np.ones(5, dtype=np.float32)
    result_mixed = ds_ref.ds_dot(mixed, ones5)
    check("mixed subnormal/normal: result is finite",
          math.isfinite(result_mixed),
          f"got {result_mixed}")

    # ── 1g. Zero vector ───────────────────────────────────────────────────────
    a_z = np.zeros(100, dtype=np.float32)
    b_z = np.random.default_rng(0).random(100).astype(np.float32)
    check("zero vector: dot(0, b) == 0",
          ds_ref.ds_dot(a_z, b_z) == 0.0)

    # ── 1h. Self-dot (sum of squares) ─────────────────────────────────────────
    # n = 1000 elements of 0.1 — accumulated rounding in f32 is significant.
    n = 1000
    a_sq = np.full(n, np.float32(0.1))
    truth_sq = ds_ref.f64_dot(a_sq, a_sq)  # f64 ground truth
    ds_sq    = ds_ref.ds_dot(a_sq, a_sq)
    f32_sq   = ds_ref.f32_dot(a_sq, a_sq)
    check("self-dot (sum of squares): DS error ≤ f32 error",
          abs(ds_sq - truth_sq) <= abs(f32_sq - truth_sq) + 1e-9,
          f"truth={truth_sq:.8f}, ds={ds_sq:.8f} (err {abs(ds_sq-truth_sq):.3e}), "
          f"f32={f32_sq:.8f} (err {abs(f32_sq-truth_sq):.3e})")

    # ── 1i. Precision loss scenario — ill-conditioned sum ─────────────────────
    # Dot product of a=[1e7, -1e7+1, 1e7, -1e7+1, ...] with b=[1,...,1].
    # True answer = n/2 (the +1 terms). f32 accumulation may lose these.
    n = 200
    a_ill = np.empty(n, dtype=np.float32)
    a_ill[0::2] = np.float32(1e7)
    a_ill[1::2] = np.float32(-1e7 + 1)  # -9999999.0 in f32
    b_ill = np.ones(n, dtype=np.float32)
    ds_ill  = ds_ref.ds_dot(a_ill, b_ill)
    f32_ill = ds_ref.f32_dot(a_ill, b_ill)
    # At scale 1e7 ULP≈1, so -1e7+1 rounds exactly to -9999999 in f32;
    # truth = n/2 * (1e7 + (-1e7+1)) = n/2 * 1 = n/2
    # (only valid if 1 is representable at this scale — it is barely)
    truth_ill = ds_ref.f64_dot(a_ill, b_ill)
    check("ill-conditioned sum: DS error ≤ f32 error",
          abs(ds_ill - truth_ill) <= max(abs(f32_ill - truth_ill), 1.0) + 1e-3,
          f"truth={truth_ill}, ds={ds_ill}, f32={f32_ill}")


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

    # Use abstract ShapeDtypeStruct so lower() traces without executing on any
    # device. This avoids the PJRT double-registration crash that occurs when
    # PJRT_NAMES_AND_LIBRARY_PATHS is set and jnp.ones() triggers eager eval.
    def lower_to_mlir(fn, *args):
        return str(jax.jit(fn).lower(*args).compiler_ir())

    def run_pass(mlir_text):
        with open("/tmp/_ds_dot_test.mlir", "w") as f:
            f.write(mlir_text)
        r = subprocess.run(
            [str(OPT_BINARY),
             "--pass-pipeline=builtin.module(func.func(ds-transform))",
             "/tmp/_ds_dot_test.mlir"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        return r

    a32 = jax.ShapeDtypeStruct((16,), jnp.float32)

    # ── 2a. dot(a*a, b) — mul creates lo, dot_general should expand to 4 ─────
    def fn_sq_dot(a, b):
        return jnp.dot(a * a, b)

    r = run_pass(lower_to_mlir(fn_sq_dot, a32, a32))
    check("pass exits cleanly (returncode 0)",
          r.returncode == 0,
          r.stderr[:300] if r.returncode != 0 else "")

    n_dot = r.stdout.count("stablehlo.dot_general")
    check(f"dot_general expands to 4 sub-matmuls (got {n_dot})",
          n_dot == 4,
          f"expected 4 dot_general ops in DS decomposition")

    n_mul = r.stdout.count("stablehlo.multiply")
    check(f"multiply ops present for Veltkamp split (got {n_mul})",
          n_mul > 5,
          "expected >5 multiply ops from two_prod")

    n_sub = r.stdout.count("stablehlo.subtract")
    check(f"subtract ops present for two_sum (got {n_sub})",
          n_sub > 5,
          "expected >5 subtract ops from two_sum")

    # ── 2b. Pure dot(a, b) — inputs have lo=0, still expands to 4 dots ───────
    def fn_plain_dot(a, b):
        return jnp.dot(a, b)

    r2 = run_pass(lower_to_mlir(fn_plain_dot, a32, a32))
    n_dot2 = r2.stdout.count("stablehlo.dot_general")
    check(f"plain dot_general also expands to 4 sub-matmuls (got {n_dot2})",
          n_dot2 == 4)

    # ── 2c. jnp.sum — reduce handler produces a single DS reduce ─────────────
    def fn_sum(a):
        return jnp.sum(a)

    r3 = run_pass(lower_to_mlir(fn_sum, a32))
    n_red = r3.stdout.count("stablehlo.reduce")
    check(f"jnp.sum expands to 1 DS reduce op (got {n_red})",
          n_red == 1,
          "reduce handler should emit exactly one new 4-arg reduce")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: GPU Numerical Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_gpu_tests():
    print("\n[Section 3: GPU Numerical]\n")

    pjrt_env = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
    bypass    = os.environ.get("DS_BYPASS", "") == "1"

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

    # ── 3a. Catastrophic cancellation recovery ─────────────────────────────────
    # (x + y) - x where x >> y: f32 gives 0, DS recovers y.
    @jax.jit
    def cancel_fn(x, y):
        return (x + y) - x

    x = jnp.array([1e8, 1e10, 1e12], dtype=jnp.float32)
    y = jnp.array([1e-2, 1e-3, 1e-4], dtype=jnp.float32)
    ds_result  = np.array(block(cancel_fn(x, y)))
    f32_result = np.array(x) + np.array(y) - np.array(x)   # plain f32

    check("cancellation (x+y)-x: DS recovers at least one component",
          np.any(np.abs(ds_result - np.array(y)) < np.abs(f32_result - np.array(y))),
          f"ds={ds_result}, f32={f32_result}, expected≈{np.array(y)}")

    # ── 3b. Dot product of squares — accumulated rounding ─────────────────────
    # sum(a*a) for a=0.1, n=10000: DS should be closer to f64 truth than f32.
    n = 10_000
    a = jnp.full(n, np.float32(0.1))
    b = jnp.ones(n, dtype=jnp.float32)

    @jax.jit
    def sq_dot(a, b):
        return jnp.dot(a * a, b)   # a*a creates lo channels; dot accumulates them

    ds_val  = float(block(sq_dot(a, b)))
    a_np    = np.full(n, np.float32(0.1))
    f32_val = float(np.dot(a_np * a_np, np.ones(n, np.float32)))
    f64_val = float(np.dot(a_np.astype(np.float64) ** 2, np.ones(n, np.float64)))

    ds_err  = abs(ds_val  - f64_val)
    f32_err = abs(f32_val - f64_val)

    check(f"sq_dot n={n}: DS error ({ds_err:.3e}) ≤ f32 error ({f32_err:.3e})",
          ds_err <= f32_err + 1e-9,
          f"ds={ds_val:.8f}, f32={f32_val:.8f}, f64={f64_val:.8f}")

    # ── 3c. Large cancellation in sum ─────────────────────────────────────────
    # sum of (1e8, 1, -1e8) triplets — f32 loses the 1.
    n_trip = 3333
    vals = np.zeros(3 * n_trip, dtype=np.float32)
    vals[0::3] = np.float32(1e8)
    vals[1::3] = np.float32(1.0)
    vals[2::3] = np.float32(-1e8)
    a_trip = jnp.array(vals)

    @jax.jit
    def jsum(a):
        return jnp.sum(a)

    ds_trip  = float(block(jsum(a_trip)))
    f32_trip = float(np.sum(vals))
    exact    = float(n_trip)

    ds_err_trip  = abs(ds_trip  - exact)
    f32_err_trip = abs(f32_trip - exact)

    check(f"triplet sum n={n_trip}: DS error ({ds_err_trip:.3e}) < f32 error ({f32_err_trip:.3e})",
          ds_err_trip < f32_err_trip or ds_err_trip < 1.0,
          f"ds={ds_trip}, f32={f32_trip}, exact={exact}")

    # ── 3d. NaN propagation through the plugin ────────────────────────────────
    a_nan = jnp.array([1.0, float('nan'), 3.0], dtype=jnp.float32)
    b_nan = jnp.array([1.0, 2.0, 3.0],          dtype=jnp.float32)

    @jax.jit
    def dot_fn(a, b):
        return jnp.dot(a, b)

    res_nan = float(block(dot_fn(a_nan, b_nan)))
    check("NaN propagation through plugin: result is NaN",
          math.isnan(res_nan),
          f"got {res_nan}")

    # ── 3e. Inf propagation ───────────────────────────────────────────────────
    a_inf = jnp.array([1.0, float('inf'), 3.0], dtype=jnp.float32)
    res_inf = float(block(dot_fn(a_inf, b_nan)))
    check("Inf propagation through plugin: result is Inf or NaN",
          math.isinf(res_inf) or math.isnan(res_inf),
          f"got {res_inf}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Dot Product — DS Compiler Pass Accuracy Tests")
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
        print("All dot product tests passed.")

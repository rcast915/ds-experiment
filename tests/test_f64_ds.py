"""
Float64 → DS accuracy tests.

The primary use case: replace slow f64 GPU arithmetic with DS-f32 pairs
(two float32 values whose sum equals the true f64 value).  On H100, cuBLAS
f64 GEMM is ~30× slower than f32; DS-f32 uses four f32 GEMMs (~4× f32 cost)
for a net speedup of roughly 7–8× over native f64 at near-f64 precision
(~48-bit effective mantissa vs f64's 53 bits).

Three sections:

  Section 1 — NumPy Reference
    Verifies the f64 → DS split and DS arithmetic against f64 ground truth.
    No GPU or JAX needed.

  Section 2 — MLIR Structural
    Lowers f64 JAX functions to StableHLO and verifies that ds-transform
    replaces f64 ops with ConvertOps + f32 DS sequences.

  Section 3 — GPU Numerical
    Runs actual f64 JAX JIT under the PJRT plugin and asserts that DS-f32
    results are close to native f64 and much better than naive f32.

Usage:
  # Full suite (inside container, after ds_setup.sh):
  JAX_ENABLE_X64=1 \\
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/test_f64_ds.py

  # NumPy + structural only (no GPU):
  JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python3 tests/test_f64_ds.py
"""

import re
import sys
import os
import subprocess
import math
from pathlib import Path

import numpy as np

# Must be set before JAX is imported anywhere in this process.
os.environ.setdefault("JAX_ENABLE_X64", "1")

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))

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
# Section 1: NumPy Reference — f64 split + DS arithmetic
# ═══════════════════════════════════════════════════════════════════════════════

def _split_f64(x):
    """Split scalar f64 into DS f32 pair (hi, lo), matching emitFromFloat."""
    hi = np.float32(x)
    lo = np.float32(np.float64(x) - np.float64(hi))
    return hi, lo


def run_numpy_tests():
    print("\n[Section 1: NumPy Reference — f64 split]\n")

    import ds_ref

    # ── 1a. Split fidelity ────────────────────────────────────────────────────
    # hi + lo (in f64) must reconstruct x within DS precision (~2^-46 rel).
    x = np.float64(1.23456789012345e10)
    hi, lo = _split_f64(x)
    recon   = np.float64(hi) + np.float64(lo)
    rel_err = abs(recon - x) / abs(x)
    check("f64 split: hi + lo reconstructs x  (rel err < 2^-46)",
          rel_err < 2.0 ** -46,
          f"hi={hi}, lo={lo}, recon={recon}, rel_err={rel_err:.3e}")

    # ── 1b. DS add of two f64-split values ≈ their f64 sum ───────────────────
    xv, yv = np.float64(1e10), np.float64(1.0)
    xh, xl = _split_f64(xv)
    yh, yl = _split_f64(yv)
    rh, rl = ds_ref.ds_add(xh, xl, yh, yl)
    ds_sum  = np.float64(rh) + np.float64(rl)
    f64_sum = xv + yv
    rel_err = abs(ds_sum - f64_sum) / abs(f64_sum)
    check("DS add of f64-split values ≈ native f64 add  (rel err < 2^-46)",
          rel_err < 2.0 ** -46,
          f"DS={ds_sum}, f64={f64_sum}, rel_err={rel_err:.3e}")

    # ── 1c. DS sum precision vs f32 sum ───────────────────────────────────────
    # Sum of n f64-split values should be closer to f64 truth than f32.
    n    = 1000
    vals = np.linspace(0.1, 1.0, n, dtype=np.float64)
    truth = float(np.sum(vals))

    acc_h, acc_l = np.float32(0.0), np.float32(0.0)
    for v in vals:
        vh, vl = _split_f64(v)
        acc_h, acc_l = ds_ref.ds_add(acc_h, acc_l, vh, vl)
    ds_val  = float(acc_h) + float(acc_l)
    f32_val = float(np.sum(vals.astype(np.float32)))

    check(f"DS sum n={n}: DS error ({abs(ds_val-truth):.3e}) ≤ f32 error ({abs(f32_val-truth):.3e})",
          abs(ds_val - truth) <= abs(f32_val - truth) + 1e-9,
          f"DS={ds_val:.10f}, f32={f32_val:.10f}, f64={truth:.10f}")

    # ── 1d. Cancellation: DS recovers small y when x >> y ─────────────────────
    # (x + y) - x with x=1e8, y=1e-2.  y is below f32 ULP at x=1e8 so naive
    # f32 gives 0; DS two_sum captures y in the lo channel.
    xv, yv = np.float64(1e8), np.float64(1e-2)
    xh, xl = _split_f64(xv)
    yh, yl = _split_f64(yv)
    sh, sl = ds_ref.ds_add(xh, xl, yh, yl)
    rh, rl = ds_ref.ds_sub(sh, sl, xh, xl)
    ds_cancel  = float(rh) + float(rl)
    f32_cancel = float(np.float32(xv) + np.float32(yv) - np.float32(xv))
    check("cancellation (x+y)-x, x=1e8, y=1e-2: DS recovers y (err < 1e-3)",
          abs(ds_cancel - float(yv)) < 1e-3,
          f"DS={ds_cancel:.6f}, f32={f32_cancel}, expected≈{yv}")
    check("cancellation: DS much better than naive f32",
          abs(ds_cancel - float(yv)) < abs(f32_cancel - float(yv)) + 1e-6,
          f"DS err={abs(ds_cancel-float(yv)):.3e}, "
          f"f32 err={abs(f32_cancel-float(yv)):.3e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: MLIR Structural — f64 inputs
# ═══════════════════════════════════════════════════════════════════════════════

def run_structural_tests():
    print("\n[Section 2: MLIR Structural — f64 inputs]\n")

    if not OPT_BINARY.exists():
        skip("all structural tests", f"mlir-ds-opt not found at {OPT_BINARY}")
        return

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        skip("all structural tests", "JAX not available")
        return

    # Must be set before any JAX computation.
    jax.config.update("jax_enable_x64", True)

    def lower_to_mlir(fn, *args):
        # ShapeDtypeStruct avoids device execution and the PJRT double-
        # registration crash (see handoff.md §Important Caveats).
        return str(jax.jit(fn).lower(*args).compiler_ir())

    def run_pass(mlir_text, tmp="/tmp/_ds_f64_test.mlir", extra_env=None):
        with open(tmp, "w") as f:
            f.write(mlir_text)
        env = dict(os.environ)
        env.pop("DS_F32_MATMUL", None)
        env.pop("DS_GEMM_KERNEL_AVAILABLE", None)   # set by the plugin if JAX loaded it
        env.update(extra_env or {})
        return subprocess.run(
            [str(OPT_BINARY),
             "--pass-pipeline=builtin.module(inline,func.func(ds-transform))",
             tmp],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env,
        )

    a64 = jax.ShapeDtypeStruct((16,), jnp.float64)

    # ── 2a. f64 add → ConvertOps + f32 DS arithmetic ─────────────────────────
    def fn_add(x, y):
        return x + y

    mlir_input = lower_to_mlir(fn_add, a64, a64)
    check("input MLIR has f64 ops  (JAX_ENABLE_X64 active)",
          "f64" in mlir_input,
          "expected f64 types in lowered MLIR before the pass")

    r = run_pass(mlir_input)
    check("f64 add: pass exits cleanly",
          r.returncode == 0,
          r.stderr[:400] if r.returncode != 0 else "")

    n_conv = r.stdout.count("stablehlo.convert")
    check(f"f64 add → DS: stablehlo.convert present (got {n_conv}, expected ≥ 4)",
          n_conv >= 4,
          "expected f64↔f32 ConvertOps at function entry and exit")

    n_sub = r.stdout.count("stablehlo.subtract")
    check(f"f64 add → DS: subtract ops for two_sum (got {n_sub}, expected ≥ 4)",
          n_sub >= 4,
          "expected subtract ops from DS two_sum expansion")

    # ── 2b. f64 sum reduction → DS reduce ────────────────────────────────────
    def fn_sum(x):
        return jnp.sum(x)

    r2 = run_pass(lower_to_mlir(fn_sum, a64))
    check("f64 sum: pass exits cleanly",
          r2.returncode == 0,
          r2.stderr[:400] if r2.returncode != 0 else "")
    n_red = len(re.findall(r"stablehlo\.reduce\b(?!_)", r2.stdout))  # not reduce_precision
    check(f"f64 sum → 1 DS reduce op (got {n_red})",
          n_red == 1,
          "reduce handler should emit exactly one 4-arg DS reduce")

    # ── 2c. f64 matmul: native f64 by default; 4 f32 GEMMs if opted in ─────
    # The 4-GEMM decomposition is only f32-accurate (each sub-GEMM rounds its
    # products and partial sums to f32), so f64 matmuls stay native f64
    # unless DS_F32_MATMUL=1 asks for the fast, f32-accurate version.
    A64 = jax.ShapeDtypeStruct((32, 32), jnp.float64)

    def fn_matmul(A, B):
        return A @ B

    # (A*A) @ B: the matmul's operand is DS-computed, so the fallback has to
    # recombine it to f64 rather than reuse the original f64 multiply.
    def fn_mul_matmul(A, B):
        return (A * A) @ B

    mm_in = lower_to_mlir(fn_matmul, A64, A64)
    r3 = run_pass(mm_in)
    check("f64 matmul: pass exits cleanly",
          r3.returncode == 0,
          r3.stderr[:400] if r3.returncode != 0 else "")
    f64_dots = re.findall(r"stablehlo\.dot_general.*tensor<32x32xf64>", r3.stdout)
    n_dot = r3.stdout.count("stablehlo.dot_general")
    check(f"f64 matmul (default) → 1 native f64 dot_general (got {n_dot}, f64-typed {len(f64_dots)})",
          n_dot == 1 and len(f64_dots) == 1,
          "default: f64 matmuls must stay native f64")

    r3b = run_pass(lower_to_mlir(fn_mul_matmul, A64, A64))
    n_mul64 = len(re.findall(r"stablehlo\.multiply.*tensor<32x32xf64>", r3b.stdout))
    n_dot_b = r3b.stdout.count("stablehlo.dot_general")
    check(f"f64 (A*A)@B (default) → DS multiply feeds 1 native f64 dot "
          f"(dots {n_dot_b}, f64 multiplies left {n_mul64})",
          r3b.returncode == 0 and n_dot_b == 1 and n_mul64 == 0,
          "the original f64 multiply should be replaced by DS, recombined into the f64 dot")

    kernel_env = {"DS_GEMM_KERNEL_AVAILABLE": "1"}
    r5 = run_pass(mm_in, extra_env=kernel_env)
    n_cc = len(re.findall(r"stablehlo\.custom_call @ds_gemm|call_target_name = \"ds_gemm\"", r5.stdout))
    n_dot5 = r5.stdout.count("stablehlo.dot_general")
    check(f"f64 matmul with the DS kernel registered → 1 custom_call @ds_gemm, no dot_general "
          f"(calls {n_cc}, dots {n_dot5})",
          r5.returncode == 0 and n_cc == 1 and n_dot5 == 0,
          r5.stderr[:400] if r5.returncode != 0 else "expected the matmul to become the DS kernel call")

    r6 = run_pass(lower_to_mlir(fn_mul_matmul, A64, A64), extra_env=kernel_env)
    n_cc6 = len(re.findall(r"stablehlo\.custom_call @ds_gemm|call_target_name = \"ds_gemm\"", r6.stdout))
    check(f"f64 (A*A)@B with the DS kernel → DS multiply feeds the kernel call (calls {n_cc6})",
          r6.returncode == 0 and n_cc6 == 1 and "stablehlo.dot_general" not in r6.stdout,
          r6.stderr[:400] if r6.returncode != 0 else "")

    def fn_batched(A, B):
        return jnp.einsum("bij,bjk->bik", A, B)
    Ab = jax.ShapeDtypeStruct((2, 16, 16), jnp.float64)
    r7 = run_pass(lower_to_mlir(fn_batched, Ab, Ab), extra_env=kernel_env)
    check("batched f64 matmul with the DS kernel → falls back to native f64 (no kernel call)",
          r7.returncode == 0 and "ds_gemm" not in r7.stdout and "stablehlo.dot_general" in r7.stdout,
          r7.stderr[:400] if r7.returncode != 0 else "the kernel only handles plain A[M,K]@B[K,N]")

    r4 = run_pass(mm_in, extra_env={"DS_F32_MATMUL": "1"})
    n_dot4 = r4.stdout.count("stablehlo.dot_general")
    check(f"f64 matmul with DS_F32_MATMUL=1 → 4 f32 sub-matmuls (got {n_dot4})",
          r4.returncode == 0 and n_dot4 == 4,
          "opt-in: 4-matmul DS decomposition")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: GPU Numerical — DS-f32 precision with f64 inputs
# ═══════════════════════════════════════════════════════════════════════════════

def run_gpu_tests():
    print("\n[Section 3: GPU Numerical — f64 inputs via DS plugin]\n")

    pjrt_env = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
    bypass   = os.environ.get("DS_BYPASS", "") == "1"

    if not pjrt_env or bypass or not PLUGIN_SO.exists():
        skip("all GPU numerical tests",
             "PJRT plugin not configured (set PJRT_NAMES_AND_LIBRARY_PATHS)")
        return

    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        skip("all GPU numerical tests", "JAX not available")
        return

    def block(x):
        return x.block_until_ready()

    # ── 3a. Cancellation with f64 inputs ─────────────────────────────────────
    # The plugin intercepts f64 ops and converts to DS-f32 internally.
    # Result should recover y where naive f32 gives 0.
    @jax.jit
    def cancel_fn(x, y):
        return (x + y) - x

    # Each pair must fit in DS's ~48 significant bits. 1e12 is not exactly
    # representable in f32 (it splits to hi=999999995904, lo=4096), and the
    # ulp of that lo word is 2^-11 ~ 4.9e-4, so a y far below that cannot
    # survive even in exact DS arithmetic; y = 1.0 can.
    x_np = np.array([1e8,  1e10, 1e12], dtype=np.float64)
    y_np = np.array([1e-2, 1e-3, 1.0], dtype=np.float64)
    x    = jnp.array(x_np)
    y    = jnp.array(y_np)

    ds_result  = np.array(block(cancel_fn(x, y)))
    f32_result = (x_np.astype(np.float32) + y_np.astype(np.float32)
                  - x_np.astype(np.float32)).astype(np.float64)

    ds_rel_err  = np.max(np.abs(ds_result  - y_np) / np.abs(y_np))
    f32_rel_err = np.max(np.abs(f32_result - y_np) / np.abs(y_np))

    check("f64 cancellation: DS-f32 recovers y  (max rel err < 0.01)",
          ds_rel_err < 0.01,
          f"DS={ds_result}, expected≈{y_np}, max_rel_err={ds_rel_err:.3e}")
    check("f64 cancellation: DS better than naive f32",
          ds_rel_err < f32_rel_err,
          f"DS max_rel_err={ds_rel_err:.3e}, f32 max_rel_err={f32_rel_err:.3e}")

    # ── 3b. DS-f32 sum precision close to f64 ────────────────────────────────
    n = 10_000
    a = jnp.full(n, np.float64(0.1))

    @jax.jit
    def fn_sum(a):
        return jnp.sum(a)

    ds_val  = float(block(fn_sum(a)))
    f64_val = float(np.sum(np.full(n, 0.1, np.float64)))
    f32_val = float(np.sum(np.full(n, np.float32(0.1))))

    ds_err  = abs(ds_val  - f64_val)
    f32_err = abs(f32_val - f64_val)

    check(f"f64 sum n={n}: DS error ({ds_err:.3e}) ≤ f32 error ({f32_err:.3e})",
          ds_err <= f32_err + 1e-6,
          f"DS={ds_val:.8f}, f32={f32_val:.8f}, f64={f64_val:.8f}")

    # ── 3c. f64 matmul stays f64-accurate through the plugin ─────────────────
    rng = np.random.default_rng(3)
    A_np = rng.standard_normal((256, 256))
    B_np = rng.standard_normal((256, 256))

    @jax.jit
    def mm_fn(A, B):
        return (A * A) @ B

    got = np.array(block(mm_fn(jnp.array(A_np), jnp.array(B_np))))
    truth = (A_np * A_np) @ B_np
    rel = float(np.linalg.norm(got - truth) / np.linalg.norm(truth))
    # The plugin sets this with setenv() after Python started, so os.environ
    # (a snapshot) can't see it; read the process environment directly.
    import ctypes
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    path = ("DS kernel" if libc.getenv(b"DS_GEMM_KERNEL_AVAILABLE") == b"1"
            else "native f64 fallback")
    check(f"f64 (A*A)@B through plugin [{path}]: f64-accurate (rel err {rel:.2e} < 1e-12)",
          rel < 1e-12,
          "a value near 1e-7 or 3e-4 means the f32-accurate decomposition ran")

    # ── 3d. NaN propagation ───────────────────────────────────────────────────
    a_nan = jnp.array([1.0, float('nan'), 3.0], dtype=jnp.float64)
    b_nan = jnp.array([1.0, 2.0,         3.0], dtype=jnp.float64)

    @jax.jit
    def dot_fn(a, b):
        return jnp.dot(a, b)

    res_nan = float(block(dot_fn(a_nan, b_nan)))
    check("NaN propagation through plugin (f64 input): result is NaN",
          math.isnan(res_nan),
          f"got {res_nan}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Float64 → DS Compiler Pass Accuracy Tests")
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
        print("All f64 DS tests passed.")

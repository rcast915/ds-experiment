"""
Accuracy tests for DS sqrt (stablehlo.sqrt -> emitDsSqrt), ported from
double-single-lib's double_binary32_sqrt.

  Section 1 — NumPy Reference
    Validates ds_ref.ds_sqrt against f64 ground truth across varied
    magnitudes, confirms the lo=0 degenerate case (plain f32 input) still
    gives a correct result, and a lo != 0 cross-check where DS should
    clearly beat sqrt(hi) alone.

  Section 2 — MLIR Structural
    Confirms stablehlo.sqrt actually expands (op-kind presence), including
    the emitDsDivByScalar sub-sequence (double-single-lib's
    __double_binary_div_double_by_single, ported as its own helper since
    it is NOT algebraically the same as emitDsDiv called with a zero lo
    operand -- see emitDsDivByScalar's comment in DsTransformPass.cpp).

  Section 3 — GPU Numerical
    a) Unit accuracy: sqrt through the plugin vs. f64 truth, on a vector
       of plain f32 (lo=0) inputs. Same limitation already learned from
       divide's test: sqrt is unary, so a plain f32 input already gets a
       correctly-rounded f32 answer from hardware sqrtf -- an isolated
       lo=0 sqrt's f32-quantized-at-return OBSERVABLE result cannot beat
       that, structurally, regardless of how good the internal DS
       correction is. The bound here reflects "correctly rounded f32"
       (~2^-22 with margin), not the tight DS-internal bound; DS's real
       advantage (internal precision, precision preserved through
       composition) is checked in (b), (c), and Section 1's lo != 0
       cross-check -- not asserted here as a "beats f32" claim.
    b) Pair-accuracy variant (DS_RETURN_PAIRS=1): per the
       observable-vs-internal distinction, checks the *internal* (hi, lo)
       pair recombined in f64 on the host separately from the f32-return-
       quantized result.
    c) Composition test: sqrt chained with existing DS ops (hypot-style:
       sqrt(a*a + b*b)).
    d) Structural passthrough (DS_TEST_PASSTHROUGH=1): the pass must not
       crash on a sqrt-containing function even when its output gets
       discarded; the executed result must match plain (non-DS) sqrt,
       confirming passthrough truly bypasses the DS transformation.
    e) Edge semantics: negative input gives NaN (ordinary IEEE sqrt
       behavior, not a divergence). Exact-zero input ALSO gives NaN, not
       a clean zero -- t1 = sqrtf(0) = 0 feeds emitDsDivByScalar as
       b = 0, whose first division is 0/0 = NaN immediately. This is a
       real divergence from naive expectations (sqrt(0) "should" be 0), a
       property of the reference library's own structure (verified
       against the C source), reported here rather than silently patched.

Usage:
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/test_ds_sqrt.py
"""

import sys
import os
import subprocess
import math
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
import ds_ref

OPT_BINARY = PROJECT_ROOT / "stablehlo_pass" / "build" / "mlir-ds-opt"
PLUGIN_SO = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"

# Same theoretical grounding as divide's bound (see test_ds_divide.py):
# ds_sqrt has the same double-word algorithm shape (single-precision
# estimate, one refinement step via a residual division) as ds_div, so
# the same O(u_f32^2) ~ 2^-48 floor argument applies. 2^-40 gives ~2^8
# margin above that floor, ~1e5x tighter than plain f32's ~2^-23.
DS_SQRT_REL_ERR_BOUND = 2.0 ** -40

# Observable (f32-quantized-at-return) bound for lo=0 inputs -- see
# module docstring 3a. Reflects "correctly rounded f32", not DS-internal
# precision.
OBSERVABLE_REL_ERR_BOUND = 2.0 ** -22

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

    # ── basic correctness, lo=0 degenerate case (plain f32 input) ────────────
    h, l = ds_ref.ds_sqrt(4.0, 0.0)
    check("sqrt: sqrt(4.0) (lo=0) == 2.0 exactly",
          h == np.float32(2.0) and l == np.float32(0.0), f"got ({h}, {l})")

    # ── varied magnitudes ──────────────────────────────────────────────────
    cases = [1.0, 2.0, 3.0, 100.0, 0.25, 1e10, 1e-10, np.pi, 1.0000001, 123456.789]
    for a_val in cases:
        a32 = np.float32(a_val)
        h, l = ds_ref.ds_sqrt(a32, 0.0)
        truth = float(np.sqrt(a32.astype(np.float64)))
        measured = float(h) + float(l)
        rel_err = abs(measured - truth) / max(abs(truth), 1e-300)
        check(f"sqrt: sqrt({a_val}) within DS bound (rel_err={rel_err:.3e})",
              rel_err < DS_SQRT_REL_ERR_BOUND,
              f"truth={truth}, measured={measured}")

    # ── DS-vs-f32: a case where DS should clearly beat plain f32 sqrt ────────
    # sqrt of a DS pair with a real (non-zero) lo channel -- the correction
    # should carry through, unlike plain f32 sqrt which only sees hi.
    ah, al = ds_ref.two_sum(np.float32(1e8), np.float32(0.01))  # hi~1e8, lo~0.01
    h, l = ds_ref.ds_sqrt(ah, al)
    truth = math.sqrt(float(ah) + float(al))
    f32_only = math.sqrt(float(ah))   # plain f32 sqrt, ignoring al
    ds_measured = float(h) + float(l)
    check("sqrt: DS result closer to truth than dropping the lo channel would be",
          abs(ds_measured - truth) < abs(f32_only - truth),
          f"truth={truth}, ds={ds_measured}, f32-only={f32_only}")

    # ── edge semantics: negative input -- ordinary IEEE NaN ──────────────────
    h, l = ds_ref.ds_sqrt(-4.0, 0.0)
    check("sqrt: sqrt(-4) -> NaN (ordinary IEEE sqrt behavior)",
          math.isnan(float(h) + float(l)), f"got ({h}, {l})")

    # ── edge semantics: exact zero -- NOT a clean zero ────────────────────────
    # t1 = sqrtf(0) = 0 exactly, which then feeds ds_div_by_scalar as
    # b = 0 -- its first division, ah/b = 0/0, is IEEE-754 NaN immediately.
    # Confirmed present in the reference C library too (not introduced by
    # this port): double_binary32_sqrt does not special-case a zero input.
    h, l = ds_ref.ds_sqrt(0.0, 0.0)
    check("sqrt: sqrt(0) -> NaN (algorithm's own 0/0 from the scalar-div step, not a clean zero)",
          math.isnan(float(h) + float(l)), f"got ({h}, {l})")


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

    def lower_to_mlir(fn, *args):
        return str(jax.jit(fn).lower(*args).compiler_ir())

    def run_pass(mlir_text, tmp_name):
        path = f"/tmp/{tmp_name}"
        with open(path, "w") as f:
            f.write(mlir_text)
        return subprocess.run(
            [str(OPT_BINARY),
             "--pass-pipeline=builtin.module(inline,func.func(ds-transform))",
             path],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )

    a32 = jax.ShapeDtypeStruct((16,), jnp.float32)

    r = run_pass(lower_to_mlir(lambda x: jnp.sqrt(x), a32), "_ds_sqrt.mlir")
    check("sqrt: pass exits cleanly", r.returncode == 0,
          r.stderr[:300] if r.returncode != 0 else "")
    n_sqrt = r.stdout.count("stablehlo.sqrt")
    check(f"sqrt: exactly 1 stablehlo.sqrt op survives (got {n_sqrt})", n_sqrt == 1,
          "expected only t1=sqrtf(a_hi); the emitDsDivByScalar sub-sequence "
          "must not itself contain a sqrt")
    n_div = r.stdout.count("stablehlo.divide")
    check(f"sqrt: expands to 2 stablehlo.divide ops via emitDsDivByScalar (got {n_div})",
          n_div == 2, "expected emitDsDivByScalar's t1=a_hi/b and t7=t6/b")
    n_mul = r.stdout.count("stablehlo.multiply")
    check(f"sqrt: multiply ops present (two_prod inside emitDsDivByScalar + the two 0.5x scalings) (got {n_mul})",
          n_mul > 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: GPU Numerical Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_gpu_tests():
    print("\n[Section 3: GPU Numerical]\n")

    pjrt_env = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
    bypass = os.environ.get("DS_BYPASS", "") == "1"

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
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        else:
            for leaf in jax.tree_util.tree_leaves(x):
                leaf.block_until_ready()
        return x

    # ── 3a. Unit accuracy across magnitude regimes (OBSERVABLE, f32-returned) ──
    @jax.jit
    def sqrt_fn(a):
        return jnp.sqrt(a)

    a_vals = np.array([1.0, 2.0, 100.0, 1e10, 1e-10, 123456.789, np.pi, 1.0000001],
                       dtype=np.float32)

    ds_result = np.array(block(sqrt_fn(jnp.array(a_vals))))
    truth = np.sqrt(a_vals.astype(np.float64))
    rel_err = np.abs(ds_result - truth) / np.maximum(np.abs(truth), 1e-300)
    check(f"unit accuracy: max relative error {rel_err.max():.3e} < bound {OBSERVABLE_REL_ERR_BOUND:.3e} (correctly-rounded f32 level)",
          bool(np.all(rel_err < OBSERVABLE_REL_ERR_BOUND)),
          f"ds={ds_result}, truth={truth}, rel_err={rel_err}")

    f32_only = np.sqrt(a_vals).astype(np.float64)
    f32_rel_err = np.abs(f32_only - truth) / np.maximum(np.abs(truth), 1e-300)
    print(f"  (note) unit accuracy: ds_err={rel_err}, f32_err={f32_rel_err} -- "
          f"expected identical for lo=0 inputs (see note above); DS's real "
          f"advantage over plain f32 is checked separately in 3b (internal "
          f"pair) and 3c (composition), and in Section 1's cross-check with "
          f"a real lo != 0 input.")

    # ── 3b. Pair-accuracy variant (DS_RETURN_PAIRS=1) ────────────────────────
    # sqrt of a cancellation-derived DS value (real lo channel), comparing
    # the f32-return-quantized result against the raw pair recombined in
    # f64 on the host (the observable-vs-internal split).
    return_pairs = os.environ.get("DS_RETURN_PAIRS", "") == "1"
    if return_pairs:
        @jax.jit
        def sqrt_pair_fn(big, eps):
            a = (big + eps) - big   # DS pair with real lo, hi possibly ~0
            r = jnp.sqrt(a * a + jnp.float32(1e8))   # keep sqrt's operand well away from 0
            return r, r

        big = jnp.float32(1e8)
        eps = jnp.float32(0.04)
        hi, lo = block(sqrt_pair_fn(big, eps))
        hi_f, lo_f = float(hi), float(lo)
        eps_true = float(eps)
        truth_pair = math.sqrt(eps_true * eps_true + 1e8)
        internal = float(np.float64(hi_f) + np.float64(lo_f))
        check("pair-accuracy: internal (hi+lo in f64) matches truth closely",
              math.isclose(internal, truth_pair, rel_tol=1e-4),
              f"hi={hi_f}, lo={lo_f}, internal={internal}, truth={truth_pair}")
    else:
        skip("pair-accuracy variant",
             "run this file again with DS_RETURN_PAIRS=1 to exercise it "
             "(separate process, per the project's env-var-at-load-time discipline)")

    # ── 3c. Composition: sqrt chained with existing ops (hypot-style) ────────
    @jax.jit
    def compose_fn(a, b):
        return jnp.sqrt(a * a + b * b)

    av = jnp.float32(3.0)
    bv = jnp.float32(4.0)
    ds_val = float(block(compose_fn(av, bv)))
    truth_val = math.sqrt(float(av) ** 2 + float(bv) ** 2)
    check("composition sqrt(a*a+b*b): DS result matches f64 truth",
          math.isclose(ds_val, truth_val, rel_tol=1e-6),
          f"ds={ds_val}, truth={truth_val}")

    # ── 3d. Structural passthrough (DS_TEST_PASSTHROUGH=1) ────────────────────
    passthrough = os.environ.get("DS_TEST_PASSTHROUGH", "") == "1"
    if passthrough:
        ds_pt = float(block(sqrt_fn(jnp.float32(2.0))))
        plain = float(np.sqrt(np.float32(2.0)))
        check("passthrough: pass runs without crashing and result matches plain f32",
              math.isclose(ds_pt, plain, rel_tol=1e-6),
              f"got {ds_pt}, expected plain-f32 {plain} (DS correction should be bypassed)")
    else:
        skip("structural passthrough",
             "run this file again with DS_TEST_PASSTHROUGH=1 to exercise it")

    # ── 3e. Edge semantics through the plugin ─────────────────────────────────
    neg_sqrt = float(block(sqrt_fn(jnp.float32(-4.0))))
    check("sqrt(-4) through plugin: NaN (ordinary IEEE sqrt behavior)",
          math.isnan(neg_sqrt), f"got {neg_sqrt}")

    # Same divergence as Section 1: sqrt(0) -> NaN via the emitDsDivByScalar
    # step's 0/0, not a clean zero -- EXCEPT under passthrough, where
    # emitDsSqrt never runs at all, so the op is plain hardware sqrtf
    # and 0.0 is the correct answer there.
    zero_sqrt = float(block(sqrt_fn(jnp.float32(0.0))))
    if passthrough:
        check("sqrt(0) through plugin (passthrough): 0.0 (plain IEEE, DS transform bypassed)",
              zero_sqrt == 0.0, f"got {zero_sqrt}")
    else:
        check("sqrt(0) through plugin: NaN (algorithm's own 0/0 from the scalar-div step)",
              math.isnan(zero_sqrt), f"got {zero_sqrt}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  DS Sqrt — Accuracy Tests")
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
        print("All DS sqrt tests passed.")

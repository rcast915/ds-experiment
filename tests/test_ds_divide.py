"""
Accuracy tests for DS divide (stablehlo.divide -> emitDsDiv), based on
double-single-lib's double_binary32_div WITH ONE DELIBERATE CORRECTION.

  Section 1 — NumPy Reference
    Validates ds_ref.ds_div against f64 ground truth across varied
    magnitudes (near 1, large/small exponents, sign mixes), and confirms
    the lo=0 degenerate case (plain f32 inputs) still gives a correct
    result, not just an efficient one.

    CORRECTED LIBRARY BUG (see ds_ref.ds_div's docstring and
    DsTransformPass.cpp's emitDsDiv comment for the full derivation):
    double_binary32_div, as literally written, drops the TwoProd/two_mul
    residual (t3) from its correction sum -- confirmed to be a genuine
    bug (not a design choice) via the library's own sibling routine,
    which keeps the equivalent term. This is not limited to a non-zero
    divisor lo: t3 captures t1's own single-precision rounding error, so
    it matters for every division, including plain lo=0/lo=0 cases with
    no cancellation at all. This port includes the `-t3` term, restoring
    ~2^-48-class double-word accuracy (confirmed empirically: worst case
    ~1.7e-14 over 500k random trials) instead of the ~2^-23-class
    (f32-ULP-level) accuracy the literal library formula achieves.
    DS_DIV_REL_ERR_BOUND reflects this corrected, deep accuracy -- the
    same double-word class the other DS ops (add/sub/mul/sqrt) achieve.

  Section 2 — MLIR Structural
    Confirms stablehlo.divide actually expands (op-kind presence).

  Section 3 — GPU Numerical
    a) Unit accuracy: DS divide through the plugin vs. f64 truth, on a
       vector of plain f32 (lo=0) inputs. Because the function returns a
       single f32 (quantized at func.return) and plain f32 division is
       already IEEE correctly-rounded, an isolated lo=0 division cannot
       observably beat plain f32 -- unrelated to and unchanged by the t3
       fix (that fix restores divide's *internal* accuracy; it can't make
       an already-correctly-rounded single f32 result any more correct).
       The bound here reflects "correctly rounded f32" (~2^-22 with
       margin), not DS_DIV_REL_ERR_BOUND -- DS's real, measurable
       advantage is checked in (b)/(c) and Section 1's lo != 0
       cross-check, where the corrected internal accuracy is observable.
    b) Pair-accuracy variant (DS_RETURN_PAIRS=1): per Experiment 3's
       observable-vs-internal distinction, checks the *internal* (hi, lo)
       pair recombined in f64 on the host separately from the f32-return-
       quantized result. Uses a loose rel_tol (not DS_DIV_REL_ERR_BOUND)
       since this checks a cancellation-recovery case, not divide's own
       precision limit.
    c) Composition test: divide chained with existing DS ops.
    d) Structural passthrough (DS_TEST_PASSTHROUGH=1): the pass must not
       crash on a divide-containing function even when its output gets
       discarded; the executed result must match plain (non-DS) division,
       confirming passthrough truly bypasses the DS transformation.
    e) Edge semantics: b_hi == 0 (div by zero) produces NaN, not a clean
       +inf -- the algorithm's own two_prod(b_hi=0, t1=inf) step hits the
       IEEE-754 0*inf indeterminate form. Confirmed present in the
       reference C library too (not introduced by this port), so this is
       reported as a documented divergence from naive IEEE expectations,
       not silently special-cased away.

Usage:
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/test_ds_divide.py
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

# Relative-error bound for divide's corrected internal accuracy -- see
# module docstring. Same theoretical grounding as the other DS ops:
# double-word division of this Newton/Sterbenz shape is O(u_f32^2), i.e.
# O(2^-48) relative error; 2^-40 gives ~2^8 margin above that floor while
# still being ~1e5x tighter than plain f32's ~2^-23. Confirmed empirically
# (500k random trials, both lo=0 and DS-pair-divisor cases): worst case
# ~1.7e-14, comfortably inside this bound.
DS_DIV_REL_ERR_BOUND = 2.0 ** -40

# Bound for the *observable* (f32-quantized-at-return) result on plain
# lo=0 inputs -- see Section 3a note. Unrelated to the t3 fix: plain f32
# division is already correctly-rounded, so an isolated lo=0 division's
# observable result is capped at roughly f32-ULP level regardless of how
# accurate divide's internal computation is.
DS_DIV_OBSERVABLE_REL_ERR_BOUND = 2.0 ** -22

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

    # ── basic correctness, lo=0 degenerate case (plain f32 inputs) ───────────
    h, l = ds_ref.ds_div(6.0, 0.0, 3.0, 0.0)
    check("div: 6.0/3.0 (lo=0 both sides) == 2.0 exactly",
          h == np.float32(2.0) and l == np.float32(0.0), f"got ({h}, {l})")

    # ── varied magnitudes: near 1, large, small, mixed signs ─────────────────
    # NOTE: magnitude spread is bounded so the *quotient* also stays inside
    # f32 range (~1.2e-38 .. 3.4e38), not just the two operands -- t1 =
    # a_hi/b_hi is computed in f32 as an actual intermediate, so e.g.
    # 1e30/1e-30 = 1e60 would overflow t1 itself to +inf before the DS
    # correction ever runs. That's a real f32 range limit, not something
    # divide's algorithm can paper over.
    cases = [
        (1.0, 3.0), (7.0, 2.0), (-5.0, 4.0), (5.0, -4.0), (-5.0, -4.0),
        (1e15, 1e-15), (1e-15, 1e15), (123456.789, 0.00012345),
        (np.pi, np.e), (1.0000001, 0.9999999),
    ]
    for a_val, b_val in cases:
        a32 = np.float32(a_val)
        b32 = np.float32(b_val)
        h, l = ds_ref.ds_div(a32, 0.0, b32, 0.0)
        truth = float(a32.astype(np.float64) / b32.astype(np.float64))
        measured = float(h) + float(l)
        rel_err = abs(measured - truth) / max(abs(truth), 1e-300)
        check(f"div: {a_val}/{b_val} within DS bound (rel_err={rel_err:.3e})",
              rel_err < DS_DIV_REL_ERR_BOUND,
              f"truth={truth}, measured={measured}")

    # ── divisor-has-real-lo: regression test for the corrected t3 term ───────
    # This case uses a divisor with a genuine non-zero lo (via two_sum) --
    # the case that most directly exercises the b_lo/t4 path. Prior to the
    # `-t3` correction this landed near the measured worst case for the
    # literal library formula (~1.16e-7, ~2^-23); with the correction it
    # should be back at DS_DIV_REL_ERR_BOUND's double-word class.
    b_hi, b_lo = ds_ref.two_sum(np.float32(-988.1205970793266),
                                 np.float32(-0.07021472772014681))
    a_val = np.float32(8252.133301039248)
    h, l = ds_ref.ds_div(a_val, np.float32(0.0), b_hi, b_lo)
    truth = float(a_val) / (float(b_hi) + float(b_lo))
    measured = float(h) + float(l)
    rel_err = abs(measured - truth) / abs(truth)
    check(f"div: divisor with real lo -- within DS bound (rel_err={rel_err:.3e})",
          rel_err < DS_DIV_REL_ERR_BOUND,
          f"truth={truth}, measured={measured}, b_hi={b_hi}, b_lo={b_lo}")
    # Also confirm the divisor's lo channel isn't dropped *entirely* (a
    # worse bug than the documented limitation): ignoring b_lo should be
    # measurably less accurate than what ds_div actually returns.
    h0, l0 = ds_ref.ds_div(a_val, np.float32(0.0), b_hi, np.float32(0.0))
    err_with_lo = abs(measured - truth)
    err_without_lo = abs((float(h0) + float(l0)) - truth)
    check("div: divisor's lo channel is actually used (not silently dropped)",
          err_with_lo < err_without_lo,
          f"err_with_lo={err_with_lo:.3e}, err_without_lo={err_without_lo:.3e}")

    # ── DS-vs-f32: a case where DS should clearly beat plain f32 division ────
    # Divide a DS pair with a real (non-zero) lo channel by a plain scalar --
    # the correction should carry through, unlike plain f32 which only sees hi.
    ah, al = ds_ref.two_sum(np.float32(1e8), np.float32(0.01))  # hi~1e8, lo~0.01
    h, l = ds_ref.ds_div(ah, al, 4.0, 0.0)
    truth = (float(ah) + float(al)) / 4.0
    f32_only = float(ah) / 4.0   # what plain f32 division would give, ignoring al
    ds_measured = float(h) + float(l)
    check("div: DS result closer to truth than dropping the lo channel would be",
          abs(ds_measured - truth) < abs(f32_only - truth),
          f"truth={truth}, ds={ds_measured}, f32-only={f32_only}")

    # ── edge semantics: divide by zero -- NOT a clean +inf ────────────────────
    # t1 = 1/0 = +inf is correct IEEE division, but the algorithm's next step
    # needs two_prod(b_hi, t1) = two_prod(0, inf), and IEEE-754 defines
    # 0 * inf = NaN (indeterminate form). This is a property of
    # double_binary32_div's own internal structure -- confirmed present in
    # the reference C library too, not something introduced by this port.
    # The library does not special-case b_hi == 0, so this port reports the
    # divergence (NaN, not inf) rather than silently patching it in.
    h, l = ds_ref.ds_div(1.0, 0.0, 0.0, 0.0)
    check("div: 1/0 -> NaN (algorithm's own 0*inf indeterminate form, not a clean IEEE +inf)",
          math.isnan(float(h) + float(l)), f"got ({h}, {l})")

    h, l = ds_ref.ds_div(float('nan'), 0.0, 2.0, 0.0)
    check("div: NaN/2 -> NaN",
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

    r = run_pass(lower_to_mlir(lambda x, y: x / y, a32, a32), "_ds_div.mlir")
    check("divide: pass exits cleanly", r.returncode == 0,
          r.stderr[:300] if r.returncode != 0 else "")
    n_div = r.stdout.count("stablehlo.divide")
    check(f"divide: expands to 2 stablehlo.divide ops (got {n_div})", n_div == 2,
          "expected t1=a_hi/b_hi and t8=t7/b_hi")
    n_mul = r.stdout.count("stablehlo.multiply")
    check(f"divide: multiply ops present for two_prod (got {n_mul})", n_mul > 3)
    n_sub = r.stdout.count("stablehlo.subtract")
    check(f"divide: subtract ops present (got {n_sub})", n_sub > 3)


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

    # ── 3a. Unit accuracy across magnitude regimes ────────────────────────────
    @jax.jit
    def div_fn(a, b):
        return a / b

    a_vals = np.array([1.0, 7.0, -5.0, 1e20, 1e-20, 123456.789, np.pi, 1.0000001],
                       dtype=np.float32)
    b_vals = np.array([3.0, 2.0, 4.0, 1e-10, 1e10, 0.00012345, np.e, 0.9999999],
                       dtype=np.float32)

    # NOTE on the bound used here: a_vals/b_vals are plain f32 inputs, i.e.
    # lo=0 on both sides. div_fn returns a single f32 (recombined via
    # emitToFloat at func.return), so the *observable* result is quantized
    # back to one f32 -- and plain f32 division is already IEEE correctly-
    # rounded, so there is no more-accurate single-f32 answer to find for
    # an isolated lo=0 division, regardless of how accurate divide's
    # internal (hi, lo) computation is. This is unrelated to and unchanged
    # by the t3 correction (see module docstring) -- it's an output-
    # quantization limit, the same "no benefit when lo=0" pattern already
    # documented for standalone matmul in this project. DS's real,
    # measurable advantage is checked in 3b (internal pair, where the t3
    # correction IS observable), 3c (composition), and Section 1's lo != 0
    # cross-check.
    ds_result = np.array(block(div_fn(jnp.array(a_vals), jnp.array(b_vals))))
    truth = a_vals.astype(np.float64) / b_vals.astype(np.float64)
    rel_err = np.abs(ds_result - truth) / np.maximum(np.abs(truth), 1e-300)
    check(f"unit accuracy: max relative error {rel_err.max():.3e} < bound {DS_DIV_OBSERVABLE_REL_ERR_BOUND:.3e} (correctly-rounded f32 level)",
          bool(np.all(rel_err < DS_DIV_OBSERVABLE_REL_ERR_BOUND)),
          f"ds={ds_result}, truth={truth}, rel_err={rel_err}")

    f32_only = (a_vals / b_vals).astype(np.float64)
    f32_rel_err = np.abs(f32_only - truth) / np.maximum(np.abs(truth), 1e-300)
    print(f"  (note) unit accuracy: ds_err={rel_err}, f32_err={f32_rel_err} -- "
          f"expected identical for lo=0 inputs (see note above); DS's real "
          f"advantage over plain f32 is checked separately in 3b (internal "
          f"pair) and 3c (composition), and in Section 1's cross-check with "
          f"a real lo != 0 input.")

    # ── 3b. Pair-accuracy variant (DS_RETURN_PAIRS=1) ────────────────────────
    # Divide a cancellation-derived DS value (real lo channel) by a constant,
    # comparing the f32-return-quantized result against the raw pair
    # recombined in f64 on the host -- same observable-vs-internal split as
    # Experiment 3.
    return_pairs = os.environ.get("DS_RETURN_PAIRS", "") == "1"
    if return_pairs:
        @jax.jit
        def div_pair_fn(big, eps):
            a = (big + eps) - big   # DS pair with real lo, hi possibly ~0
            r = a / jnp.float32(4.0)
            return r, r

        big = jnp.float32(1e8)
        eps = jnp.float32(0.04)
        hi, lo = block(div_pair_fn(big, eps))
        hi_f, lo_f = float(hi), float(lo)
        truth_pair = float(eps) / 4.0
        internal = float(np.float64(hi_f) + np.float64(lo_f))
        check("pair-accuracy: internal (hi+lo in f64) matches truth closely",
              math.isclose(internal, truth_pair, rel_tol=1e-4),
              f"hi={hi_f}, lo={lo_f}, internal={internal}, truth={truth_pair}")
    else:
        skip("pair-accuracy variant",
             "run this file again with DS_RETURN_PAIRS=1 to exercise it "
             "(separate process, per the project's env-var-at-load-time discipline)")

    # ── 3c. Composition: divide chained with existing ops ─────────────────────
    @jax.jit
    def compose_fn(a, b, c):
        return (a * a) / (b + c)   # multiply feeds divide feeds through add

    av = jnp.float32(3.0)
    bv = jnp.float32(1e8)
    cv = jnp.float32(1e8)   # b+c cancels-ish in magnitude terms with a*a below
    ds_val = float(block(compose_fn(av, bv, cv)))
    truth_val = (float(av) ** 2) / (float(bv) + float(cv))
    check("composition (a*a)/(b+c): DS result matches f64 truth",
          math.isclose(ds_val, truth_val, rel_tol=1e-5),
          f"ds={ds_val}, truth={truth_val}")

    # ── 3d. Structural passthrough (DS_TEST_PASSTHROUGH=1) ────────────────────
    passthrough = os.environ.get("DS_TEST_PASSTHROUGH", "") == "1"
    if passthrough:
        ds_pt = float(block(div_fn(jnp.float32(7.0), jnp.float32(3.0))))
        plain = float(np.float32(7.0)) / float(np.float32(3.0))
        check("passthrough: pass runs without crashing and result matches plain f32",
              math.isclose(ds_pt, plain, rel_tol=1e-6),
              f"got {ds_pt}, expected plain-f32 {plain} (DS correction should be bypassed)")
    else:
        skip("structural passthrough",
             "run this file again with DS_TEST_PASSTHROUGH=1 to exercise it")

    # ── 3e. Edge semantics through the plugin ─────────────────────────────────
    # Same divergence as Section 1: b_hi == 0 hits two_prod(0, inf) = NaN
    # inside the algorithm itself, not a clean +inf. Confirmed on real
    # hardware here, matching the NumPy reference -- EXCEPT under
    # DS_TEST_PASSTHROUGH=1, where emitDsDiv never runs at all, so the op
    # is plain hardware f32 division and +inf is the correct answer there.
    zero_div = float(block(div_fn(jnp.float32(1.0), jnp.float32(0.0))))
    if passthrough:
        check("div by zero through plugin (passthrough): +inf (plain IEEE, DS transform bypassed)",
              math.isinf(zero_div), f"got {zero_div}")
    else:
        check("div by zero through plugin: NaN (algorithm's own 0*inf indeterminate form)",
              math.isnan(zero_div), f"got {zero_div}")

    nan_div = float(block(div_fn(jnp.float32('nan'), jnp.float32(2.0))))
    check("NaN/2 through plugin: NaN",
          math.isnan(nan_div), f"got {nan_div}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  DS Divide — Accuracy Tests")
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
        print("All DS divide tests passed.")

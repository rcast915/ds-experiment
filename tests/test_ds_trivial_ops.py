"""
Accuracy tests for the DS pass's trivial/comparison ops: negate, abs,
compare, select, maximum, minimum. Same three-section structure as
test_dot_product.py.

  Section 1 — NumPy Reference
    Validates ds_ref.py's ds_negate/ds_abs/ds_compare/ds_max/ds_min against
    hand-worked edge cases, including the case that distinguishes the
    library's actual abs condition (hi >= -lo) from the naive sign(hi)
    guess. No GPU or JAX needed.

  Section 2 — MLIR Structural
    Runs mlir-ds-opt and asserts each new op actually expands (op-kind
    presence, not brittle exact counts -- see inline notes on which counts
    are hand-derived-and-asserted-exactly vs. loosely bounded).

  Section 3 — GPU Numerical
    Runs the actual JAX jit under the PJRT plugin. negate/abs are exact
    (no new rounding introduced, so DS should reproduce f64 truth exactly
    up to the f32-return quantization); compare/select/maximum/minimum are
    tested specifically on values that are equal in f32 but distinguishable
    in DS, to confirm the hi-then-lo tiebreak actually matters and works.

  DS_WARN_UNSUPPORTED diagnostic
    Structural-only (no GPU needed): confirms the pass warns on stderr for
    a known-unsupported op (sin) when DS_WARN_UNSUPPORTED=1, and stays
    silent both for the same op when unset and for supported ops when set.

Usage:
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/test_ds_trivial_ops.py

  JAX_PLATFORMS=cpu python3 tests/test_ds_trivial_ops.py   # NumPy + structural only
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

    # ── negate ─────────────────────────────────────────────────────────────
    h, l = ds_ref.ds_negate(np.float32(3.5), np.float32(1e-6))
    check("negate: (-3.5, -1e-6)", h == np.float32(-3.5) and l == np.float32(-1e-6),
          f"got ({h}, {l})")

    h2, l2 = ds_ref.ds_negate(*ds_ref.ds_negate(np.float32(3.5), np.float32(1e-6)))
    check("negate: double negation is identity",
          h2 == np.float32(3.5) and l2 == np.float32(1e-6), f"got ({h2}, {l2})")

    # ── abs: the case that distinguishes the real algorithm from a naive guess ──
    # hi == 0.0, lo < 0: value is actually negative, but sign(hi) alone would
    # say non-negative. The library's hi >= -lo condition catches this and
    # negates -- to (-0.0, +0.001), NOT a "canonical" (+0.001, ~0) form; the
    # algorithm makes no renormalization promise, it just negates the pair
    # it's given. Check the *represented value* (hi+lo), not the individual
    # components, since -0.0 + 0.001 == 0.001 is correct even though hi
    # itself stays (numerically) zero.
    h, l = ds_ref.ds_abs(np.float32(0.0), np.float32(-0.001))
    check("abs: hi=0.0, lo=-0.001 (value is negative) -> represents +0.001",
          math.isclose(float(h) + float(l), 0.001, rel_tol=1e-5),
          f"got (hi={h}, lo={l}) representing {float(h)+float(l)} -- naive "
          f"sign(hi) would have left this unchanged, representing -0.001")

    # Ordinary positive/negative cases.
    h, l = ds_ref.ds_abs(np.float32(-5.0), np.float32(0.25))
    check("abs: hi=-5.0 (clearly negative) -> negates both",
          h == np.float32(5.0) and l == np.float32(-0.25), f"got ({h}, {l})")

    h, l = ds_ref.ds_abs(np.float32(5.0), np.float32(-0.25))
    check("abs: hi=5.0 (clearly positive) -> unchanged",
          h == np.float32(5.0) and l == np.float32(-0.25), f"got ({h}, {l})")

    # ── compare: hi-first, lo-tiebreak ────────────────────────────────────
    check("compare: hi differs -> decided by hi alone",
          ds_ref.ds_compare(2.0, 100.0, 3.0, -100.0) == -1)

    check("compare: hi tied, lo decides (a<b)",
          ds_ref.ds_compare(2.0, 0.001, 2.0, 0.002) == -1)

    check("compare: hi tied, lo decides (a>b)",
          ds_ref.ds_compare(2.0, 0.002, 2.0, 0.001) == 1)

    check("compare: fully equal",
          ds_ref.ds_compare(2.0, 0.001, 2.0, 0.001) == 0)

    check("compare: NaN is unordered",
          ds_ref.ds_compare(float('nan'), 0.0, 2.0, 0.0) is None)

    # ── maximum / minimum built from compare ──────────────────────────────
    h, l = ds_ref.ds_max(2.0, 0.001, 2.0, 0.002)
    check("max: hi-tied case picks larger lo",
          h == np.float32(2.0) and l == np.float32(0.002), f"got ({h}, {l})")

    h, l = ds_ref.ds_min(2.0, 0.001, 2.0, 0.002)
    check("min: hi-tied case picks smaller lo",
          h == np.float32(2.0) and l == np.float32(0.001), f"got ({h}, {l})")


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

    def run_pass(mlir_text, tmp_name="_ds_trivial_ops_test.mlir", env=None):
        path = f"/tmp/{tmp_name}"
        with open(path, "w") as f:
            f.write(mlir_text)
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        r = subprocess.run(
            [str(OPT_BINARY),
             "--pass-pipeline=builtin.module(inline,func.func(ds-transform))",
             path],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=full_env,
        )
        return r

    a32 = jax.ShapeDtypeStruct((16,), jnp.float32)

    # ── 2a. negate: hand-derived exact count -- emitDsAbs is not involved,
    # so this one is safe to assert exactly: 2 new negates (hi, lo), the
    # original negate erased.
    r = run_pass(lower_to_mlir(lambda x: -x, a32), "_ds_negate.mlir")
    check("negate: pass exits cleanly", r.returncode == 0,
          r.stderr[:300] if r.returncode != 0 else "")
    n_neg = r.stdout.count("stablehlo.negate")
    check(f"negate: expands to 2 stablehlo.negate ops (got {n_neg})", n_neg == 2)

    # ── 2b. abs: hand-derived exact count from emitDsAbs -- 2 negates,
    # 1 compare, 2 selects.
    r = run_pass(lower_to_mlir(lambda x: jnp.abs(x), a32), "_ds_abs.mlir")
    check("abs: pass exits cleanly", r.returncode == 0,
          r.stderr[:300] if r.returncode != 0 else "")
    n_neg = r.stdout.count("stablehlo.negate")
    n_cmp = r.stdout.count("stablehlo.compare")
    n_sel = r.stdout.count("stablehlo.select")
    check(f"abs: 2 negates (got {n_neg})", n_neg == 2)
    check(f"abs: 1 compare (got {n_cmp})", n_cmp == 1)
    check(f"abs: 2 selects (got {n_sel})", n_sel == 2)

    # ── 2c. compare (standalone) -- at least one compare must survive
    # (loosely bounded: exact count depends on comparison direction chosen
    # by JAX's lowering, which this test doesn't pin down).
    r = run_pass(lower_to_mlir(lambda x, y: jnp.greater(x, y), a32, a32), "_ds_gt.mlir")
    check("compare: pass exits cleanly", r.returncode == 0,
          r.stderr[:300] if r.returncode != 0 else "")
    n_cmp = r.stdout.count("stablehlo.compare")
    check(f"compare: expands to >=3 compares (got {n_cmp})", n_cmp >= 3,
          "expected hi-eq + hi-strict + lo-tiebreak compares")

    # ── 2d. maximum -- at least the expected compare+select shape survives.
    r = run_pass(lower_to_mlir(lambda x, y: jnp.maximum(x, y), a32, a32), "_ds_max.mlir")
    check("maximum: pass exits cleanly", r.returncode == 0,
          r.stderr[:300] if r.returncode != 0 else "")
    n_cmp = r.stdout.count("stablehlo.compare")
    n_sel = r.stdout.count("stablehlo.select")
    check(f"maximum: expands to >=3 compares (got {n_cmp})", n_cmp >= 3)
    check(f"maximum: expands to 2 selects (got {n_sel})", n_sel == 2)


# ═══════════════════════════════════════════════════════════════════════════════
# DS_WARN_UNSUPPORTED diagnostic (structural-only, no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════════

def run_warn_unsupported_tests():
    print("\n[DS_WARN_UNSUPPORTED diagnostic]\n")

    if not OPT_BINARY.exists():
        skip("DS_WARN_UNSUPPORTED tests", f"mlir-ds-opt not found at {OPT_BINARY}")
        return

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        skip("DS_WARN_UNSUPPORTED tests", "JAX not available")
        return

    def lower_to_mlir(fn, *args):
        return str(jax.jit(fn).lower(*args).compiler_ir())

    a32 = jax.ShapeDtypeStruct((16,), jnp.float32)
    # sin has no DS routine in double-single-lib and is not implemented --
    # exactly the "known-unsupported op" the task asks to demonstrate this on.
    sin_mlir = lower_to_mlir(lambda x: jnp.sin(x), a32)
    add_mlir = lower_to_mlir(lambda x, y: x + y, a32, a32)

    def run_pass(mlir_text, tmp_name, warn_env):
        path = f"/tmp/{tmp_name}"
        with open(path, "w") as f:
            f.write(mlir_text)
        env = dict(os.environ)
        if warn_env:
            env["DS_WARN_UNSUPPORTED"] = "1"
        else:
            env.pop("DS_WARN_UNSUPPORTED", None)
        return subprocess.run(
            [str(OPT_BINARY),
             "--pass-pipeline=builtin.module(inline,func.func(ds-transform))",
             path],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env,
        )

    r_warn_on = run_pass(sin_mlir, "_ds_warn_sin_on.mlir", warn_env=True)
    check("DS_WARN_UNSUPPORTED=1 on sin(x): warning fires",
          "WARNING" in r_warn_on.stderr and "sin" in r_warn_on.stderr.lower(),
          f"stderr: {r_warn_on.stderr[:300]}")

    r_warn_off = run_pass(sin_mlir, "_ds_warn_sin_off.mlir", warn_env=False)
    check("DS_WARN_UNSUPPORTED unset on sin(x): silent (default unchanged)",
          "WARNING" not in r_warn_off.stderr,
          f"stderr: {r_warn_off.stderr[:300]}")

    r_warn_on_add = run_pass(add_mlir, "_ds_warn_add_on.mlir", warn_env=True)
    check("DS_WARN_UNSUPPORTED=1 on a supported op (add): silent",
          "WARNING" not in r_warn_on_add.stderr,
          f"stderr: {r_warn_on_add.stderr[:300]}")


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
        # Handles both a single array and a pytree of them (cmp_select_fn
        # below returns a 3-tuple) -- matching paper/ds_reeval/common.py's
        # block_until_ready helper.
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        else:
            for leaf in jax.tree_util.tree_leaves(x):
                leaf.block_until_ready()
        return x

    # ── 3a. negate: exact, no new rounding -- DS should match f64 truth to
    # within the f32-return half-ulp floor (same output-quantization story
    # as Experiment 3; negate itself introduces zero additional error).
    @jax.jit
    def neg_fn(x):
        return -x

    x = jnp.array([1e8, -3.25, 1e-30], dtype=jnp.float32)
    ds_result = np.array(block(neg_fn(x)))
    truth = -np.array(x, dtype=np.float64)
    check("negate: DS matches f64 truth exactly (no rounding to introduce)",
          np.allclose(ds_result, truth, rtol=0, atol=0),
          f"ds={ds_result}, truth={truth}")

    # ── 3b. abs: composition test -- chain with a prior DS op (a - a big
    # cancellation) so lo is non-trivial, then abs, confirming dsMap plumbs
    # through mixed sequences and the sign is corrected regardless of which
    # component initially carries it.
    @jax.jit
    def abs_of_diff(x, y):
        return jnp.abs(x - y)

    x2 = jnp.array([1e8, 1e8], dtype=jnp.float32)
    y2 = jnp.array([1e8 + 3.0, 1e8 - 3.0], dtype=jnp.float32)
    ds_result2 = np.array(block(abs_of_diff(x2, y2)))
    truth2 = np.abs(np.array(x2, dtype=np.float64) - np.array(y2, dtype=np.float64))
    check("abs(x - y) composition: DS recovers the correct magnitude",
          np.allclose(ds_result2, truth2, rtol=1e-5),
          f"ds={ds_result2}, truth={truth2} -- plain f32 abs(x-y) would give "
          f"{np.abs(np.array(x2) - np.array(y2))}")

    # ── 3c/3d/3e. compare / select / maximum: values equal in f32 but
    # distinguishable via DS lo-channel cancellation -- (x+eps)-x recovers
    # eps in lo while hi may coincide, exercising the lo-tiebreak specifically.
    eps_a = jnp.float32(1e-2)
    eps_b = jnp.float32(2e-2)
    big = jnp.float32(1e8)

    @jax.jit
    def cmp_select_fn(big, ea, eb):
        a = (big + ea) - big   # DS: hi could be 0 or small, lo carries ea
        b = (big + eb) - big
        greater = a > b
        return jnp.where(greater, a, b), a, b

    ds_max_val, ds_a, ds_b = block(cmp_select_fn(big, eps_a, eps_b))
    ds_max_val, ds_a, ds_b = float(ds_max_val), float(ds_a), float(ds_b)
    check("compare+select: DS recovers a=eps_a, b=eps_b distinctly",
          math.isclose(ds_a, float(eps_a), rel_tol=1e-3) and
          math.isclose(ds_b, float(eps_b), rel_tol=1e-3),
          f"a={ds_a}, b={ds_b} (expected ~{float(eps_a)}, ~{float(eps_b)}) -- "
          f"plain f32 (big+eps)-big would give 0 for both, making this "
          f"comparison meaningless without DS")
    check("compare+select: jnp.where(a>b, a, b) picks the larger (b)",
          math.isclose(ds_max_val, float(eps_b), rel_tol=1e-3),
          f"got {ds_max_val}, expected ~{float(eps_b)}")

    @jax.jit
    def max_fn(big, ea, eb):
        a = (big + ea) - big
        b = (big + eb) - big
        return jnp.maximum(a, b), jnp.minimum(a, b)

    mx, mn = block(max_fn(big, eps_a, eps_b))
    mx, mn = float(mx), float(mn)
    check("maximum/minimum: correctly ordered despite f32-indistinguishable inputs",
          math.isclose(mx, float(eps_b), rel_tol=1e-3) and
          math.isclose(mn, float(eps_a), rel_tol=1e-3),
          f"max={mx} (expected ~{float(eps_b)}), min={mn} (expected ~{float(eps_a)})")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Trivial/Comparison DS Ops — Accuracy Tests")
    print("=" * 60)

    run_numpy_tests()
    run_structural_tests()
    run_warn_unsupported_tests()
    run_gpu_tests()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} test(s):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All trivial-ops tests passed.")

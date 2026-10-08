"""
Accuracy tests for DS exp and log (stablehlo.exponential -> emitDsExp,
stablehlo.log -> emitDsLog), ported from double-single-libm's expds and
logds.

  Section 1 — NumPy Reference
    Validates ds_ref.ds_exp / ds_ref.ds_log against f64 ground truth over
    random inputs with a real lo channel, plus the special cases (NaN,
    +/-Inf, overflow, underflow, negative and zero arguments).

  Section 2 — MLIR Structural
    Confirms stablehlo.exponential and stablehlo.log are fully expanded (no
    native op survives) and that the table reads appear as stablehlo.gather
    (2 for exp: table hi/lo; 3 for log: reciprocal, -log(w) hi/lo).

  Section 3 — GPU Numerical
    a) Unit accuracy: exp/log through the plugin vs. f64 truth on plain f32
       (lo=0) inputs. As with sqrt, the f32-quantized-at-return result of a
       single unary op cannot beat a correctly rounded f32 answer, so the
       bound here is f32-level.
    b) Pair-accuracy variant (DS_RETURN_PAIRS=1): the internal (hi, lo)
       pair recombined in f64 on the host, where the double-single accuracy
       is visible.
    c) Composition: exp(a) * exp(-a) - 1 and log(exp(a)) - a, which are
       ~1e-7 in plain f32 and must be far smaller when the pair survives
       between the ops.
    d) Special cases through the plugin.

Usage:
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/test_ds_exp_log.py
"""

import sys
import os
import re
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

# Measured worst cases of the reference algorithms (bit-identical to the C
# library) over ~1M random double-single inputs: exp 2^-42.9 on results in
# the normal range; log 2^-38.5, reached for arguments within about 1% of
# 1.0 (2^-47.5 for arguments >= 2). Bounds leave ~2 bits of margin.
DS_EXP_REL_ERR_BOUND = 2.0 ** -41
DS_LOG_REL_ERR_BOUND = 2.0 ** -36

# Observable (f32-quantized-at-return) bound for lo=0 inputs.
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


def to_ds(v):
    """Nearest double-single pair to the f64 value v."""
    h = np.float32(v)
    return h, np.float32(float(v) - float(h))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: NumPy Reference Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_numpy_tests():
    print("\n[Section 1: NumPy Reference]\n")
    rng = np.random.default_rng(0)

    def worst_rel_err(fn, truth_fn, xs):
        worst, at = 0.0, None
        for v in xs:
            h, l = to_ds(v)
            rh, rl = fn(h, l)
            truth = truth_fn(float(h) + float(l))
            if truth == 0.0:
                continue
            err = abs(float(rh) + float(rl) - truth) / abs(truth)
            if err > worst:
                worst, at = err, v
        return worst, at

    # ── exp ──────────────────────────────────────────────────────────────────
    h, l = ds_ref.ds_exp(0.0, 0.0)
    check("exp: exp(0) == 1 exactly", h == np.float32(1.0) and l == np.float32(0.0),
          f"got ({h}, {l})")

    for lo, hi in [(-1.0, 1.0), (-20.0, 20.0), (-60.0, 80.0)]:
        w, at = worst_rel_err(ds_ref.ds_exp, math.exp, rng.uniform(lo, hi, 20000))
        check(f"exp: x in [{lo}, {hi}] within DS bound (worst rel_err={w:.3e})",
              w < DS_EXP_REL_ERR_BOUND, f"worst at x={at}")

    h, l = ds_ref.ds_exp(np.nan, 0.0)
    check("exp: exp(NaN) -> NaN", math.isnan(float(h)) and math.isnan(float(l)))
    h, l = ds_ref.ds_exp(np.inf, 0.0)
    check("exp: exp(+Inf) -> +Inf", h == np.inf, f"got ({h}, {l})")
    h, l = ds_ref.ds_exp(-np.inf, 0.0)
    check("exp: exp(-Inf) -> 0", h == 0.0 and l == 0.0, f"got ({h}, {l})")
    h, l = ds_ref.ds_exp(89.0, 0.0)
    check("exp: exp(89) overflows to +Inf", h == np.inf, f"got ({h}, {l})")
    h, l = ds_ref.ds_exp(-104.0, 0.0)
    check("exp: exp(-104) underflows to 0", h == 0.0 and l == 0.0, f"got ({h}, {l})")

    # ── log ──────────────────────────────────────────────────────────────────
    h, l = ds_ref.ds_log(1.0, 0.0)
    check("log: log(1) == 0 exactly", h == 0.0 and l == 0.0, f"got ({h}, {l})")

    log_ranges = [
        ("x in [2, 1e6]", rng.uniform(2.0, 1e6, 20000)),
        ("x in [0.5, 2]", rng.uniform(0.5, 2.0, 20000)),
        ("x log-uniform in [1e-30, 1e30]", np.exp(rng.uniform(-69.0, 69.0, 20000))),
        ("x subnormal", rng.uniform(1e-45, 1e-39, 2000)),
    ]
    for name, xs in log_ranges:
        w, at = worst_rel_err(ds_ref.ds_log, math.log, xs)
        check(f"log: {name} within DS bound (worst rel_err={w:.3e})",
              w < DS_LOG_REL_ERR_BOUND, f"worst at x={at}")

    h, l = ds_ref.ds_log(np.nan, 0.0)
    check("log: log(NaN) -> NaN", math.isnan(float(h)) and math.isnan(float(l)))
    h, l = ds_ref.ds_log(np.inf, 0.0)
    check("log: log(+Inf) -> +Inf", h == np.inf, f"got ({h}, {l})")
    h, l = ds_ref.ds_log(-1.0, 0.0)
    check("log: log(-1) -> NaN", math.isnan(float(h)), f"got ({h}, {l})")
    # Deliberate deviation from the library, whose code returns +Inf here
    # despite its "Return -inf" comment -- see ds_ref.ds_log's docstring.
    h, l = ds_ref.ds_log(0.0, 0.0)
    check("log: log(0) -> -Inf", h == -np.inf and l == 0.0, f"got ({h}, {l})")

    # ── round trip with a real lo channel ────────────────────────────────────
    worst = 0.0
    for v in rng.uniform(-20.0, 20.0, 5000):
        h, l = to_ds(v)
        rh, rl = ds_ref.ds_log(*ds_ref.ds_exp(h, l))
        worst = max(worst, abs(float(rh) + float(rl) - (float(h) + float(l))))
    check(f"log(exp(x)) == x to DS accuracy (worst abs err={worst:.3e})", worst < 1e-10)


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

    def count_gathers(text):
        # The op's dimension_numbers attribute prints as #stablehlo.gather<...>,
        # so count op names only.
        return len(re.findall(r"(?<!#)stablehlo\.gather\b", text))

    def count(pattern, text):
        return len(re.findall(pattern, text))

    for shape, tag in [((16,), "vector"), ((), "scalar"), ((4, 8), "matrix")]:
        a32 = jax.ShapeDtypeStruct(shape, jnp.float32)

        r = run_pass(lower_to_mlir(lambda x: jnp.exp(x), a32), f"_ds_exp_{tag}.mlir")
        check(f"exp ({tag}): pass exits cleanly", r.returncode == 0,
              r.stderr[:300] if r.returncode != 0 else "")
        n_exp = count(r"stablehlo\.exponential\b", r.stdout)
        check(f"exp ({tag}): no native stablehlo.exponential survives (got {n_exp})",
              n_exp == 0)
        n_gather = count_gathers(r.stdout)
        check(f"exp ({tag}): 2 table gathers, hi and lo (got {n_gather})", n_gather == 2)

        r = run_pass(lower_to_mlir(lambda x: jnp.log(x), a32), f"_ds_log_{tag}.mlir")
        check(f"log ({tag}): pass exits cleanly", r.returncode == 0,
              r.stderr[:300] if r.returncode != 0 else "")
        n_log = count(r"stablehlo\.log\b", r.stdout)
        check(f"log ({tag}): no native stablehlo.log survives (got {n_log})", n_log == 0)
        n_gather = count_gathers(r.stdout)
        check(f"log ({tag}): 3 table gathers, reciprocal and -log(w) hi/lo (got {n_gather})",
              n_gather == 3)


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
        for leaf in jax.tree_util.tree_leaves(x):
            leaf.block_until_ready()
        return x

    @jax.jit
    def exp_fn(a):
        return jnp.exp(a)

    @jax.jit
    def log_fn(a):
        return jnp.log(a)

    rng = np.random.default_rng(1)

    # ── 3a. Unit accuracy (OBSERVABLE, f32-returned) ─────────────────────────
    x_exp = rng.uniform(-20.0, 20.0, 4096).astype(np.float32)
    got = np.array(block(exp_fn(jnp.array(x_exp)))).astype(np.float64)
    truth = np.exp(x_exp.astype(np.float64))
    rel = np.abs(got - truth) / truth
    check(f"exp unit accuracy: max rel err {rel.max():.3e} < {OBSERVABLE_REL_ERR_BOUND:.3e}",
          bool(np.all(rel < OBSERVABLE_REL_ERR_BOUND)),
          f"worst at x={x_exp[rel.argmax()]}")

    x_log = np.exp(rng.uniform(-30.0, 30.0, 4096)).astype(np.float32)
    got = np.array(block(log_fn(jnp.array(x_log)))).astype(np.float64)
    truth = np.log(x_log.astype(np.float64))
    err = np.abs(got - truth) / np.maximum(np.abs(truth), 1e-3)
    check(f"log unit accuracy: max rel err {err.max():.3e} < {OBSERVABLE_REL_ERR_BOUND:.3e}",
          bool(np.all(err < OBSERVABLE_REL_ERR_BOUND)),
          f"worst at x={x_log[err.argmax()]}")

    # The GPU result must be the NumPy reference's pair, rounded to f32.
    ref = np.array([float(np.float32(float(h) + float(l)))
                    for h, l in (ds_ref.ds_exp(v, 0.0) for v in x_exp[:256])])
    got = np.array(block(exp_fn(jnp.array(x_exp[:256])))).astype(np.float64)
    check("exp: plugin result equals the NumPy reference (bitwise, 256 inputs)",
          bool(np.array_equal(ref, got)),
          f"{int(np.sum(ref != got))} of 256 differ")
    ref = np.array([float(np.float32(float(h) + float(l)))
                    for h, l in (ds_ref.ds_log(v, 0.0) for v in x_log[:256])])
    got = np.array(block(log_fn(jnp.array(x_log[:256])))).astype(np.float64)
    check("log: plugin result equals the NumPy reference (bitwise, 256 inputs)",
          bool(np.array_equal(ref, got)),
          f"{int(np.sum(ref != got))} of 256 differ")

    # ── 3b. Pair-accuracy variant (DS_RETURN_PAIRS=1) ────────────────────────
    if os.environ.get("DS_RETURN_PAIRS", "") == "1":
        @jax.jit
        def exp_pair_fn(a):
            r = jnp.exp(a)
            return r, r

        @jax.jit
        def log_pair_fn(a):
            r = jnp.log(a)
            return r, r

        hi, lo = block(exp_pair_fn(jnp.array(x_exp)))
        internal = np.array(hi, np.float64) + np.array(lo, np.float64)
        truth = np.exp(x_exp.astype(np.float64))
        rel = np.abs(internal - truth) / truth
        check(f"exp pair accuracy: max rel err {rel.max():.3e} < {DS_EXP_REL_ERR_BOUND:.3e}",
              bool(np.all(rel < DS_EXP_REL_ERR_BOUND)),
              f"worst at x={x_exp[rel.argmax()]}")

        hi, lo = block(log_pair_fn(jnp.array(x_log)))
        internal = np.array(hi, np.float64) + np.array(lo, np.float64)
        truth = np.log(x_log.astype(np.float64))
        rel = np.abs(internal - truth) / np.maximum(np.abs(truth), 1e-3)
        check(f"log pair accuracy: max rel err {rel.max():.3e} < {DS_LOG_REL_ERR_BOUND:.3e}",
              bool(np.all(rel < DS_LOG_REL_ERR_BOUND)),
              f"worst at x={x_log[rel.argmax()]}")
    else:
        skip("pair-accuracy variant",
             "run this file again with DS_RETURN_PAIRS=1 to exercise it")

    # ── 3c. Composition: the pair must survive between ops ───────────────────
    @jax.jit
    def exp_cancel_fn(a):
        return jnp.exp(a) * jnp.exp(-a) - jnp.float32(1.0)

    @jax.jit
    def log_exp_fn(a):
        return jnp.log(jnp.exp(a)) - a

    a = rng.uniform(-10.0, 10.0, 4096).astype(np.float32)
    worst = float(np.max(np.abs(np.array(block(exp_cancel_fn(jnp.array(a)))))))
    check(f"composition exp(a)*exp(-a) - 1: max |err| {worst:.3e} < 1e-10 (f32: ~1e-7)",
          worst < 1e-10)
    worst = float(np.max(np.abs(np.array(block(log_exp_fn(jnp.array(a)))))))
    check(f"composition log(exp(a)) - a: max |err| {worst:.3e} < 1e-10 (f32: ~1e-6)",
          worst < 1e-10)

    # ── 3d. Special cases through the plugin ─────────────────────────────────
    sp = np.array([np.nan, np.inf, -np.inf, 89.0, -104.0, 0.0], dtype=np.float32)
    got = np.array(block(exp_fn(jnp.array(sp))))
    check("exp specials: NaN, +Inf, 0, +Inf (overflow), 0 (underflow), 1",
          math.isnan(got[0]) and got[1] == np.inf and got[2] == 0.0
          and got[3] == np.inf and got[4] == 0.0 and got[5] == 1.0,
          f"got {got}")

    sp = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 1.0], dtype=np.float32)
    got = np.array(block(log_fn(jnp.array(sp))))
    check("log specials: NaN, +Inf, NaN, NaN (negative), -Inf (zero), 0",
          math.isnan(got[0]) and got[1] == np.inf and math.isnan(got[2])
          and math.isnan(got[3]) and got[4] == -np.inf and got[5] == 0.0,
          f"got {got}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  DS Exp / Log — Accuracy Tests")
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
        print("All DS exp/log tests passed.")

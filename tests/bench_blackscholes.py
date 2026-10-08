"""
Black-Scholes benchmark: native f64 vs DS-f32 vs native f32.

Prices a batch of European options with the PARSEC Black-Scholes kernel
(CNDF + BlkSchlsEqEuroNoDiv), written in JAX with the same statement order
as blackscholes/blackscholes_torch.py. The kernel uses add, subtract,
multiply, divide, sqrt, exp, log, negate, compare and select -- every float
op in it is DS-transformed by the pass.

Three runs per batch size, each in its own subprocess:

  f64  — float64 inputs, DS_BYPASS=1   (native f64 baseline)
  DS   — float64 inputs, plugin active (DS-f32 arithmetic, f64 at the boundary)
  f32  — float32 inputs, DS_BYPASS=1   (native f32, for the accuracy comparison)

Reported per run: median wall time, and the error of the returned prices
against a NumPy f64 evaluation of the same kernel.

Usage (inside container, after ds_setup.sh):
  JAX_ENABLE_X64=1 \\
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/bench_blackscholes.py
"""

import sys
import os
import subprocess
import json
from pathlib import Path

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PLUGIN_SO    = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"

WARMUP_REPS = 5
BENCH_REPS  = 30

BATCH_SIZES = [1_024, 65_536, 1_048_576, 16_777_216]


# ── Subprocess benchmark payload ──────────────────────────────────────────────
# Injected via -c; receives (batch, warmup, reps, dtype) as argv[1..4].

_BENCH_CODE = """
import sys, os, time, json
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

n      = int(sys.argv[1])
warmup = int(sys.argv[2])
reps   = int(sys.argv[3])
dtype  = sys.argv[4]

INV_SQRT_2PI = 0.39894228040143270286


def cndf(xp, x):
    sign = x < 0.0
    x = xp.where(sign, -x, x)
    n_prime = xp.exp(-0.5 * x * x) * INV_SQRT_2PI
    k = 1.0 / (1.0 + 0.2316419 * x)
    k2 = k * k
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k
    local_1 = k * 0.319381530
    local_2 = k2 * -0.356563782
    local_2 = local_2 + k3 * 1.781477937
    local_2 = local_2 + k4 * -1.821255978
    local_2 = local_2 + k5 * 1.330274429
    out = 1.0 - (local_2 + local_1) * n_prime
    return xp.where(sign, 1.0 - out, out)


def black_scholes(xp, spot, strike, rate, vol, time_, otype):
    sqrt_time = xp.sqrt(time_)
    log_term = xp.log(spot / strike)
    d1 = (rate + vol * vol * 0.5) * time_ + log_term
    den = vol * sqrt_time
    d1 = d1 / den
    d2 = d1 - den
    n_d1 = cndf(xp, d1)
    n_d2 = cndf(xp, d2)
    future_value = strike * xp.exp(-rate * time_)
    call = spot * n_d1 - future_value * n_d2
    put = future_value * (1.0 - n_d2) - spot * (1.0 - n_d1)
    return xp.where(otype == 0, call, put)


rng = np.random.default_rng(42)
np_dtype = np.float64 if dtype == "float64" else np.float32
inputs = [rng.uniform(lo, hi, n).astype(np_dtype) for lo, hi in
          [(10.0, 200.0), (10.0, 200.0), (0.01, 0.10), (0.05, 0.65), (0.05, 2.0)]]
otype = rng.integers(0, 2, n).astype(np.int32)

# Truth: the same kernel in NumPy f64, on the values the run actually sees.
truth = black_scholes(np, *[a.astype(np.float64) for a in inputs], otype)

fn = jax.jit(lambda *a: black_scholes(jnp, *a))
args = [jnp.array(a) for a in inputs] + [jnp.array(otype)]

for _ in range(warmup):
    fn(*args).block_until_ready()

times = []
for _ in range(reps):
    t0 = time.perf_counter()
    fn(*args).block_until_ready()
    times.append(time.perf_counter() - t0)

got = np.array(fn(*args)).astype(np.float64)
abs_err = np.abs(got - truth)
priced = truth > 1e-3   # relative error is meaningless for ~zero prices
print(json.dumps({
    "n":           n,
    "median":      float(np.median(times)),
    "max_abs_err": float(abs_err.max()),
    "max_rel_err": float((abs_err[priced] / truth[priced]).max()),
    "nan_count":   int(np.isnan(got).sum()),
}))
"""


def run_bench(n: int, dtype: str, bypass: bool) -> dict:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    if bypass:
        env["DS_BYPASS"] = "1"
    else:
        env.pop("DS_BYPASS", None)

    result = subprocess.run(
        [sys.executable, "-c", _BENCH_CODE,
         str(n), str(WARMUP_REPS), str(BENCH_REPS), dtype],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [subprocess stderr] {result.stderr[-600:]}", file=sys.stderr)
        return {}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"  [bad output] {result.stdout[:200]}", file=sys.stderr)
        return {}


def fmt_ms(d: dict) -> str:
    return f"{d['median'] * 1000:>10.3f}" if d else f"{'ERR':>10}"


def fmt_err(d: dict) -> str:
    if not d:
        return f"{'ERR':>11}"
    if d["nan_count"]:
        return f"{str(d['nan_count']) + ' NaN':>11}"
    return f"{d['max_rel_err']:>11.2e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pjrt_env = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
    if not pjrt_env or not PLUGIN_SO.exists():
        print("Black-Scholes benchmark needs the PJRT plugin "
              "(set PJRT_NAMES_AND_LIBRARY_PATHS) — skipped.")
        sys.exit(0)

    print("=" * 96)
    print("  Black-Scholes (PARSEC kernel) — native f64 vs DS-f32 vs native f32")
    print(f"  warmup={WARMUP_REPS}, reps={BENCH_REPS}, median reported")
    print("=" * 96)
    print(f"\n  {'Batch':>12}  {'f64 (ms)':>10}  {'DS (ms)':>10}  {'f32 (ms)':>10}  "
          f"{'f64/DS':>8}  {'f64 relerr':>11}  {'DS relerr':>11}  {'f32 relerr':>11}")
    print("  " + "-" * 94)

    for n in BATCH_SIZES:
        f64 = run_bench(n, "float64", bypass=True)
        ds  = run_bench(n, "float64", bypass=False)
        f32 = run_bench(n, "float32", bypass=True)
        speedup = (f"{f64['median'] / ds['median']:>7.2f}×"
                   if f64 and ds and ds["median"] > 0 else f"{'ERR':>8}")
        print(f"  {n:>12,}  {fmt_ms(f64)}  {fmt_ms(ds)}  {fmt_ms(f32)}  "
              f"{speedup}  {fmt_err(f64)}  {fmt_err(ds)}  {fmt_err(f32)}")

    print()
    print("Notes:")
    print("  f64/DS > 1 means DS-f32 is faster than native f64.")
    print("  relerr = max relative error of the returned prices against a NumPy")
    print("    f64 evaluation of the same kernel, over prices above 1e-3.")
    print("  A NaN count in place of an error means that run produced NaNs.")

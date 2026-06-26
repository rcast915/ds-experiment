"""
Dot product performance benchmark for the DS compiler pass.

Measures wall time and effective GFLOPS for jnp.dot(a*a, b) across a range
of vector sizes, comparing DS (plugin active) against f32 baseline
(DS_BYPASS=1).  The baseline is obtained via a subprocess so both measurements
come from the same script invocation.

Reported metrics per size:
  f32_ms    — median wall time with DS_BYPASS=1 (plain float32)
  ds_ms     — median wall time with DS transform active
  overhead  — ds_ms / f32_ms (1.0 = no overhead)
  f32_GFLOPS — effective throughput at f32 FLOPs count (2N)
  ds_GFLOPS  — same denominator so overhead is directly visible

Usage (inside container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/bench_dot_product.py

  python3 tests/bench_dot_product.py  # CPU-only mode (uses JAX cpu, no plugin needed)
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path

import numpy as np

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PLUGIN_SO    = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"

WARMUP_REPS = 10
BENCH_REPS  = 50

# Vector sizes to benchmark.
SIZES = [1_000, 10_000, 100_000, 1_000_000]


# ── Inner benchmark function (runs inside a clean subprocess) ─────────────────

_BENCH_CODE = """
import sys, os, time, json
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

n       = int(sys.argv[1])
warmup  = int(sys.argv[2])
reps    = int(sys.argv[3])

a = jnp.full(n, np.float32(0.1))
b = jnp.ones(n, dtype=jnp.float32)

@jax.jit
def fn(a, b):
    return jnp.dot(a * a, b)

# Warm up
for _ in range(warmup):
    fn(a, b).block_until_ready()

# Measure
times = []
for _ in range(reps):
    t0 = time.perf_counter()
    fn(a, b).block_until_ready()
    times.append(time.perf_counter() - t0)

times = sorted(times)
print(json.dumps({
    "n":      n,
    "median": float(np.median(times)),
    "mean":   float(np.mean(times)),
    "min":    float(np.min(times)),
    "p25":    float(times[len(times)//4]),
    "p75":    float(times[3*len(times)//4]),
}))
"""


def run_bench_subprocess(n: int, bypass: bool) -> dict:
    """Run the inner benchmark in a subprocess, returning timing dict."""
    env = dict(os.environ)
    if bypass:
        env["DS_BYPASS"] = "1"
    else:
        env.pop("DS_BYPASS", None)

    result = subprocess.run(
        [sys.executable, "-c", _BENCH_CODE, str(n), str(WARMUP_REPS), str(BENCH_REPS)],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [subprocess stderr] {result.stderr[:400]}", file=sys.stderr)
        return {}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"  [bad output] {result.stdout[:200]}", file=sys.stderr)
        return {}


def gflops(n: int, time_s: float) -> float:
    """Effective GFLOPS using standard dot product FLOPs = 2N."""
    return (2.0 * n) / time_s / 1e9


def print_header():
    print(f"\n{'Size':>10}  {'f32 (ms)':>10}  {'DS (ms)':>10}  {'overhead':>10}  "
          f"{'f32 GFLOPS':>12}  {'DS GFLOPS':>12}")
    print("-" * 72)


def print_row(n: int, f32: dict, ds: dict):
    if not f32 or not ds:
        print(f"{n:>10}  {'ERR':>10}  {'ERR':>10}  {'ERR':>10}")
        return
    f32_ms  = f32["median"] * 1000
    ds_ms   = ds["median"]  * 1000
    ratio   = ds_ms / f32_ms if f32_ms > 0 else float("inf")
    f32_g   = gflops(n, f32["median"])
    ds_g    = gflops(n, ds["median"])
    print(f"{n:>10,}  {f32_ms:>10.3f}  {ds_ms:>10.3f}  {ratio:>10.2f}×  "
          f"{f32_g:>12.2f}  {ds_g:>12.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pjrt_env = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
    gpu_mode = bool(pjrt_env) and PLUGIN_SO.exists()

    print("=" * 72)
    print("  Dot Product — Performance Benchmark")
    print(f"  fn: jnp.dot(a * a, b)  [mul creates DS lo channels]")
    print(f"  warmup={WARMUP_REPS}, reps={BENCH_REPS}, median reported")
    print(f"  mode: {'GPU (PJRT plugin)' if gpu_mode else 'CPU (no plugin)'}")
    print("=" * 72)

    print_header()

    all_results = []

    for n in SIZES:
        f32_data = run_bench_subprocess(n, bypass=True)
        ds_data  = run_bench_subprocess(n, bypass=False)
        print_row(n, f32_data, ds_data)
        if f32_data and ds_data:
            all_results.append((n, f32_data, ds_data))

    # Summary
    if all_results:
        overheads = [
            ds["median"] / f32["median"]
            for _, f32, ds in all_results
            if f32.get("median") and ds.get("median")
        ]
        if overheads:
            print()
            print(f"  Overhead summary: min {min(overheads):.2f}× | "
                  f"mean {sum(overheads)/len(overheads):.2f}× | "
                  f"max {max(overheads):.2f}×")
            print(f"  (overhead > 1 means DS is slower than f32)")

    print()
    print("Note: FLOPs denominator is 2N (standard dot). DS internally executes")
    print("      ~20× more f32 ops for the Veltkamp split + two_prod/two_sum;")
    print("      the overhead column shows the true wall-time cost of DS.")

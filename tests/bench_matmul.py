"""
Matrix multiply performance benchmark for the DS compiler pass.

Benchmarks (A * A) @ B across a range of square matrix sizes, comparing DS
against f32 baseline.  Uses A * A so the element-wise multiply creates
non-zero lo channels, giving the DS matmul decomposition real work to do.

Also benchmarks the standalone A @ B case to measure the overhead of the
4-matmul DS decomposition even when lo=0 (the worst-case overhead scenario).

Reported metrics per size:
  f32_ms     — median wall time, DS_BYPASS=1
  ds_ms      — median wall time, DS transform active
  overhead   — ds_ms / f32_ms
  f32_TFLOPS — throughput at standard matmul FLOPs = 2·M·K·N
  ds_TFLOPS  — same denominator

Usage (inside container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/bench_matmul.py
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

WARMUP_REPS = 5
BENCH_REPS  = 30

# Square matrix sizes to benchmark.
SIZES = [64, 256, 512, 1024, 2048]


# ── Inner benchmark code (injected into subprocess) ───────────────────────────

_BENCH_SQ_CODE = """
import sys, os, time, json
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

n      = int(sys.argv[1])
warmup = int(sys.argv[2])
reps   = int(sys.argv[3])
mode   = sys.argv[4]   # "sq" or "plain"

rng = np.random.default_rng(42)
A = jnp.array(rng.standard_normal((n, n)).astype(np.float32))
B = jnp.array(rng.standard_normal((n, n)).astype(np.float32))

if mode == "sq":
    @jax.jit
    def fn(A, B):
        return (A * A) @ B
else:
    @jax.jit
    def fn(A, B):
        return A @ B

for _ in range(warmup):
    fn(A, B).block_until_ready()

times = []
for _ in range(reps):
    t0 = time.perf_counter()
    fn(A, B).block_until_ready()
    times.append(time.perf_counter() - t0)

times = sorted(times)
print(json.dumps({
    "n":      n,
    "mode":   mode,
    "median": float(np.median(times)),
    "mean":   float(np.mean(times)),
    "min":    float(np.min(times)),
    "p25":    float(times[len(times)//4]),
    "p75":    float(times[3*len(times)//4]),
}))
"""


def run_bench(n: int, mode: str, bypass: bool) -> dict:
    env = dict(os.environ)
    if bypass:
        env["DS_BYPASS"] = "1"
    else:
        env.pop("DS_BYPASS", None)

    result = subprocess.run(
        [sys.executable, "-c", _BENCH_SQ_CODE,
         str(n), str(WARMUP_REPS), str(BENCH_REPS), mode],
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


def tflops(n: int, time_s: float) -> float:
    """TFLOPS using standard matmul FLOPs = 2·N³ for square N×N."""
    return (2.0 * n ** 3) / time_s / 1e12


def print_header(fn_label: str):
    print(f"\n  fn: {fn_label}")
    print(f"  {'Size':>8}  {'f32 (ms)':>10}  {'DS (ms)':>10}  "
          f"{'overhead':>10}  {'f32 TFLOPS':>12}  {'DS TFLOPS':>12}")
    print("  " + "-" * 68)


def print_row(n: int, f32: dict, ds: dict):
    if not f32 or not ds:
        print(f"  {n:>8,}  {'ERR':>10}  {'ERR':>10}  {'ERR':>10}")
        return
    f32_ms = f32["median"] * 1000
    ds_ms  = ds["median"]  * 1000
    ratio  = ds_ms / f32_ms if f32_ms > 0 else float("inf")
    f32_t  = tflops(n, f32["median"])
    ds_t   = tflops(n, ds["median"])
    print(f"  {n:>8,}  {f32_ms:>10.2f}  {ds_ms:>10.2f}  {ratio:>10.2f}×  "
          f"{f32_t:>12.3f}  {ds_t:>12.3f}")


def run_suite(mode: str, label: str) -> list:
    print_header(label)
    results = []
    for n in SIZES:
        f32_data = run_bench(n, mode, bypass=True)
        ds_data  = run_bench(n, mode, bypass=False)
        print_row(n, f32_data, ds_data)
        if f32_data and ds_data:
            results.append((n, f32_data, ds_data))
    return results


def summarize(results: list, label: str):
    if not results:
        return
    overheads = [
        ds["median"] / f32["median"]
        for _, f32, ds in results
        if f32.get("median") and ds.get("median")
    ]
    if overheads:
        print(f"\n  [{label}] overhead: min {min(overheads):.2f}× | "
              f"mean {sum(overheads)/len(overheads):.2f}× | "
              f"max {max(overheads):.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pjrt_env = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
    gpu_mode = bool(pjrt_env) and PLUGIN_SO.exists()

    print("=" * 72)
    print("  Matrix Multiply — Performance Benchmark")
    print(f"  warmup={WARMUP_REPS}, reps={BENCH_REPS}, median reported")
    print(f"  mode: {'GPU (PJRT plugin)' if gpu_mode else 'CPU (no plugin)'}")
    print("=" * 72)

    # Benchmark 1: (A*A) @ B — DS has real lo channels to work with.
    res_sq = run_suite("sq", "(A * A) @ B  [mul creates DS lo channels]")
    summarize(res_sq, "(A*A)@B")

    # Benchmark 2: A @ B — lo=0 for all inputs; measures raw overhead of the
    # 4-matmul decomposition with no precision payoff.
    res_plain = run_suite("plain", "A @ B  [standalone, lo=0, DS overhead only]")
    summarize(res_plain, "A@B")

    print()
    print("Notes:")
    print("  TFLOPS denominator = 2·N³ (standard matmul). DS runs 4 sub-matmuls")
    print("  internally, so true DS FLOPs ≈ 4× the standard count.")
    print("  Overhead > 1 in the (A*A)@B case reflects the precision/cost tradeoff.")
    print("  Overhead in the A@B case is the worst-case raw decomposition cost.")

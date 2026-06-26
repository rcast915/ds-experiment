"""
Float64 vs DS-f32 performance benchmark — the primary research result.

On H100, cuBLAS f64 GEMM is ~30× slower than f32 (no tensor-core support).
DS-f32 replaces each f64 op with DS arithmetic over two f32 values, keeping
~48-bit precision while running at f32 hardware speed.  For matrix multiply
this means 4 f32 GEMMs (~4× f32 cost) vs one f64 GEMM (~30× f32 cost): an
expected 7–8× speedup over native f64 at near-f64 precision.

Benchmarks three operation classes, each with a f64 baseline (DS_BYPASS=1)
and a DS-f32 run (plugin active), both with JAX_ENABLE_X64=1:

  1. Element-wise arithmetic  — A * A + B
  2. Large reduction          — jnp.sum(A)
  3. Matrix multiply          — A @ B  (main story: largest speedup expected)

Reported metrics:
  f64_ms   — median wall time, native f64  (DS_BYPASS=1)
  ds_ms    — median wall time, DS-f32      (plugin active)
  speedup  — f64_ms / ds_ms  (> 1 means DS is faster)

Usage (inside container, after ds_setup.sh):
  JAX_ENABLE_X64=1 \\
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/bench_f64_vs_ds.py
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

VECTOR_SIZES = [1_000, 100_000, 1_000_000]
MATRIX_SIZES = [256, 512, 1024, 2048]


# ── Subprocess benchmark payload ──────────────────────────────────────────────
# Injected via -c; receives (size, warmup, reps, mode) as argv[1..4].
# mode: "elemwise" | "reduce" | "matmul"

_BENCH_CODE = """
import sys, os, time, json
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

n      = int(sys.argv[1])
warmup = int(sys.argv[2])
reps   = int(sys.argv[3])
mode   = sys.argv[4]

rng = np.random.default_rng(42)

if mode == "matmul":
    A = jnp.array(rng.standard_normal((n, n)), dtype=jnp.float64)
    B = jnp.array(rng.standard_normal((n, n)), dtype=jnp.float64)
    @jax.jit
    def fn(A, B): return A @ B
    args = (A, B)
elif mode == "reduce":
    A = jnp.array(rng.standard_normal((n,)), dtype=jnp.float64)
    @jax.jit
    def fn(A): return jnp.sum(A)
    args = (A,)
else:  # elemwise
    A = jnp.array(rng.standard_normal((n,)), dtype=jnp.float64)
    B = jnp.array(rng.standard_normal((n,)), dtype=jnp.float64)
    @jax.jit
    def fn(A, B): return A * A + B
    args = (A, B)

for _ in range(warmup):
    fn(*args).block_until_ready()

times = []
for _ in range(reps):
    t0 = time.perf_counter()
    fn(*args).block_until_ready()
    times.append(time.perf_counter() - t0)

times.sort()
print(json.dumps({
    "n":      n,
    "mode":   mode,
    "median": float(np.median(times)),
    "min":    float(times[0]),
    "p25":    float(times[len(times) // 4]),
    "p75":    float(times[3 * len(times) // 4]),
}))
"""


def run_bench(n: int, mode: str, bypass: bool) -> dict:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    if bypass:
        env["DS_BYPASS"] = "1"
    else:
        env.pop("DS_BYPASS", None)

    result = subprocess.run(
        [sys.executable, "-c", _BENCH_CODE,
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


def gflops_elemwise(n: int, t: float) -> float:
    return (3.0 * n) / t / 1e9   # mul + mul + add


def gflops_reduce(n: int, t: float) -> float:
    return n / t / 1e9


def tflops_matmul(n: int, t: float) -> float:
    return (2.0 * n ** 3) / t / 1e12


def print_header_vec(label: str):
    print(f"\n  fn: {label}")
    print(f"  {'Size':>12}  {'f64 (ms)':>10}  {'DS (ms)':>10}  "
          f"{'speedup':>9}  {'f64 GFLOPS':>12}  {'DS GFLOPS':>12}")
    print("  " + "-" * 72)


def print_row_vec(n: int, f64: dict, ds: dict, flops_fn):
    if not f64 or not ds:
        print(f"  {n:>12,}  {'ERR':>10}  {'ERR':>10}  {'ERR':>9}")
        return
    f64_ms  = f64["median"] * 1000
    ds_ms   = ds["median"]  * 1000
    speedup = f64_ms / ds_ms if ds_ms > 0 else float("inf")
    f64_g   = flops_fn(n, f64["median"])
    ds_g    = flops_fn(n, ds["median"])
    print(f"  {n:>12,}  {f64_ms:>10.3f}  {ds_ms:>10.3f}  "
          f"{speedup:>8.2f}×  {f64_g:>12.3f}  {ds_g:>12.3f}")


def print_header_mat(label: str):
    print(f"\n  fn: {label}")
    print(f"  {'Size':>8}  {'f64 (ms)':>10}  {'DS (ms)':>10}  "
          f"{'speedup':>9}  {'f64 TFLOPS':>12}  {'DS TFLOPS':>12}")
    print("  " + "-" * 68)


def print_row_mat(n: int, f64: dict, ds: dict):
    if not f64 or not ds:
        print(f"  {n:>8,}  {'ERR':>10}  {'ERR':>10}  {'ERR':>9}")
        return
    f64_ms  = f64["median"] * 1000
    ds_ms   = ds["median"]  * 1000
    speedup = f64_ms / ds_ms if ds_ms > 0 else float("inf")
    f64_t   = tflops_matmul(n, f64["median"])
    ds_t    = tflops_matmul(n, ds["median"])
    print(f"  {n:>8,}  {f64_ms:>10.2f}  {ds_ms:>10.2f}  "
          f"{speedup:>8.2f}×  {f64_t:>12.3f}  {ds_t:>12.3f}")


def run_vec_suite(mode: str, label: str, sizes: list, flops_fn) -> list:
    print_header_vec(label)
    results = []
    for n in sizes:
        f64_data = run_bench(n, mode, bypass=True)
        ds_data  = run_bench(n, mode, bypass=False)
        print_row_vec(n, f64_data, ds_data, flops_fn)
        if f64_data and ds_data:
            results.append((n, f64_data, ds_data))
    return results


def run_mat_suite(sizes: list) -> list:
    print_header_mat("A @ B  (f64 vs DS-f32 — main speedup result)")
    results = []
    for n in sizes:
        f64_data = run_bench(n, "matmul", bypass=True)
        ds_data  = run_bench(n, "matmul", bypass=False)
        print_row_mat(n, f64_data, ds_data)
        if f64_data and ds_data:
            results.append((n, f64_data, ds_data))
    return results


def summarize(results: list, label: str):
    if not results:
        return
    speedups = [
        f64["median"] / ds["median"]
        for _, f64, ds in results
        if f64.get("median") and ds.get("median")
    ]
    if speedups:
        print(f"\n  [{label}] speedup: min {min(speedups):.2f}× | "
              f"mean {sum(speedups)/len(speedups):.2f}× | "
              f"max {max(speedups):.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pjrt_env = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
    gpu_mode = bool(pjrt_env) and PLUGIN_SO.exists()

    print("=" * 72)
    print("  Float64 vs DS-f32 — Performance Benchmark")
    print("  Baseline: native f64 (DS_BYPASS=1)  |  DS: plugin active")
    print(f"  warmup={WARMUP_REPS}, reps={BENCH_REPS}, median reported")
    print(f"  mode: {'GPU (PJRT plugin)' if gpu_mode else 'CPU (no plugin)'}")
    print("=" * 72)

    res_ew  = run_vec_suite("elemwise", "A * A + B  (element-wise)",
                            VECTOR_SIZES, gflops_elemwise)
    summarize(res_ew, "element-wise")

    res_red = run_vec_suite("reduce",   "jnp.sum(A)  (reduction)",
                            VECTOR_SIZES, gflops_reduce)
    summarize(res_red, "reduction")

    res_mat = run_mat_suite(MATRIX_SIZES)
    summarize(res_mat, "matmul")

    print()
    print("Notes:")
    print("  speedup > 1 means DS-f32 is faster than native f64.")
    print("  Matmul: H100 f64 GEMM has no tensor-core support (~30× slower")
    print("    than f32). DS uses 4 f32 GEMMs, so expected speedup ≈ 7–8×.")
    print("  TFLOPS denominator = 2·N³ (standard matmul). DS runs 4 sub-")
    print("    matmuls, so true DS FLOPs ≈ 4× the standard count.")

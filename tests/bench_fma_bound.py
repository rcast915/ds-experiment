"""
How much could a real FMA in XLA speed up DS?  An upper bound, measured.

The pass computes every error-free product (two_prod) with a Veltkamp split:
17 f32 ops where an FMA-based version needs 2 (p = a*b, e = fma(a, b, -p)).
StableHLO has no FMA, so that version can't be built today. Instead, the pass's
DS_FAKE_FMA_TIMING=1 mode swaps the error term for a one-op stand-in of the
same cost. Its RESULTS ARE WRONG; only its run time means anything: it is what
DS would cost if XLA had an FMA.

Runs the Black-Scholes kernel from bench_blackscholes.py four ways, each in its
own subprocess:

  f64       f64 inputs, DS_BYPASS=1             (native f64)
  DS        f64 inputs, plugin, Veltkamp        (today's DS)
  DS+FMA*   f64 inputs, plugin, FMA-cost stand-in  (timing only)
  f32       f32 inputs, DS_BYPASS=1             (native f32)

and first reports how many StableHLO ops the pass emits for the kernel in each
DS mode.

Reading the result: if f64/DS+FMA* stays below 1 on a GPU, adding an FMA to
XLA cannot make DS beat native f64 on this kernel there, and the XLA work is
not worth doing for speed.

Usage (inside container, after ds_setup.sh and a rebuild of mlir-ds-opt):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/bench_fma_bound.py
"""

import sys
import os
import re
import subprocess
import json
import tempfile
from pathlib import Path

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PLUGIN_SO    = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"
OPT_BINARY   = PROJECT_ROOT / "stablehlo_pass" / "build" / "mlir-ds-opt"

sys.path.insert(0, str(TESTS_DIR))
from bench_blackscholes import _BENCH_CODE  # noqa: E402  (the Black-Scholes payload)

WARMUP_REPS = 5
BENCH_REPS  = 50
BATCH_SIZES = [1_048_576, 16_777_216, 67_108_864]

# The kernel alone (the payload's definitions, without its benchmark loop),
# lowered on CPU for the op count.
_KERNEL_SRC = _BENCH_CODE[_BENCH_CODE.index("INV_SQRT_2PI"):_BENCH_CODE.index("rng = np.random")]
_LOWER_CODE = (
    "import os\nos.environ['JAX_ENABLE_X64'] = '1'\n"
    "import numpy as np\nimport jax\nimport jax.numpy as jnp\n"
    + _KERNEL_SRC +
    "\nf = lambda *a: black_scholes(jnp, *a)\n"
    "s = jax.ShapeDtypeStruct((1024,), jnp.float64)\n"
    "i = jax.ShapeDtypeStruct((1024,), jnp.int32)\n"
    "print(jax.jit(f).lower(s, s, s, s, s, i).compiler_ir())\n"
)


def count_ops() -> dict:
    """StableHLO op count of the kernel: as written, and after each DS mode."""
    env = {k: v for k, v in os.environ.items() if k != "PJRT_NAMES_AND_LIBRARY_PATHS"}
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run([sys.executable, "-c", _LOWER_CODE], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr[-400:]}

    def ops(text):
        return len(re.findall(r"=\s*\"?stablehlo\.", text))

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(r.stdout)
    out = {"native": ops(r.stdout)}
    for label, fake in (("DS", "0"), ("DS+FMA*", "1")):
        e = dict(env, DS_FAKE_FMA_TIMING=fake)
        p = subprocess.run(
            [str(OPT_BINARY),
             "--pass-pipeline=builtin.module(inline,func.func(ds-transform))", f.name],
            env=e, capture_output=True, text=True)
        out[label] = ops(p.stdout) if p.returncode == 0 else "ERR"
    os.unlink(f.name)
    return out


def run_bench(n: int, dtype: str, bypass: bool, fake_fma: bool) -> dict:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    if bypass:
        env["DS_BYPASS"] = "1"
    else:
        env.pop("DS_BYPASS", None)
    env["DS_FAKE_FMA_TIMING"] = "1" if fake_fma else "0"
    r = subprocess.run(
        [sys.executable, "-c", _BENCH_CODE, str(n), str(WARMUP_REPS), str(BENCH_REPS), dtype],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [subprocess stderr] {r.stderr[-600:]}", file=sys.stderr)
        return {}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}


def ms(d: dict) -> str:
    return f"{d['median'] * 1000:>10.3f}" if d else f"{'ERR':>10}"


def ratio(num: dict, den: dict) -> str:
    if num and den and den["median"] > 0:
        return f"{num['median'] / den['median']:>8.2f}x"
    return f"{'-':>9}"


if __name__ == "__main__":
    if not os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS") or not PLUGIN_SO.exists():
        print("Needs the PJRT plugin (set PJRT_NAMES_AND_LIBRARY_PATHS).")
        sys.exit(1)

    print("=" * 92)
    print("  FMA upper bound — Black-Scholes, native f64 vs DS vs DS with FMA-cost products")
    print(f"  warmup={WARMUP_REPS}, reps={BENCH_REPS}, median reported")
    print("  DS+FMA* = DS_FAKE_FMA_TIMING=1: timing only, its results are deliberately wrong")
    print("=" * 92)

    c = count_ops()
    print("\n  StableHLO ops in the kernel:")
    if "error" in c:
        print(f"    (op count failed: {c['error']})")
    else:
        print(f"    native {c['native']}   ->   DS {c['DS']}   ->   DS+FMA* {c['DS+FMA*']}")

    print(f"\n  {'Batch':>12}  {'f64 ms':>10}  {'DS ms':>10}  {'DS+FMA* ms':>10}  {'f32 ms':>10}"
          f"  {'f64/DS':>9}  {'f64/DS+FMA*':>11}  {'DS/DS+FMA*':>10}")
    print("  " + "-" * 90)
    for n in BATCH_SIZES:
        f64  = run_bench(n, "float64", bypass=True,  fake_fma=False)
        ds   = run_bench(n, "float64", bypass=False, fake_fma=False)
        fma  = run_bench(n, "float64", bypass=False, fake_fma=True)
        f32  = run_bench(n, "float32", bypass=True,  fake_fma=False)
        print(f"  {n:>12,}  {ms(f64)}  {ms(ds)}  {ms(fma)}  {ms(f32)}"
              f"  {ratio(f64, ds)}  {ratio(f64, fma):>11}  {ratio(ds, fma):>10}")

    print()
    print("Notes:")
    print("  f64/X > 1 means X is faster than native f64.")
    print("  DS/DS+FMA* is the speedup a real FMA could give DS at most.")
    print("  If f64/DS+FMA* is below 1, an FMA in XLA cannot make DS beat native f64 here.")

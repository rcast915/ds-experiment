"""
Diagnostic: how accurate is the DS matmul, and what does it cost?

The pass turns one matmul into four f32 GEMMs (hi*hi, hi*lo, lo*hi, lo*lo).
Each GEMM rounds every product and every partial sum to f32 and none of
that rounding is captured, so the result may be no better than plain f32
even though the inputs are carried exactly. This script measures it.

For A @ B on random n x n matrices, five configurations, each in its own
subprocess:

  f64          f64 inputs, DS_BYPASS=1                     (native f64)
  DS           f64 inputs, plugin active, default precision
  DS highest   f64 inputs, plugin active, precision=HIGHEST
  f32          f32 inputs, DS_BYPASS=1, default precision  (native f32)
  f32 highest  f32 inputs, DS_BYPASS=1, precision=HIGHEST

Reported per configuration and size:

  ms       median wall time
  relerr   ||C - T||_F / ||T||_F, with T = NumPy f64 A @ B on the same
           input values the run saw

Reading the result: native f64 lands near 1e-16 and native f32 near 1e-7.
If DS is near 1e-7 too, its matmul is f32-accurate and its speed advantage
over f64 is not a like-for-like comparison.

Usage (inside container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/diag_f64_matmul.py
"""

import sys
import os
import subprocess
import json
from pathlib import Path

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PLUGIN_SO    = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"

SIZES  = [256, 1024, 2048]
WARMUP = 3
REPS   = 15

# Injected via -c; receives (dtype, precision, warmup, reps, sizes...) as argv.
_CODE = """
import sys, os, time, json
import numpy as np
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp

dtype, precision = sys.argv[1], sys.argv[2]
warmup, reps = int(sys.argv[3]), int(sys.argv[4])
sizes = [int(s) for s in sys.argv[5:]]
np_dtype = np.float64 if dtype == "float64" else np.float32
prec = None if precision == "default" else precision

fn = jax.jit(lambda a, b: jnp.matmul(a, b, precision=prec))

out = {}
for n in sizes:
    rng = np.random.default_rng(n)
    A = rng.standard_normal((n, n)).astype(np_dtype)
    B = rng.standard_normal((n, n)).astype(np_dtype)
    truth = A.astype(np.float64) @ B.astype(np.float64)
    a, b = jnp.array(A), jnp.array(B)
    try:
        for _ in range(warmup):
            fn(a, b).block_until_ready()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(a, b).block_until_ready()
            times.append(time.perf_counter() - t0)
        got = np.asarray(fn(a, b), dtype=np.float64)
        out[str(n)] = {
            "ms": float(np.median(times)) * 1000.0,
            "relerr": float(np.linalg.norm(got - truth) / np.linalg.norm(truth)),
        }
    except Exception as e:  # report and keep going
        out[str(n)] = {"error": type(e).__name__}
print("RESULT " + json.dumps(out))
"""

CONFIGS = [
    # label,         dtype,     precision, bypass
    ("f64",          "float64", "default", True),
    ("DS",           "float64", "default", False),
    ("DS highest",   "float64", "highest", False),
    ("f32",          "float32", "default", True),
    ("f32 highest",  "float32", "highest", True),
]


def run(dtype: str, precision: str, bypass: bool) -> dict:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    if bypass:
        env["DS_BYPASS"] = "1"
    else:
        env.pop("DS_BYPASS", None)
    r = subprocess.run(
        [sys.executable, "-c", _CODE, dtype, precision, str(WARMUP), str(REPS)]
        + [str(n) for n in SIZES],
        env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    print(f"  [subprocess failed]\n{r.stderr[-1500:]}", file=sys.stderr)
    return {}


if __name__ == "__main__":
    if not os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS") or not PLUGIN_SO.exists():
        print("Needs the PJRT plugin (set PJRT_NAMES_AND_LIBRARY_PATHS).")
        sys.exit(1)

    results = {label: run(dtype, precision, bypass)
               for label, dtype, precision, bypass in CONFIGS}

    print("\nA @ B, random normal n x n  (ms = median wall time; "
          "relerr = ||C - T||_F / ||T||_F vs NumPy f64)\n")
    header = f"  {'n':>6}"
    for label, *_ in CONFIGS:
        header += f"  | {label + ' ms':>14} {'relerr':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in SIZES:
        row = f"  {n:>6}"
        for label, *_ in CONFIGS:
            d = results[label].get(str(n))
            if not d:
                row += f"  | {'-':>14} {'-':>9}"
            elif "error" in d:
                row += f"  | {'ERR':>14} {d['error'][:9]:>9}"
            else:
                row += f"  | {d['ms']:>14.3f} {d['relerr']:>9.1e}"
        print(row)

    print("\n  f64/DS time ratio (> 1 means DS is faster):")
    for n in SIZES:
        f64 = results["f64"].get(str(n), {})
        parts = []
        for label in ("DS", "DS highest"):
            ds = results[label].get(str(n), {})
            if "ms" in f64 and "ms" in ds and ds["ms"] > 0:
                parts.append(f"{label}: {f64['ms'] / ds['ms']:.2f}x")
        print(f"    n={n:<5} " + "   ".join(parts))

    print("\n  Native f64 should be ~1e-16 and native f32 ~1e-7.")
    print("  DS near 1e-7 = its matmul is f32-accurate, not f64-accurate.")

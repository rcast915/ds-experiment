"""
Per-op accuracy diagnostic for f64 inputs through the DS plugin.

Each case is one small jitted function on f64 inputs, compared against the
same computation in NumPy f64. A DS-f32 result should agree to roughly
1e-12 or better; a case at ~1e-7 means some step in it ran at plain f32
precision, and anything in between points at a partially lost error term.

Every case runs twice, each in its own subprocess:

  default   — the normal plugin pipeline
  no-algsimp — XLA's algebraic simplifier disabled
               (XLA_FLAGS=--xla_disable_hlo_passes=algsimp)

A case that is accurate only with the simplifier disabled is being damaged
by an XLA rewrite of the emitted sequence, not by the sequence itself.

Usage (inside container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/diag_f64_ops.py
"""

import sys
import os
import subprocess
import json
from pathlib import Path

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PLUGIN_SO    = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"

_CODE = """
import sys, os, json
import numpy as np
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp

n = 4096
rng = np.random.default_rng(7)
x = rng.uniform(0.5, 3.0, n)          # generic positive operands
y = rng.uniform(0.5, 3.0, n)
spot, strike = rng.uniform(10.0, 200.0, n), rng.uniform(10.0, 200.0, n)
rate, vol, time_ = rng.uniform(0.01, 0.10, n), rng.uniform(0.05, 0.65, n), rng.uniform(0.05, 2.0, n)

INV_SQRT_2PI = 0.39894228040143270286

def cndf(xp, v):
    sign = v < 0.0
    v = xp.where(sign, -v, v)
    n_prime = xp.exp(-0.5 * v * v) * INV_SQRT_2PI
    k = 1.0 / (1.0 + 0.2316419 * v)
    k2 = k * k; k3 = k2 * k; k4 = k3 * k; k5 = k4 * k
    local_2 = k2 * -0.356563782 + k3 * 1.781477937
    local_2 = local_2 + k4 * -1.821255978
    local_2 = local_2 + k5 * 1.330274429
    out = 1.0 - (local_2 + k * 0.319381530) * n_prime
    return xp.where(sign, 1.0 - out, out)

def d1_fn(xp, s, k, r, v, t):
    return ((r + v * v * 0.5) * t + xp.log(s / k)) / (v * xp.sqrt(t))

bs = (spot, strike, rate, vol, time_)
CASES = [
    # name, function(xp, *args), args
    ("identity  x + 0.0",        lambda xp, a: a + 0.0,                (x,)),
    ("add       x + y",          lambda xp, a, b: a + b,               (x, y)),
    ("sub       x - y",          lambda xp, a, b: a - b,               (x, y)),
    ("mul       x * y",          lambda xp, a, b: a * b,               (x, y)),
    ("mul const x * 0.2316419",  lambda xp, a: a * 0.2316419,          (x,)),
    ("add const 1.0 + x",        lambda xp, a: 1.0 + a,                (x,)),
    ("sub const 1.0 - x",        lambda xp, a: 1.0 - a,                (x,)),
    ("mul chain -0.5 * x * x",   lambda xp, a: -0.5 * a * a,           (x,)),
    ("div       x / y",          lambda xp, a, b: a / b,               (x, y)),
    ("div const 1.0 / x",        lambda xp, a: 1.0 / a,                (x,)),
    ("sqrt      sqrt(x)",        lambda xp, a: xp.sqrt(a),             (x,)),
    ("exp       exp(x)",         lambda xp, a: xp.exp(a),              (x,)),
    ("exp       exp(-x)",        lambda xp, a: xp.exp(-a),             (x,)),
    ("log       log(x)",         lambda xp, a: xp.log(a),              (x,)),
    ("log       log(x / y)",     lambda xp, a, b: xp.log(a / b),       (x, y)),
    ("select    where(x<y,-x,x)", lambda xp, a, b: xp.where(a < b, -a, a), (x, y)),
    ("sum       sum(x)",         lambda xp, a: xp.sum(a),              (x,)),
    ("sum       sum(full 0.1)",  lambda xp, a: xp.sum(a),              (np.full(10000, 0.1),)),
    ("BS d1",                    d1_fn,                                bs),
    ("BS cndf(x - 1.5)",         lambda xp, a: cndf(xp, a - 1.5),      (x,)),
    ("BS cndf(d1)",              lambda xp, *a: cndf(xp, d1_fn(xp, *a)), bs),
    ("BS strike*exp(-r*t)",      lambda xp, k, r, t: k * xp.exp(-r * t), (strike, rate, time_)),
]

out = {}
for name, fn, args in CASES:
    truth = np.asarray(fn(np, *args), dtype=np.float64)
    try:
        got = np.asarray(jax.jit(lambda *a: fn(jnp, *a))(*[jnp.array(a) for a in args]),
                         dtype=np.float64)
        scale = np.maximum(np.abs(truth), 1e-300)
        out[name] = float(np.max(np.abs(got - truth) / scale))
    except Exception as e:  # report and keep going
        out[name] = "ERR " + type(e).__name__
print("RESULT " + json.dumps(out))
"""


def run(extra_env: dict) -> dict:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env.pop("DS_BYPASS", None)
    env.update(extra_env)
    r = subprocess.run([sys.executable, "-c", _CODE], env=env,
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    print(f"  [subprocess failed]\n{r.stderr[-1500:]}", file=sys.stderr)
    return {}


def fmt(v) -> str:
    if v is None:
        return f"{'-':>12}"
    if isinstance(v, str):
        return f"{v[:12]:>12}"
    return f"{v:>12.2e}"


if __name__ == "__main__":
    if not os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS") or not PLUGIN_SO.exists():
        print("Needs the PJRT plugin (set PJRT_NAMES_AND_LIBRARY_PATHS).")
        sys.exit(1)

    default = run({})
    no_simp = run({"XLA_FLAGS": (os.environ.get("XLA_FLAGS", "") +
                                 " --xla_disable_hlo_passes=algsimp").strip()})
    native  = run({"DS_BYPASS": "1"})

    print("\nMax relative error vs NumPy f64 (f64 inputs, n=4096)\n")
    print(f"  {'case':<28}  {'DS default':>12}  {'DS no-algsimp':>13}  {'native f64':>12}")
    print("  " + "-" * 72)
    for name in default or no_simp or native:
        print(f"  {name:<28}  {fmt(default.get(name))}  {fmt(no_simp.get(name)):>13}  "
              f"{fmt(native.get(name))}")
    print("\n  DS should be ~1e-12 or better. ~1e-7 = a step ran at f32 precision.")
    print("  Accurate only under no-algsimp = an XLA rewrite is breaking the sequence.")

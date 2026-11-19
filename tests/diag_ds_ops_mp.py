"""
Per-operation accuracy of the DS pass against a 50-digit reference.

For each of add, mul, divide, sqrt, exp and log: about a million random
inputs spread across the exponent range, plus edge cases, run on the GPU
through the plugin with DS_RETURN_PAIRS=1 (the function returns the raw
(hi, lo) pair, so no rounding to f64 hides the DS result). The error of
hi + lo is measured against Python's `decimal` module at 50 significant
digits (correctly rounded exp, ln, sqrt and division), not against f64,
which has only ~4 bits of margin over DS.

Inputs are built as exact DS values: x = f64(h) + f64(l) with h, l f32 and
|l| < ulp(h)/4, chosen so that x is also exactly an f64. The pass's split of
x then gives back exactly (h, l), so the measured error is the operation's
own error, not the input split's.

Reported per op and input set: maximum and 99.9th-percentile relative
error in units of 2^-49 (DS unit roundoff is ~2^-48 to 2^-49), the input at
the maximum, and the same for native f64 (DS_BYPASS=1) in units of 2^-53.

Usage (inside the container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/diag_ds_ops_mp.py [n_random]          # default 1,000,000 per op
"""

import os
import sys
import json
import subprocess
import tempfile
from decimal import Decimal, getcontext

import numpy as np

getcontext().prec = 50
U49 = 2.0 ** -49
U53 = 2.0 ** -53

# Runs one op on the GPU; argv: op, mode (ds|f64), in .npz, out .npz
_GPU_CODE = r"""
import sys, os
os.environ["JAX_ENABLE_X64"] = "1"
import numpy as np, jax, jax.numpy as jnp
op, mode, inp, out = sys.argv[1:5]
d = np.load(inp)
a = jnp.array(d["a"]); b = jnp.array(d["b"])
f = {"add": lambda a, b: a + b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
     "sqrt": lambda a, b: jnp.sqrt(a), "exp": lambda a, b: jnp.exp(a),
     "log": lambda a, b: jnp.log(a)}[op]
if mode == "ds":
    g = jax.jit(lambda a, b: (lambda r: (r, r))(f(a, b)))      # DS_RETURN_PAIRS -> (hi, lo)
    hi, lo = g(a, b)
    np.savez(out, hi=np.asarray(hi), lo=np.asarray(lo))
else:
    r = jax.jit(f)(a, b)
    np.savez(out, hi=np.asarray(r), lo=np.zeros_like(np.asarray(r)))
"""


def ds_values(rng, n, log2_lo, log2_hi, positive=False):
    """n f64 values that are exact DS pairs, |hi| log-uniform in [2^lo, 2^hi]."""
    e = rng.uniform(log2_lo, log2_hi, n)
    h = (2.0 ** e).astype(np.float32)
    if not positive:
        h = h * rng.choice(np.array([-1, 1], np.float32), n)
    h = h.astype(np.float64)
    ulp = np.spacing(np.abs(h.astype(np.float32))).astype(np.float64)
    # |l| in [ulp/2^5, ulp/4): keeps x exactly representable in f64 (<= 53 bits)
    l_mag = ulp * 2.0 ** rng.uniform(-5, -2, n)
    l = (l_mag * rng.choice([-1.0, 1.0], n)).astype(np.float32).astype(np.float64)
    x = h + l
    assert np.all(x - h == l), "input is not an exact DS pair"
    return x


def input_sets(rng, n):
    """op -> list of (set name, a, b)."""
    one = np.ones(1)
    def rnd(lo, hi, m=n, pos=False):
        return ds_values(rng, m, lo, hi, pos)
    s = {}
    k = max(n // 20, 1000)                       # size of each edge-case set
    s["add"] = [
        ("random, exponents in [-30, 30]", rnd(-30, 30), rnd(-30, 30)),
    ]
    a = rnd(-10, 10, k)
    s["add"].append(("cancellation b ~ -a (rel. diff 2^-10..2^-40)", a,
                     ds_values_like(-a * (1 + 2.0 ** rng.uniform(-40, -10, k)
                                          * rng.choice([-1, 1], k)))))
    s["mul"] = [
        ("random, exponents in [-60, 60]", rnd(-60, 60), rnd(-60, 60)),
        ("operand near Veltkamp limit, |a| in [2^110, 2^120]", rnd(110, 120, k), rnd(-8, 0, k)),
    ]
    s["div"] = [
        ("random, exponents in [-60, 60]", rnd(-60, 60), rnd(-60, 60)),
        ("divisor near Veltkamp limit, |b| in [2^110, 2^120]", rnd(110, 120, k), rnd(110, 120, k)),
        ("dividend large, |a| in [2^110, 2^120]", rnd(110, 120, k), rnd(-4, 4, k)),
    ]
    s["sqrt"] = [
        ("random, x in [2^-120, 2^120]", rnd(-120, 120, pos=True), one.repeat(n)),
        ("x near f32 max, [2^120, 2^127]", rnd(120, 127, k, True), one.repeat(k)),
        ("x near Veltkamp limit, [2^110, 2^120]", rnd(110, 120, k, True), one.repeat(k)),
    ]
    xe = rng.uniform(-87.0, 88.0, n)
    xe = ds_values_like(xe)
    s["exp"] = [
        ("random, x in [-87, 88]", xe, one.repeat(n)),
        ("large x, [80, 88.7]", ds_values_like(rng.uniform(80.0, 88.7, k)), one.repeat(k)),
        ("small results, x in [-87.3, -60]", ds_values_like(rng.uniform(-87.3, -60.0, k)), one.repeat(k)),
        ("tiny |x|, [2^-40, 2^-10]", rnd(-40, -10, k), one.repeat(k)),
    ]
    t = 2.0 ** rng.uniform(-40, -1, k) * rng.choice([-1.0, 1.0], k)
    s["log"] = [
        ("random, x in [2^-120, 2^120]", rnd(-120, 120, pos=True), one.repeat(n)),
        ("near 1, x = 1 + t, |t| in [2^-40, 2^-1]", ds_values_like(1.0 + t), one.repeat(k)),
        ("near f32 limits, [2^-126, 2^-120] and [2^120, 2^127]",
         np.concatenate([rnd(-126, -120, k // 2, True), rnd(120, 127, k // 2, True)]),
         one.repeat(2 * (k // 2))),
    ]
    return s


def ds_values_like(v):
    """Round arbitrary f64 values to the nearest exact DS pair (h + l)."""
    h = v.astype(np.float32).astype(np.float64)
    l = (v - h).astype(np.float32).astype(np.float64)
    return h + l


def reference(op, a, b):
    """op(a, b) at 50 digits, as Decimal, for each element."""
    out = []
    for x, y in zip(a.tolist(), b.tolist()):
        X = Decimal(x)
        if op == "add":   r = X + Decimal(y)
        elif op == "mul": r = X * Decimal(y)
        elif op == "div": r = X / Decimal(y)
        elif op == "sqrt": r = X.sqrt()
        elif op == "exp": r = X.exp()
        else:             r = X.ln()
        out.append(r)
    return out


def rel_errors(ref, hi, lo):
    err = np.empty(len(ref))
    for i, (r, h, l) in enumerate(zip(ref, hi.tolist(), lo.tolist())):
        got = Decimal(h) + Decimal(l)
        err[i] = float(abs(got - r) / abs(r)) if r != 0 else float(abs(got))
    return err


def run_gpu(op, mode, a, b, tmp):
    inp = os.path.join(tmp, f"{op}_in.npz")
    out = os.path.join(tmp, f"{op}_{mode}_out.npz")
    np.savez(inp, a=a, b=b)
    env = dict(os.environ, JAX_ENABLE_X64="1")
    env.pop("DS_BYPASS", None); env.pop("DS_RETURN_PAIRS", None)
    if mode == "ds":
        env["DS_RETURN_PAIRS"] = "1"
    else:
        env["DS_BYPASS"] = "1"
    r = subprocess.run([sys.executable, "-c", _GPU_CODE, op, mode, inp, out],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [{op} {mode} failed] {r.stderr[-600:]}")
        return None
    d = np.load(out)
    return d["hi"].astype(np.float64), d["lo"].astype(np.float64)


def summarize(err, unit, a, b, op):
    finite = np.isfinite(err)
    if not finite.any():
        return "all non-finite"
    e = err[finite] / unit
    i = int(np.argmax(np.where(finite, err, -1)))
    arg = f"a={a[i]:.17g}" + (f", b={b[i]:.17g}" if op in ("add", "mul", "div") else "")
    nonfin = int((~finite).sum())
    return (f"max {e.max():10.2f}   p99.9 {np.quantile(e, 0.999):9.2f}   median {np.median(e):7.3f}"
            f"   at {arg}" + (f"   ({nonfin} non-finite)" if nonfin else ""))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    if not os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS"):
        print("Needs the PJRT plugin (set PJRT_NAMES_AND_LIBRARY_PATHS).")
        return 1
    rng = np.random.default_rng(2026)
    sets = input_sets(rng, n)
    print("=" * 100)
    print(f"  DS per-operation accuracy vs 50-digit decimal reference; {n:,} random inputs per op")
    print("  DS: raw (hi, lo) via DS_RETURN_PAIRS=1, error in units of 2^-49")
    print("  f64: native (DS_BYPASS=1), error in units of 2^-53")
    print("=" * 100)
    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        for op, lst in sets.items():
            print(f"\n[{op}]")
            for name, a, b in lst:
                ref = reference(op, a, b)
                print(f"  {name}  (n={len(a):,})")
                for mode, unit, label in (("ds", U49, "DS "), ("f64", U53, "f64")):
                    res = run_gpu(op, mode, a, b, tmp)
                    if res is None:
                        continue
                    err = rel_errors(ref, *res)
                    print(f"    {label} {summarize(err, unit, a, b, op)}")
                    summary[f"{op} | {name} | {mode}"] = float(np.nanmax(err) / unit)
    print("\nNotes:")
    print("  A relative error of k units of 2^-49 means |hi + lo - exact| / |exact| = k * 2^-49 (~k * 1.8e-15).")
    print("  Inputs are exact DS pairs and exact f64 values, so input splitting contributes no error.")
    print("RESULT " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

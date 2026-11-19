"""
Where does DS Black-Scholes lose accuracy?  Stage-by-stage, against a
50-digit reference.

At 1,024 options DS is ~1,400x less accurate than native f64 (3.2e-11 vs
2.3e-14); at 1M-16M only 21-30x. This script tests three explanations:

  A. Constants. Do the kernel's f64 literals (0.2316419, 0.319381530, ...,
     1/sqrt(2*pi)) reach the GPU as full (hi, lo) pairs, or as (f32(c), 0)?
     Checked on the GPU: x * c for x = 1 with DS_RETURN_PAIRS=1 must return
     exactly c's DS split.
  B. Constants passed as function arguments instead of literals. If the
     final error changes, literals are mishandled.
  C. Per stage: every intermediate (sqrt(T), log(S/K), d1, d2, n'(d1),
     the rational term k, N(d1), N(d2), exp(-rT), the final price) returned
     as raw DS pairs and compared with the same formula evaluated in
     Python's decimal at 50 digits on the exact DS-split inputs. Native f64
     (DS_BYPASS=1) is measured the same way on its own inputs.

For the final price it also prints a cancellation factor,
(|S N(d1)| + |K e^{-rT} N(d2)|) / |price| for calls (the analogous sum for
puts): error / (factor * unit) near a constant across sizes means the error
is the arithmetic's unit error amplified by cancellation in the formula.

Errors: relative, DS in units of 2^-49, f64 in units of 2^-53. Prices below
1e-3 are excluded from the price row, as in bench_blackscholes.py.

Usage (inside the container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/diag_bs_floor.py [n ...]              # default 1024 65536
"""

import os
import sys
import json
import subprocess
import tempfile
from decimal import Decimal, getcontext

import numpy as np

getcontext().prec = 50
U49, U53 = 2.0 ** -49, 2.0 ** -53

CONSTS = {"INV_SQRT_2PI": 0.39894228040143270286, "half": 0.5, "k0": 0.2316419,
          "a1": 0.319381530, "a2": -0.356563782, "a3": 1.781477937,
          "a4": -1.821255978, "a5": 1.330274429}
# N(d) is computed as 1 - out for d < 0, which cancels: tiny N values come out
# as 0 even in f64. Their error is reported as absolute error (N is in [0, 1]).
ABSOLUTE = {"N_d1", "N_d2"}
STAGES = ["sqrt_time", "log_term", "d1", "d2", "nprime_d1", "k_d1", "N_d1", "N_d2",
          "exp_rt", "future_value", "price"]

# Kernel with every intermediate exposed. Same operations, same order as
# bench_blackscholes.py; constants come from `c`, which is either the
# literal values (literal mode) or traced scalar arguments (args mode).
_KERNEL = r"""
def cndf_parts(xp, x, c):
    sign = x < 0.0
    x = xp.where(sign, -x, x)
    n_prime = xp.exp(-c["half"] * x * x) * c["INV_SQRT_2PI"]
    k = 1.0 / (1.0 + c["k0"] * x)
    k2 = k * k; k3 = k2 * k; k4 = k3 * k; k5 = k4 * k
    local_1 = k * c["a1"]
    local_2 = k2 * c["a2"]
    local_2 = local_2 + k3 * c["a3"]
    local_2 = local_2 + k4 * c["a4"]
    local_2 = local_2 + k5 * c["a5"]
    out = 1.0 - (local_2 + local_1) * n_prime
    return xp.where(sign, 1.0 - out, out), n_prime, k

def bs_parts(xp, spot, strike, rate, vol, time_, otype, c):
    sqrt_time = xp.sqrt(time_)
    log_term = xp.log(spot / strike)
    d1 = (rate + vol * vol * c["half"]) * time_ + log_term
    den = vol * sqrt_time
    d1 = d1 / den
    d2 = d1 - den
    n_d1, np_d1, k_d1 = cndf_parts(xp, d1, c)
    n_d2, _, _ = cndf_parts(xp, d2, c)
    exp_rt = xp.exp(-rate * time_)
    future_value = strike * exp_rt
    call = spot * n_d1 - future_value * n_d2
    put = future_value * (1.0 - n_d2) - spot * (1.0 - n_d1)
    price = xp.where(otype == 0, call, put)
    return [sqrt_time, log_term, d1, d2, np_d1, k_d1, n_d1, n_d2, exp_rt, future_value, price]
"""

# GPU side; argv: mode (ds|f64), cmode (literal|args), in .npz, out .npz
_GPU_CODE = r"""
import sys, os
os.environ["JAX_ENABLE_X64"] = "1"
import numpy as np, jax, jax.numpy as jnp
""" + _KERNEL + r"""
mode, cmode, inp, out = sys.argv[1:5]
CONSTS = __CONSTS__
d = np.load(inp)
args = [jnp.array(d[k]) for k in ("spot", "strike", "rate", "vol", "time")] + [jnp.array(d["otype"])]
names = list(CONSTS)
if cmode == "literal":
    def f(*a):
        return bs_parts(jnp, *a, CONSTS)
    cargs = []
else:
    def f(*a):
        base, cs = a[:6], a[6:]
        return bs_parts(jnp, *base, dict(zip(names, cs)))
    cargs = [jnp.array(np.float64(CONSTS[k])) for k in names]
if mode == "ds":
    g = jax.jit(lambda *a: tuple(v for p in f(*a) for v in (p, p)))   # (hi, lo) per stage
    res = g(*args, *cargs)
    np.savez(out, **{f"s{i}_hi": np.asarray(res[2 * i]) for i in range(len(res) // 2)},
             **{f"s{i}_lo": np.asarray(res[2 * i + 1]) for i in range(len(res) // 2)})
else:
    res = jax.jit(lambda *a: tuple(f(*a)))(*args, *cargs)
    np.savez(out, **{f"s{i}_hi": np.asarray(r) for i, r in enumerate(res)},
             **{f"s{i}_lo": np.zeros_like(np.asarray(r)) for i, r in enumerate(res)})

# Part A: does each literal constant arrive as its full DS pair?
if mode == "ds" and cmode == "literal":
    one = jnp.ones(4, jnp.float64)
    chk = {}
    for k, v in CONSTS.items():
        h, l = jax.jit(lambda x: (lambda r: (r, r))(x * v))(one)
        chk[k] = [float(np.asarray(h)[0]), float(np.asarray(l)[0])]
    np.save(out + ".consts.npy", np.array([chk[k] for k in CONSTS]))
""".replace("__CONSTS__", repr(CONSTS))


def ds_split(v):
    """The pass's split of an f64 value (round-to-nearest to f32, then the rest)."""
    h = np.asarray(v, np.float64).astype(np.float32).astype(np.float64)
    l = (np.asarray(v, np.float64) - h).astype(np.float32).astype(np.float64)
    return h, l


def ref_parts(spot, strike, rate, vol, time_, otype):
    """The kernel in decimal, one option at a time; returns list of stage lists."""
    D = Decimal
    C = {k: D(v) for k, v in CONSTS.items()}
    def cndf(x):
        sign = x < 0
        x = -x if sign else x
        npr = (-C["half"] * x * x).exp() * C["INV_SQRT_2PI"]
        k = D(1) / (D(1) + C["k0"] * x)
        poly = (k * k * C["a2"] + k ** 3 * C["a3"] + k ** 4 * C["a4"] + k ** 5 * C["a5"]) + k * C["a1"]
        out = D(1) - poly * npr
        return (D(1) - out if sign else out), npr, k
    rows = []
    for S, K, r, v, T, o in zip(spot, strike, rate, vol, time_, otype):
        S, K, r, v, T = (D(x) for x in (S, K, r, v, T))
        st = T.sqrt(); lt = (S / K).ln()
        den = v * st
        d1 = ((r + v * v * C["half"]) * T + lt) / den
        d2 = d1 - den
        n1, np1, k1 = cndf(d1); n2, _, _ = cndf(d2)
        er = (-r * T).exp(); fv = K * er
        call = S * n1 - fv * n2
        put = fv * (D(1) - n2) - S * (D(1) - n1)
        price = call if o == 0 else put
        cancel = ((abs(S * n1) + abs(fv * n2)) if o == 0 else
                  (abs(fv * (D(1) - n2)) + abs(S * (D(1) - n1)))) / abs(price) if price != 0 else D(0)
        rows.append([st, lt, d1, d2, np1, k1, n1, n2, er, fv, price, cancel])
    return rows


def run(mode, cmode, inputs, tmp):
    inp = os.path.join(tmp, "in.npz"); out = os.path.join(tmp, f"out_{mode}_{cmode}.npz")
    np.savez(inp, **inputs)
    env = dict(os.environ, JAX_ENABLE_X64="1")
    for k in ("DS_BYPASS", "DS_RETURN_PAIRS"):
        env.pop(k, None)
    env["DS_RETURN_PAIRS" if mode == "ds" else "DS_BYPASS"] = "1"
    r = subprocess.run([sys.executable, "-c", _GPU_CODE, mode, cmode, inp, out],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [{mode}/{cmode} failed] {r.stderr[-800:]}")
        return None, None
    d = np.load(out)
    stages = [(d[f"s{i}_hi"].astype(np.float64), d[f"s{i}_lo"].astype(np.float64))
              for i in range(len(STAGES))]
    consts = np.load(out + ".consts.npy") if os.path.exists(out + ".consts.npy") else None
    return stages, consts


def stage_errors(ref_rows, stages, mask):
    """Max relative error per stage over the options in `mask` (price: priced only)."""
    out = []
    for j, (hi, lo) in enumerate(stages):
        errs = []
        for i in np.nonzero(mask[j])[0]:
            r = ref_rows[i][j]
            d = abs(Decimal(hi[i]) + Decimal(lo[i]) - r)
            if STAGES[j] in ABSOLUTE:          # N(d) in [0, 1], computed as 1 - out
                errs.append(float(d))
            elif r != 0:
                errs.append(float(d / abs(r)))
        out.append(np.array(errs) if errs else np.array([np.nan]))
    return out


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [1024, 65536]
    if not os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS"):
        print("Needs the PJRT plugin (set PJRT_NAMES_AND_LIBRARY_PATHS).")
        return 1
    print("=" * 104)
    print("  Black-Scholes DS accuracy by stage, vs a 50-digit decimal reference")
    print("  DS errors in units of 2^-49 (raw pairs, DS_RETURN_PAIRS=1); f64 in units of 2^-53")
    print("=" * 104)
    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n in sizes:
            # Same input distribution and seed as bench_blackscholes.py.
            rng = np.random.default_rng(42)
            spot, strike, rate, vol, time_ = [rng.uniform(lo, hi, n) for lo, hi in
                [(10.0, 200.0), (10.0, 200.0), (0.01, 0.10), (0.05, 0.65), (0.05, 2.0)]]
            otype = rng.integers(0, 2, n).astype(np.int32)
            inputs = dict(spot=spot, strike=strike, rate=rate, vol=vol, time=time_, otype=otype)

            print(f"\n  n = {n:,}")
            # Reference on the exact DS-split inputs (for DS) and on the f64 inputs (for f64).
            split = {k: sum(ds_split(v)) for k, v in inputs.items() if k != "otype"}
            ref_ds = ref_parts(split["spot"], split["strike"], split["rate"], split["vol"], split["time"], otype)
            ref_64 = ref_parts(spot, strike, rate, vol, time_, otype)

            results = {}
            for mode, cmode in (("ds", "literal"), ("ds", "args"), ("f64", "literal")):
                st, consts = run(mode, cmode, inputs, tmp)
                results[(mode, cmode)] = st
                if consts is not None:
                    print("\n  A. Literal constants on the GPU (x * c, x = 1), DS pair vs the exact DS split of c:")
                    for (k, v), (h, l) in zip(CONSTS.items(), consts):
                        eh, el = ds_split(v)
                        ok = (h == eh and l == el)
                        print(f"     {k:13s} c = {v:<22.17g} got ({h:.9g}, {l:.9g})  "
                              f"expected ({float(eh):.9g}, {float(el):.9g})  {'OK' if ok else 'MISMATCH'}")

            priced = np.array([float(r[10]) > 1e-3 for r in ref_64])
            mask = [np.ones(n, bool)] * (len(STAGES) - 1) + [priced]
            print(f"\n  C. Max (and 99.9th pct) relative error per stage over {n:,} options "
                  f"(price row: {int(priced.sum()):,} priced > 1e-3)")
            print(f"     {'stage':14s} {'DS literal':>22s} {'DS consts-as-args':>22s} {'f64 (2^-53)':>22s}")
            cols = {}
            for key, ref in ((("ds", "literal"), ref_ds), (("ds", "args"), ref_ds), (("f64", "literal"), ref_64)):
                st = results.get(key)
                cols[key] = stage_errors(ref, st, mask) if st else None
            for j, name in enumerate(STAGES):
                cells = []
                for key, unit in ((("ds", "literal"), U49), (("ds", "args"), U49), (("f64", "literal"), U53)):
                    e = cols[key][j] if cols[key] is not None else np.array([np.nan])
                    cells.append(f"{np.nanmax(e) / unit:10.1f} ({np.nanquantile(e, 0.999) / unit:7.1f})")
                print(f"     {name:14s} " + " ".join(f"{c:>22s}" for c in cells))
                summary[f"n={n} {name}"] = cells

            # Price error vs cancellation factor.
            if cols[("ds", "literal")] is not None:
                hi, lo = results[("ds", "literal")][10]
                cancel = np.array([float(r[11]) for r in ref_ds])
                err = np.array([float(abs(Decimal(hi[i]) + Decimal(lo[i]) - ref_ds[i][10]) / abs(ref_ds[i][10]))
                                if priced[i] else np.nan for i in range(n)])
                i = int(np.nanargmax(err))
                ratio = err / (cancel * U49)
                print(f"\n     Worst DS price: option {i}, error {err[i] / U49:.1f} x 2^-49, "
                      f"cancellation factor {cancel[i]:.3g}, error/(factor * 2^-49) = {ratio[i]:.2f}")
                print(f"     error/(factor * 2^-49) over priced options: max {np.nanmax(ratio):.2f}, "
                      f"median {np.nanmedian(ratio):.3f}")
    print("\nNotes:")
    print("  A: MISMATCH would mean a literal reaches the GPU without its DS low word.")
    print("  B: if 'DS consts-as-args' differs from 'DS literal', literals are the cause.")
    print("  C: each stage's error includes error carried in from earlier stages (e.g. n'(d1)")
    print("     amplifies d1's relative error by ~d1^2). Read DS and f64 together: DS units")
    print("     (2^-49) equal to f64 units (2^-53) means DS is 16x worse, the expected ratio.")
    print("     A stage where DS units jump well above f64 units is where DS loses accuracy;")
    print("     compare with tests/diag_ds_ops_mp.py for that operation's own error.")
    print("  N_d1, N_d2 rows are absolute errors (N in [0, 1]); all other rows are relative.")
    print("RESULT " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

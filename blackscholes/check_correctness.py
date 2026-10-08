"""Correctness check: vectorized PyTorch port vs. scalar reference vs. PARSEC C.

For each precision (float64 = `fptype double`, float32 = PARSEC's `fptype float`):
  1. reference_scalar.py vs. the verbatim PARSEC C built with gcc (ctypes).
     Expected: bit-identical. This validates the scalar translation itself.
  2. blackscholes_torch.py vs. reference_scalar.py, within a stated tolerance.
     float32 tolerance is looser because the C evaluates double-literal
     expressions in double before rounding to float, while the torch graph
     stays in float32 (see blackscholes_torch.py docstring).

Usage: python check_correctness.py [--n 20000] [--seed 0]
"""

import argparse
import ctypes
import math
import pathlib
import subprocess

import torch

import reference_scalar as ref
from blackscholes_torch import blk_schls_eq_euro_no_div, cndf

HERE = pathlib.Path(__file__).resolve().parent

# name -> (torch dtype, scalar rounding fn, ctypes type, price atol, cndf atol)
PRECISIONS = {
    "float64": (torch.float64, ref.as_double, ctypes.c_double, 1e-12, 1e-15),
    "float32": (torch.float32, ref.as_float, ctypes.c_float, 2e-4, 1e-6),
}


def build_oracle(fptype):
    out = HERE / "build" / f"libparsec_{fptype}.so"
    out.parent.mkdir(exist_ok=True)
    # -ffp-contract=off: forbid FMA contraction so gcc keeps the C's exact op order.
    subprocess.run(
        ["gcc", "-O2", "-ffp-contract=off", "-fPIC", "-shared", f"-Dfptype={fptype}",
         str(HERE / "parsec_oracle.c"), "-o", str(out), "-lm"],
        check=True,
    )
    return ctypes.CDLL(str(out))


def make_inputs(n, seed):
    g = torch.Generator().manual_seed(seed)

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)

    cols = {
        "sptprice": u(10.0, 200.0),
        "strike": u(10.0, 200.0),
        "rate": u(0.01, 0.10),
        "volatility": u(0.05, 0.65),
        "otime": u(0.05, 2.0),
        "otype": torch.randint(0, 2, (n,), generator=g, dtype=torch.int32),
    }
    # Edge cases: at-the-money, deep ITM/OTM, very short/long expiry, low/high vol.
    edge = [
        (100.0, 100.0, 0.05, 0.20, 1.00),
        (100.0, 100.0, 0.01, 0.05, 0.05),
        (200.0, 10.0, 0.10, 0.65, 2.00),
        (10.0, 200.0, 0.01, 0.05, 0.05),
        (42.0, 40.0, 0.10, 0.20, 0.50),
        (100.0, 100.0, 0.0275, 0.65, 1e-4),
    ]
    for s, k, r, v, t in edge:
        for ot in (0, 1):
            for name, val in zip(("sptprice", "strike", "rate", "volatility", "otime"), (s, k, r, v, t)):
                cols[name] = torch.cat([cols[name], torch.tensor([val], dtype=torch.float64)])
            cols["otype"] = torch.cat([cols["otype"], torch.tensor([ot], dtype=torch.int32)])
    return cols


def compare(label, got, want, atol):
    diff = (got - want).abs()
    rel = diff / want.abs().clamp_min(torch.finfo(want.dtype).tiny)
    n_exact = int((got == want).sum())
    worst = float(diff.max())
    ok = worst <= atol
    print(f"  {label:<34} n={want.numel():>6}  bit-exact={n_exact:>6}  "
          f"max|diff|={worst:.3e}  max rel={float(rel.max()):.3e}  "
          f"tol={atol:.0e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_precision(name, cols):
    dtype, F, ctype, price_atol, cndf_atol = PRECISIONS[name]
    lib = build_oracle("double" if dtype == torch.float64 else "float")
    lib.CNDF.restype = ctype
    lib.CNDF.argtypes = [ctype]
    lib.BlkSchlsEqEuroNoDiv.restype = ctype
    lib.BlkSchlsEqEuroNoDiv.argtypes = [ctype] * 5 + [ctypes.c_int, ctypes.c_float]

    fcols = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in cols.items()}
    # .tolist() on a float32 tensor yields the exact float32 values as Python floats.
    rows = list(zip(*(fcols[k].tolist() for k in
                      ("sptprice", "strike", "rate", "volatility", "otime", "otype"))))

    print(f"[{name}]")
    ok = True

    # CNDF on a grid that crosses the sign branch, including +0.0 and -0.0.
    xs = torch.cat([torch.linspace(-10.0, 10.0, 4001, dtype=torch.float64),
                    torch.tensor([0.0, -0.0, 1e-8, -1e-8])]).to(dtype)
    x_list = xs.tolist()
    cndf_ref = torch.tensor([ref.CNDF(x, F) for x in x_list], dtype=dtype)
    cndf_c = torch.tensor([lib.CNDF(x) for x in x_list], dtype=dtype)
    ok &= compare("CNDF: scalar ref vs PARSEC C", cndf_ref, cndf_c, 0.0)
    ok &= compare("CNDF: torch vs scalar ref", cndf(xs), cndf_ref, cndf_atol)

    price_ref = torch.tensor([ref.BlkSchlsEqEuroNoDiv(*r, F=F) for r in rows], dtype=dtype)
    price_c = torch.tensor([lib.BlkSchlsEqEuroNoDiv(*r, 0.0) for r in rows], dtype=dtype)
    ok &= compare("price: scalar ref vs PARSEC C", price_ref, price_c, 0.0)
    price_torch = blk_schls_eq_euro_no_div(*(fcols[k] for k in
                                             ("sptprice", "strike", "rate", "volatility", "otime", "otype")))
    ok &= compare("price: torch vs scalar ref", price_torch, price_ref, price_atol)
    ok &= bool(torch.isfinite(price_torch).all())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cols = make_inputs(args.n, args.seed)
    results = {name: check_precision(name, cols) for name in PRECISIONS}
    print("ALL PASS" if all(results.values()) else f"FAILURES: {results}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

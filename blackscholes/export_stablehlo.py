"""Export the vectorized Black-Scholes model to StableHLO via torch_xla.

    torch.export.export(BlackScholes(), sample_args)
      -> torch_xla.stablehlo.exported_program_to_stablehlo(ep)
      -> stablehlo/blackscholes_<dtype>_b<batch>.mlir

float64 is the form the DS StableHLO pass consumes; float32 matches PARSEC's
default `fptype float`. The batch dimension is static. torch_xla may number the
MLIR arguments differently from the Python signature, so each saved file starts
with a `// %argN = <name>` header. The exported program is executed and checked
against reference_scalar.py with the same tolerances as check_correctness.py.

Usage: python export_stablehlo.py [--batch 1024] [--dtype float64 float32] [--quiet]
"""

import argparse
import collections
import os
import pathlib
import re

os.environ.setdefault("PJRT_DEVICE", "CPU")

import torch
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

import reference_scalar as ref
from blackscholes_torch import BlackScholes
from check_correctness import PRECISIONS

HERE = pathlib.Path(__file__).resolve().parent
DTYPES = {"float64": torch.float64, "float32": torch.float32}
PARAM_NAMES = ("sptprice", "strike", "rate", "volatility", "otime", "otype")


def sample_args(batch, dtype):
    g = torch.Generator().manual_seed(0)

    def u(lo, hi):
        return (lo + (hi - lo) * torch.rand(batch, generator=g, dtype=torch.float64)).to(dtype)

    return (u(10.0, 200.0), u(10.0, 200.0), u(0.01, 0.10), u(0.05, 0.65), u(0.05, 2.0),
            torch.randint(0, 2, (batch,), generator=g, dtype=torch.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dtype", nargs="+", choices=list(DTYPES), default=list(DTYPES))
    ap.add_argument("--quiet", action="store_true", help="don't print the StableHLO text")
    args = ap.parse_args()

    out_dir = HERE / "stablehlo"
    out_dir.mkdir(exist_ok=True)
    model = BlackScholes().eval()
    ok = True

    for name in args.dtype:
        sample = sample_args(args.batch, DTYPES[name])
        ep = export(model, sample)
        shlo = exported_program_to_stablehlo(ep)
        text = shlo.get_stablehlo_text("forward")

        locations = shlo._bundle.stablehlo_funcs[0].meta.input_locations
        assert all(loc.type_.value == "input_arg" for loc in locations), locations
        header = f"// BlackScholes {name}, batch {args.batch}. Argument mapping:\n" + "".join(
            f"//   %arg{i} = {PARAM_NAMES[loc.position]}\n" for i, loc in enumerate(locations))
        text = header + text

        path = out_dir / f"blackscholes_{name}_b{args.batch}.mlir"
        path.write_text(text)
        print(f"===== {name}: wrote {path.relative_to(HERE)} =====")
        if not args.quiet:
            print(text)

        ops = collections.Counter(re.findall(r"stablehlo\.([a-z_]+)", text))
        print(f"----- {name}: StableHLO op counts -----")
        for op, count in sorted(ops.items()):
            print(f"  stablehlo.{op:<16} {count}")

        # Run the exported program and check it against the scalar reference.
        _, F, _, price_atol, _ = PRECISIONS[name]
        rows = zip(*(t.tolist() for t in sample))
        want = torch.tensor([ref.BlkSchlsEqEuroNoDiv(*r, F=F) for r in rows], dtype=DTYPES[name])
        exported = shlo(*sample).cpu()
        vs_ref = float((exported - want).abs().max())
        vs_eager = float((exported - model(*sample)).abs().max())
        passed = vs_ref <= price_atol
        ok &= passed
        print(f"----- {name}: exported vs scalar ref max|diff| = {vs_ref:.3e} (tol {price_atol:.0e}) "
              f"{'PASS' if passed else 'FAIL'}; exported vs eager = {vs_eager:.3e} -----\n")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

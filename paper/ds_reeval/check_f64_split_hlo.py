#!/usr/bin/env python3
"""
Diagnostic (not a regression test): dumps the FULLY XLA-OPTIMIZED HLO for a
minimal f64-input DS computation and checks whether emitFromFloat's f64
split subtract (`diff = v - convert(convert(v, f32), f64)`, i.e. `hi_as_f64`)
survives into the module that actually executes on the GPU.

Motivation: every prior inspection in this investigation (mlir-ds-opt run
standalone, structural op-count tests) only ever looked at the DS pass's
OWN output -- the StableHLO it hands to the real CUDA backend -- not what
XLA's OWN optimizer does to that StableHLO afterward. Hypothesis (see
handoff discussion): XLA's algebraic simplifier may be folding
`convert(convert(v, f32), f64)` back to `v` outright -- narrow-then-widen
looks like a no-op ONLY when v already fits exactly in f32, which is false
in general and is exactly the class of hazard this project's own FMA/
simplifier-contraction concerns (Experiment 2b) already flagged for other
op sequences. If that fold happens, `diff = v - v = 0` identically for
EVERY f64 input, and `lo = convert(diff, f32) = 0` always -- which would
mean f64 inputs have been silently truncated to f32 at the split, for
every DS op that has ever run on f64 input, not just the multiply-by-a-
zero-lo-constant case found in split_fidelity.

Uses this project's existing XLA-dump convention (see exp2_tf32_dispatch.py):
XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text", set before jax
import, then grep the `*after_optimizations*.txt` file(s) -- the fully
optimized module that is actually compiled to PTX and executed.

Usage:
  python3 check_f64_split_hlo.py --dump-dir /tmp/f64_split_dump
"""
import argparse
import glob
import os
import re
import sys


def find_optimized_hlo_files(dump_dir):
    patterns = ["*after_optimizations*.txt", "*after-optimizations*.txt", "*.txt"]
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(dump_dir, pat)))
        if matches:
            return matches
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--val", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    existing = os.environ.get("XLA_FLAGS", "")
    dump_flags = "--xla_dump_to={} --xla_dump_hlo_as_text".format(args.dump_dir)
    os.environ["XLA_FLAGS"] = (existing + " " + dump_flags).strip()

    import jax
    import jax.numpy as jnp
    import numpy as np

    # Observed-behavior check (not just the config flag), matching
    # common.assert_x64_enabled()'s approach -- kept standalone
    # deliberately, since this script must set XLA_FLAGS before ANY jax
    # import, including common's.
    dtype_check = jnp.zeros(1).dtype
    if dtype_check != jnp.float64:
        print("FAIL: JAX_ENABLE_X64 not active (jnp.zeros(1).dtype == {}); "
              "set JAX_ENABLE_X64=1 before running this script.".format(dtype_check),
              file=sys.stderr)
        sys.exit(1)

    a_np = np.full(args.n, args.val, dtype=np.float64)
    a = jnp.asarray(a_np)

    # Minimal computation that forces the f64 split to matter: sum of
    # squares, matching what the rest of this investigation has been
    # measuring. Single return -- irrelevant to what's being checked here
    # (the split happens at argument-processing time, before any return
    # logic runs at all).
    @jax.jit
    def fn(x):
        return jnp.sum(x * x)

    result = fn(a)
    result.block_until_ready()
    print("[check_f64_split_hlo] result = {!r} (n={}, val={})".format(
        float(result), args.n, args.val), file=sys.stderr)

    hlo_files = find_optimized_hlo_files(args.dump_dir)
    if not hlo_files:
        print("FAIL: no optimized HLO dump files found under {}".format(args.dump_dir),
              file=sys.stderr)
        sys.exit(1)

    # The split's subtract is the ONLY subtract that directly involves the
    # raw f64 parameter (the reduction body's own two_sum machinery has
    # many subtracts too, but those operate on the f32 hi/lo components,
    # never on the f64-typed parameter itself). Search for a subtract
    # whose operands include the f64 parameter -- report every subtract
    # line for manual inspection either way, since HLO variable naming
    # varies by XLA version.
    subtract_lines = []
    convert_lines = []
    for path in hlo_files:
        with open(path, "r", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if re.search(r"=\s*f64\[.*\]\s*subtract\(", stripped) or \
                   (" subtract(" in stripped and "f64" in stripped):
                    subtract_lines.append((path, lineno, stripped))
                if "convert(" in stripped and ("f64" in stripped or "f32" in stripped):
                    convert_lines.append((path, lineno, stripped))

    print("\n=== Optimized HLO files inspected ===", file=sys.stderr)
    for p in hlo_files:
        print("  " + p, file=sys.stderr)

    print("\n=== f64-involving subtract lines in optimized HLO ===", file=sys.stderr)
    if subtract_lines:
        for path, lineno, line in subtract_lines:
            print("  {}:{}: {}".format(os.path.basename(path), lineno, line), file=sys.stderr)
    else:
        print("  (none found)", file=sys.stderr)

    print("\n=== convert(...) lines in optimized HLO (first 40) ===", file=sys.stderr)
    for path, lineno, line in convert_lines[:40]:
        print("  {}:{}: {}".format(os.path.basename(path), lineno, line), file=sys.stderr)
    if len(convert_lines) > 40:
        print("  ... ({} more)".format(len(convert_lines) - 40), file=sys.stderr)

    verdict = (
        "SPLIT SUBTRACT SURVIVES (found {} f64-involving subtract(s)) -- "
        "convert-folding hypothesis NOT confirmed by this check alone; "
        "read the lines above to judge whether they're the split's own "
        "residual subtract.".format(len(subtract_lines))
        if subtract_lines else
        "NO f64-involving subtract found in optimized HLO -- CONSISTENT "
        "with the convert-folding hypothesis (the split's "
        "`diff = v - convert(convert(v,f32),f64)` subtract may have been "
        "eliminated by XLA's optimizer, which would make lo always zero)."
    )
    print("\n=== VERDICT ===\n{}".format(verdict), file=sys.stderr)


if __name__ == "__main__":
    main()

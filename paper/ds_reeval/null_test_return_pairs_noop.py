#!/usr/bin/env python3
"""
Null test for DS_RETURN_PAIRS: verifies the flag is truly a no-op for a
function whose return is NOT the exact doubled-return pattern it's
documented to special-case.

Motivated by exp5_f64_reduction_bisection.py's length_scan stage, which
showed a plain single-return f64 computation giving a DIFFERENT numerical
result depending solely on whether DS_RETURN_PAIRS=1 was set in the
process environment -- something the documented design says should be
impossible ("A value returned only once... is unaffected (falls back to
normal recombination)"). That symptom appeared only in code paths using
the NEW f64 extension to DS_RETURN_PAIRS (added this session); this test
isolates whether the ORIGINAL, already-shipped f32-only flag mechanism --
the one Experiment 3's headline figure was measured through -- has the
same property, independent of anything f64-specific.

Two dtypes tested:
  --dtype f32 : exercises ONLY the original, pre-existing flag code path
                (orig.getType() == hi.getType() branch). This is the
                mechanism Experiment 3 relies on.
  --dtype f64 : exercises the new extension added this session.

Same single-return computation (jnp.sum(x*x)), each run TWICE in separate
fresh processes (DS_RETURN_PAIRS unset vs =1 -- env vars are read at
process start and cannot be toggled mid-process). Compare the two runs'
result_hex fields: float.hex() gives the exact IEEE-754 bit pattern, so
"identical" here means genuinely bit-for-bit identical, not merely close
-- any difference at all would mean the flag has an effect its own
documentation says is impossible for this case.

Usage:
  python3 null_test_return_pairs_noop.py --dtype f32 --flag off --out /tmp/f32_off.json
  python3 null_test_return_pairs_noop.py --dtype f32 --flag on  --out /tmp/f32_on.json
  python3 null_test_return_pairs_noop.py --dtype f64 --flag off --out /tmp/f64_off.json
  python3 null_test_return_pairs_noop.py --dtype f64 --flag on  --out /tmp/f64_on.json
  # then diff each pair's result_hex field
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", required=True, choices=["f32", "f64"])
    p.add_argument("--flag", required=True, choices=["on", "off"],
                    help="Informational only -- the actual flag state comes from "
                         "DS_RETURN_PAIRS in the process environment, which the "
                         "caller must set to match before invoking this script.")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    gpu = common.detect_gpu(required=True)

    if args.dtype == "f64":
        common.assert_x64_enabled()
        np_dtype = np.float64
    else:
        np_dtype = np.float32

    a_np = np.full(args.n, args.val, dtype=np_dtype)
    a = jnp.asarray(a_np)

    # Single, non-doubled return -- per DS_RETURN_PAIRS's documented scope,
    # this should be COMPLETELY UNAFFECTED by the flag, for either dtype.
    @jax.jit
    def fn(x):
        return jnp.sum(x * x)

    result = fn(a)
    common.block_until_ready(result)
    result_f = float(result)

    result_json = {
        "test": "null_test_return_pairs_noop",
        "dtype": args.dtype,
        "flag_arg": args.flag,
        "gpu": gpu,
        "env_relevant": common.snapshot_relevant_env(),
        "git": common.git_commit_info(),
        "n": args.n,
        "val": args.val,
        "result_float": result_f,
        "result_hex": result_f.hex(),
        "jax_version": jax.__version__,
        "timestamp": common.utc_now_iso(),
    }
    with open(args.out, "w") as f:
        json.dump(result_json, f, indent=2)
        f.write("\n")
    print(
        "[null_test] dtype={} flag_arg={} DS_RETURN_PAIRS_env={} result={!r} hex={}".format(
            args.dtype, args.flag, result_json["env_relevant"]["DS_RETURN_PAIRS"],
            result_f, result_f.hex()),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment 3 worker -- unrecombined accuracy: measure the (hi, lo) pair
directly.

Claim under test / hypothesis: the paper's f32-input sum-of-squares
reduction (n=10,000 elements of 0.1_f32, true result ~100) reports DS-f32
error 4.65e-6 vs plain-f32 error ~7-8e-5, i.e. ~15x (paper/main.tex
"Reduction Precision", ~line 587-600). The hypothesis under test here is
that 4.65e-6 is dominated by output-quantization at the f32 function
return (half-ulp of f32 at magnitude ~100 is ~3.8e-6), not by accumulated
DS arithmetic error -- and that the *internal* DS precision gain is much
larger than 15x once you stop throwing it away at the return boundary.

Do not confuse this with the *separate* f64-input version of the same
reduction reported elsewhere in the paper (~line 602-606, a 8x figure);
that one already has its own explanation (input-representation error) and
is not what this experiment re-measures.

Two modes, each its own process (DS_RETURN_PAIRS, like DS_BYPASS, is read
at plugin/pass load time and cannot be toggled mid-process):

  --mode standard : ordinary jitted reduction, single f32 return value.
                     Should reproduce the paper's 4.65e-6 (error (a)).
  --mode pairs    : DS_RETURN_PAIRS=1 must be set in the environment. The
                     jitted function returns the same DS-tracked value
                     *twice* (`return s, s`); under DS_RETURN_PAIRS=1 the
                     pass substitutes (hi, lo) for the two occurrences
                     instead of recombining each independently -- see
                     stablehlo_pass/DsTransformPass.cpp and
                     ds_reeval/test_return_pairs_structural.py. Host
                     recombines in f64: np.float64(hi) + np.float64(lo)
                     (error (b)).

report.py joins the two output JSONs and applies the verdict logic:
  b << a  and  a ~= half-ulp floor  -> CONFIRMED (output quantization)
  b ~= a                            -> REFUTED
  otherwise                         -> INCONCLUSIVE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["standard", "pairs"])
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    gpu = common.detect_gpu(required=True)
    # This claim is explicitly about f32 inputs (paper's own protocol) --
    # do not assert x64 here (it would be asserting the wrong thing); just
    # record what was observed for transparency in the JSON.
    x64_status = common.x64_status()

    a_np = np.full((args.n,), args.val, dtype=np.float32)
    a = jnp.asarray(a_np)

    # Ground truth: cast the *actual* f32 input values to f64 and sum in
    # f64 -- not np.float64(args.val) ** 2 * n, which would silently use a
    # different (more precise) literal than what the f32 inputs actually
    # hold. See handoff.md "f64 truth must use actual f32 input values".
    truth = float(np.sum(a_np.astype(np.float64) ** 2))

    if args.mode == "standard":
        @jax.jit
        def fn(x):
            return jnp.sum(x * x)

        result = fn(a)
        common.block_until_ready(result)
        measured = float(result)
        error = abs(measured - truth)
        extra = {"measured_f32_result": measured}
    else:
        # Requires DS_RETURN_PAIRS=1 in the environment (set by the parent
        # driver before this process started -- see ds_reeval/README.md).
        # Returning the same DS-tracked value twice keeps the function's
        # output arity/types exactly as JAX originally traced them (two f32
        # scalars) regardless of whether DS_RETURN_PAIRS is on or off, so
        # there is no risk of a PJRT output-count mismatch: only *which*
        # value (hi vs. the ordinary recombination) ends up in each slot
        # changes. See DsTransformPass.cpp's func.return handling.
        @jax.jit
        def fn(x):
            s = jnp.sum(x * x)
            return s, s

        hi, lo = fn(a)
        common.block_until_ready((hi, lo))
        hi_f, lo_f = float(hi), float(lo)
        measured = float(np.float64(hi_f) + np.float64(lo_f))
        error = abs(measured - truth)
        extra = {"hi": hi_f, "lo": lo_f, "host_recombined_f64_result": measured}
        if hi_f == lo_f:
            # If DS_RETURN_PAIRS didn't actually take effect (e.g. env var
            # not propagated), both slots would fall back to the ordinary
            # recombined value and hi == lo -- that's a silent-failure mode
            # worth catching loudly rather than reporting a misleadingly
            # small "error (b)".
            print(
                "[exp3:pairs] WARNING: hi == lo ({}) -- DS_RETURN_PAIRS may not be active; "
                "expected two different raw DS components. Check the environment and rerun "
                "test_return_pairs_structural.py.".format(hi_f),
                file=sys.stderr,
            )
            extra["suspicious_hi_equals_lo"] = True

    half_ulp = common.half_ulp_f32(truth)

    result_json = {
        "experiment": "exp3_pair_accuracy",
        "mode": args.mode,
        "n": args.n,
        "val": args.val,
        "gpu": gpu,
        "env_relevant": common.snapshot_relevant_env(),
        "x64_status": x64_status,
        "truth_f64": truth,
        "error": error,
        "half_ulp_f32_floor": half_ulp,
        **extra,
        "jax_version": jax.__version__,
        "timestamp": common.utc_now_iso(),
    }

    out_path = args.out or str(common.default_result_path(gpu, "exp3_pair_accuracy", args.mode))
    common.write_json_atomic(out_path, result_json)
    print(
        "[exp3:{}] error={:.6e} half_ulp_floor={:.6e}; wrote {}".format(
            args.mode, error, half_ulp, out_path),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

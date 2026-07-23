#!/usr/bin/env python3
"""
Experiment 5 worker -- bisects the f64-input reduction ingestion path to
root-cause (or narrowly bracket) the residual described in paper/main.tex's
Reduction Precision section: for a sum-of-squares reduction over
n = 10,000 with f64 inputs, DS-f32 error is ~1.49e-5 absolute (~1.5e-7
relative) -- roughly f32-epsilon scale, and roughly nine orders of
magnitude worse than the same reduction's internal DS accuracy with f32
inputs (2.84e-14, from Experiment 3's pairs mode).

RESULT AS OF THIS WRITING: Stage 1 (ground_truth) answers the question on
its own and does not depend on anything below -- see its finding: the
existing f64-input reduction test measures a different operation (plain
sum) than the f32-input claim it's compared against (sum of squares),
which alone explains the gap without any pass defect.

FIXED (previously "known unreliability"): stages split_fidelity/single_op/
length_scan/recombination all depend on DsTransformPass.cpp's
DS_RETURN_PAIRS f64-return extension, which initially showed two
unexplained symptoms -- split_fidelity's lo always reading back as zero,
and a plain non-doubled f64 return giving a different numerical result
depending solely on whether DS_RETURN_PAIRS=1 was set. Root-caused and
fixed: the func.return substitution logic decided whether to substitute
hi/lo per-operand from "is this the 1st or 2nd occurrence of this value
SEEN SO FAR" (a running index), with no check on how many times the value
appears in TOTAL -- so a value returned exactly once also hit "1st
occurrence" and got silently substituted with hi alone, dropping lo,
instead of falling through to ordinary recombination. This bug predates
this session's f64 extension (present since DS_RETURN_PAIRS was first
added) but was invisible for f32: hi alone and hi+lo rounded back to f32
are typically bit-identical anyway (the same output-quantization effect
documented throughout this project), and no prior test exercised "single
return, flag on" at all -- Experiment 3 only ever used the genuine
doubled-return pattern. f64 has enough precision to make the dropped lo
visible, which is how this surfaced. Fixed in DsTransformPass.cpp by
counting each value's TOTAL occurrences across the return list first and
only ever substituting when that total is exactly 2; confirmed via
null_test_return_pairs_noop.py (f64 flag on/off now bit-identical for a
single return) and a new structural regression case in
test_return_pairs_structural.py. Does not affect Experiment 3's own
figure or exp4's divide worst-case measurement -- both use the
already-correct doubled-return pattern on f32 arrays throughout, confirmed
unaffected before this fix was even found (see null_test_return_pairs_noop.py's
f32 case, which was bit-identical from the start, and Experiment 3's own
reproduction run, which matched the paper's published 2.84e-14 exactly).

Five stages, EACH ITS OWN PROCESS (DS_RETURN_PAIRS, like DS_BYPASS, is read
at plugin/pass load time and cannot be toggled mid-process):

  --stage ground_truth      : cheapest, most embarrassing check first.
                               Reproduces tests/test_f64_ds.py's existing
                               f64 reduction test EXACTLY as written, next
                               to the apples-to-apples version (same
                               operation, jnp.sum(x*x), as the f32-input
                               claim) at f64 input. Does NOT need
                               DS_RETURN_PAIRS.
  --stage split_fidelity    : verifies f64->DS argument splitting, both
                               on the host (independent of the pass) and
                               through the compiled pass via
                               DS_RETURN_PAIRS=1. REQUIRES DS_RETURN_PAIRS=1.
  --stage single_op         : one element per test value, f64 in, DS
                               square, raw pair via DS_RETURN_PAIRS=1,
                               compared to f64 ground truth.
                               REQUIRES DS_RETURN_PAIRS=1.
  --stage length_scan       : n in {10,100,1000,10000}, both the internal
                               (DS_RETURN_PAIRS doubled-return trick) and
                               observable (ordinary single return, which
                               is NOT special-cased by DS_RETURN_PAIRS --
                               see README.md's "Scope" note) paths in the
                               same process. REQUIRES DS_RETURN_PAIRS=1
                               (the observable path is unaffected by it).
  --stage recombination     : the doubled-return (hi, lo) pair for the
                               n=10000 sum-of-squares reduction, host-
                               recombined in f64. report.py cross-
                               references this against ground_truth's own
                               standard-path (pass-recombined) result at
                               the same n to determine whether
                               recombination itself is where any residual
                               gap opens up. REQUIRES DS_RETURN_PAIRS=1.

report.py assembles all five JSONs into a per-stage table and states
either a root cause or the narrowest bracket the stages establish
("error absent at stage X, present at stage Y").
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DOUBLE_WORD_CLASS_BOUND = 2.0 ** -40  # same grounding used throughout this
                                        # suite/project for "clean" DS accuracy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                    choices=["ground_truth", "split_fidelity", "single_op",
                             "length_scan", "recombination"])
    p.add_argument("--n", type=int, default=10000,
                    help="Reduction length for ground_truth/recombination stages")
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _base_json(stage, gpu, extra):
    import jax
    doc = {
        "experiment": "exp5_f64_reduction_bisection",
        "stage": stage,
        "gpu": gpu,
        "env_relevant": common.snapshot_relevant_env(),
        "x64_status": common.x64_status(),
        "git": common.git_commit_info(),
        "jax_version": jax.__version__,
        "timestamp": common.utc_now_iso(),
    }
    doc.update(extra)
    return doc


def stage_ground_truth(args, gpu):
    import jax
    import jax.numpy as jnp
    import numpy as np

    common.assert_x64_enabled()
    n, val = args.n, args.val

    # --- (a) reproduce tests/test_f64_ds.py's EXISTING f64 reduction test
    # exactly as currently written ("3b. DS-f32 sum precision"):
    #   a = jnp.full(n, np.float64(0.1)); fn_sum = jax.jit(lambda a: jnp.sum(a))
    #   f64_val = float(np.sum(np.full(n, 0.1, np.float64)))
    # This is a PLAIN SUM, not a sum of squares.
    a_existing = jnp.full(n, np.float64(val))

    @jax.jit
    def fn_sum_existing(a):
        return jnp.sum(a)

    ds_existing = float(common.block_until_ready(fn_sum_existing(a_existing)))
    truth_existing = float(np.sum(np.full(n, val, np.float64)))
    f32_baseline_existing = float(np.sum(np.full(n, np.float32(val))))
    existing_test = {
        "description": "Reproduces tests/test_f64_ds.py's existing f64 "
                        "reduction test EXACTLY: jnp.sum(a), a = f64 array "
                        "of value {} repeated {} times.".format(val, n),
        "operation": "plain_sum",
        "ds_result": ds_existing,
        "truth_f64": truth_existing,
        "ds_error_abs": abs(ds_existing - truth_existing),
        "ds_error_rel": abs(ds_existing - truth_existing) / abs(truth_existing),
        "f32_baseline_result": f32_baseline_existing,
        "f32_baseline_error_abs": abs(f32_baseline_existing - truth_existing),
    }

    # --- (b) the apples-to-apples operation: SAME op (jnp.sum(x*x)) as the
    # f32-input reduction claim, this time with f64-native input.
    a_corrected = jnp.full(n, np.float64(val))

    @jax.jit
    def fn_sumsq_corrected(a):
        return jnp.sum(a * a)

    ds_corrected = float(common.block_until_ready(fn_sumsq_corrected(a_corrected)))
    actual_inputs = np.full(n, val, dtype=np.float64)
    # math.fsum: compensated summation, so "truth" doesn't itself pick up
    # f64 accumulation rounding at n=10000 (negligible here vs. what's
    # being measured, but keeps truth strictly more precise than measured).
    truth_corrected = math.fsum(float(x) ** 2 for x in actual_inputs)
    f32_baseline_corrected = float(np.sum(np.full(n, np.float32(val)) ** 2))
    corrected_test = {
        "description": "Apples-to-apples version: SAME operation "
                        "(jnp.sum(x*x)) as the f32-input reduction test, "
                        "with f64-native input instead of f32.",
        "operation": "sum_of_squares",
        "ds_result": ds_corrected,
        "truth_f64_fsum": truth_corrected,
        "ds_error_abs": abs(ds_corrected - truth_corrected),
        "ds_error_rel": abs(ds_corrected - truth_corrected) / abs(truth_corrected),
        "f32_baseline_result": f32_baseline_corrected,
        "f32_baseline_error_abs": abs(f32_baseline_corrected - truth_corrected),
    }

    methodology_finding = (
        "MISMATCH: tests/test_f64_ds.py's existing f64-input reduction "
        "test computes jnp.sum(a) (a plain sum), not jnp.sum(a*a) (sum of "
        "squares) -- a DIFFERENT mathematical operation from the f32-input "
        "reduction test that paper/main.tex's prose describes as 'the same "
        "reduction'. The existing test's ground truth "
        "(np.sum(np.full(n, val, np.float64))) IS computed correctly from "
        "the actual f64 input values -- this is not a separate "
        "ground-truth-computation bug -- but the two 'DS-f32 error' "
        "figures placed side by side in the paper measure two different "
        "computations, which alone is sufficient to explain a large gap "
        "between them without requiring any pass defect. See "
        "corrected_apples_to_apples_test above for the number that "
        "actually corresponds to the f32-input claim's operation."
    )

    extra = {
        "n": n, "val": val,
        "existing_test_reproduction": existing_test,
        "corrected_apples_to_apples_test": corrected_test,
        "same_operation_as_f32_input_claim": False,
        "methodology_finding": methodology_finding,
    }
    return _base_json("ground_truth", gpu, extra)


def stage_split_fidelity(args, gpu):
    import jax
    import jax.numpy as jnp
    import numpy as np

    common.assert_x64_enabled()

    # A diverse set of f64 values, not just one repeated constant: the
    # paper's actual 0.1, values with long mantissas, wide magnitude
    # spread, and a negative value.
    values = np.array([
        0.1, 1.0 / 3.0, np.pi, np.e, 1e8, 1e-8, -0.1, 123456.789012345,
        1.0 + 2.0 ** -40, 2.0 ** 60, 2.0 ** -60,
    ], dtype=np.float64)

    # Host-side split, independent of the compiled pass.
    host_hi = values.astype(np.float32)
    host_lo = (values - host_hi.astype(np.float64)).astype(np.float32)
    host_reconstructed = host_hi.astype(np.float64) + host_lo.astype(np.float64)
    host_exact = bool(np.array_equal(host_reconstructed, values))

    # Compiled-pass split, read back via DS_RETURN_PAIRS=1. NOT a bare
    # `return x, x` identity function -- confirmed (via the MLIR the pass
    # itself emits, checked directly with mlir-ds-opt on a hand-written
    # input) that the pass's own rewrite is correct in that case, but the
    # end-to-end compiled result reads back lo as zero regardless.
    #
    # Three prior attempts at forcing "genuine computation" all failed
    # identically: `y = x + 0.0`, `y = (x + x) - x`, and
    # `y = jax.lax.optimization_barrier(x)`. The first two are exactly,
    # provably equal to x under IEEE-754 (not just approximately, or only
    # under a specific syntactic pattern) -- XLA's optimizer is evidently
    # sophisticated enough to prove that numerically and CSE the result
    # straight back to x's own split values, regardless of the arithmetic
    # used to arrive there. optimization_barrier avoids that, but isn't in
    # this pass's list of recognized ops (add/sub/mul/div/sqrt/negate/abs/
    # compare/select/max/min/dot_general/reduce), so its output is never
    # entered into dsMap -- DS_RETURN_PAIRS has nothing DS-tracked to
    # substitute for it, and x passes through untouched instead.
    #
    # `y = x * 2.0` sidesteps both problems: it is NOT equal to x (a
    # different value XLA has no reason to fold back to x's own split),
    # and multiply IS a recognized, DS-tracked op (emitDsMul). Multiplying
    # by exactly 2 never rounds for any non-overflowing finite value, so
    # y's *combined* (hi+lo, in f64) value should reconstruct 2*v exactly
    # if the split feeding into it was exact -- this doesn't isolate the
    # split from emitDsMul's own arithmetic the way a true identity would
    # have, but it does confirm the split+multiply pipeline is lossless
    # end-to-end, and can be read directly against single_op's finding
    # (which already exercises emitDsMul on f64-sourced values via x*x).
    @jax.jit
    def fn_times_two(x):
        y = x * 2.0
        return y, y

    a = jnp.asarray(values)
    hi, lo = fn_times_two(a)
    common.block_until_ready((hi, lo))
    hi_np = np.asarray(hi, dtype=np.float32).astype(np.float64)
    lo_np = np.asarray(lo, dtype=np.float32).astype(np.float64)
    reconstructed_2v = hi_np + lo_np
    recovered_v = reconstructed_2v / 2.0  # exact: dividing by 2 never rounds
    rel_err = np.abs(recovered_v - values) / np.maximum(np.abs(values), 1e-300)

    compiled_exact = bool(np.array_equal(recovered_v, values))
    clean = bool(np.all(rel_err < 2.0 ** -40))
    extra = {
        "values_tested": values.tolist(),
        "host_split": {
            "hi": host_hi.tolist(), "lo": host_lo.tolist(),
            "reconstruction_exact_for_every_element": host_exact,
        },
        "compiled_split_times_two_via_DS_RETURN_PAIRS": {
            "hi_of_2v": (hi_np).tolist(), "lo_of_2v": (lo_np).tolist(),
            "recovered_v": recovered_v.tolist(),
            "rel_err_vs_true_v": rel_err.tolist(),
            "max_rel_err": float(np.max(rel_err)),
            "reconstruction_exact_for_every_element": compiled_exact,
        },
        "verdict": ("CLEAN: split+multiply-by-2 pipeline reconstructs every "
                     "value to double-word accuracy (max rel_err {:.3e})".format(
                         float(np.max(rel_err))) if clean else
                     "F32-SCALE OR WORSE ERROR (max rel_err {:.3e}): split+multiply-by-2 "
                     "does not reconstruct v to double-word accuracy -- see "
                     "compiled_split_times_two_via_DS_RETURN_PAIRS above".format(
                         float(np.max(rel_err)))),
    }
    return _base_json("split_fidelity", gpu, extra)


def stage_single_op(args, gpu):
    import jax
    import jax.numpy as jnp
    import numpy as np

    common.assert_x64_enabled()

    values = np.array([0.1, 1.0 / 3.0, np.pi, 1e8, 1e-8, -0.1], dtype=np.float64)

    @jax.jit
    def fn_square(x):
        r = x * x
        return r, r  # DS_RETURN_PAIRS: raw (hi, lo)

    a = jnp.asarray(values)
    hi, lo = fn_square(a)
    common.block_until_ready((hi, lo))
    hi_np = np.asarray(hi, dtype=np.float32).astype(np.float64)
    lo_np = np.asarray(lo, dtype=np.float32).astype(np.float64)
    measured = hi_np + lo_np

    truth = values ** 2  # python/numpy float64 squaring: correctly
                           # rounded f64, adequate truth for a single op
    rel_err = np.abs(measured - truth) / np.maximum(np.abs(truth), 1e-300)
    max_rel_err = float(np.max(rel_err))
    clean = max_rel_err < DOUBLE_WORD_CLASS_BOUND

    extra = {
        "values_tested": values.tolist(),
        "truth": truth.tolist(),
        "hi": hi_np.tolist(), "lo": lo_np.tolist(),
        "measured": measured.tolist(),
        "rel_err": rel_err.tolist(),
        "max_rel_err": max_rel_err,
        "expected_scale_if_clean": "~2^-47 (double-word class)",
        "verdict": (
            "CLEAN: max rel_err {:.3e} is double-word class".format(max_rel_err)
            if clean else
            "F32-SCALE ERROR ({:.3e}): localizes the residual to the "
            "multiply-on-f64-sourced-values path".format(max_rel_err)
        ),
    }
    return _base_json("single_op", gpu, extra)


def stage_length_scan(args, gpu):
    import jax
    import jax.numpy as jnp
    import numpy as np

    common.assert_x64_enabled()
    val = args.val
    lengths = [10, 100, 1000, 10000]

    @jax.jit
    def fn_sumsq_internal(a):
        r = jnp.sum(a * a)
        return r, r  # doubled return -> DS_RETURN_PAIRS substitutes (hi, lo)

    @jax.jit
    def fn_sumsq_observable(a):
        return jnp.sum(a * a)  # single return -> NOT special-cased by
                                 # DS_RETURN_PAIRS even with the flag set,
                                 # see this file's module docstring

    rows = []
    for n in lengths:
        a_np = np.full(n, val, dtype=np.float64)
        a = jnp.asarray(a_np)
        truth = math.fsum(float(x) ** 2 for x in a_np)

        hi, lo = fn_sumsq_internal(a)
        common.block_until_ready((hi, lo))
        internal = float(np.float64(np.float32(hi)) + np.float64(np.float32(lo)))

        obs = float(common.block_until_ready(fn_sumsq_observable(a)))

        rows.append({
            "n": n,
            "truth": truth,
            "internal_result": internal,
            "internal_abs_err": abs(internal - truth),
            "internal_rel_err": abs(internal - truth) / abs(truth),
            "observable_result": obs,
            "observable_abs_err": abs(obs - truth),
            "observable_rel_err": abs(obs - truth) / abs(truth),
        })

    # Crude growth classification for the *internal* (arithmetic, not
    # output-quantization-limited) error series: flat vs sqrt(n)-like vs
    # only-appears-at-largest-n.
    internal_errs = [r["internal_abs_err"] for r in rows]
    if internal_errs[0] == 0.0:
        growth = "first point is exactly zero -- cannot classify growth rate from ratios"
    else:
        ratios = [internal_errs[i] / internal_errs[0] for i in range(len(internal_errs))]
        n_ratios = [lengths[i] / lengths[0] for i in range(len(lengths))]
        sqrt_expected = [r ** 0.5 for r in n_ratios]
        # Compare final observed ratio against flat (1x) and sqrt(n) (last
        # entry of sqrt_expected) to bucket into one of three qualitative
        # regimes -- not a statistical fit, just a coarse classifier to
        # report which hypothesis in the task's own framing it is closest to.
        final_ratio = ratios[-1]
        if final_ratio < 3.0:
            growth = "roughly FLAT from n=10 -- consistent with a per-element " \
                     "(split or multiply) residual, not accumulation"
        elif abs(final_ratio - sqrt_expected[-1]) < abs(final_ratio - n_ratios[-1]) / 2:
            growth = "roughly sqrt(n)-like -- consistent with ordinary " \
                     "accumulation-error growth, not a per-element defect"
        elif final_ratio >= n_ratios[-1] * 0.5:
            growth = "grows close to linearly with n, or jumps mainly at the " \
                     "largest n -- consistent with something size-dependent " \
                     "in the reduction lowering, not a per-element defect"
        else:
            growth = "grows faster than flat but slower than sqrt(n) by this " \
                     "coarse classifier -- see raw per-n rows for the actual shape"

    extra = {
        "val": val,
        "rows": rows,
        "growth_classification": growth,
    }
    return _base_json("length_scan", gpu, extra)


def stage_recombination(args, gpu):
    import jax
    import jax.numpy as jnp
    import numpy as np

    common.assert_x64_enabled()
    n, val = args.n, args.val

    @jax.jit
    def fn_sumsq_pairs(a):
        r = jnp.sum(a * a)
        return r, r

    a_np = np.full(n, val, dtype=np.float64)
    a = jnp.asarray(a_np)
    truth = math.fsum(float(x) ** 2 for x in a_np)

    hi, lo = fn_sumsq_pairs(a)
    common.block_until_ready((hi, lo))
    hi_f, lo_f = float(hi), float(lo)
    host_recombined = float(np.float64(hi_f) + np.float64(lo_f))
    error = abs(host_recombined - truth)

    suspicious = (hi_f == lo_f)
    if suspicious:
        print(
            "[exp5:recombination] WARNING: hi == lo ({}) -- DS_RETURN_PAIRS "
            "may not be active; expected two different raw DS components.".format(hi_f),
            file=sys.stderr,
        )

    extra = {
        "n": n, "val": val,
        "truth_f64_fsum": truth,
        "hi": hi_f, "lo": lo_f,
        "host_recombined_f64_result": host_recombined,
        "error_abs": error,
        "error_rel": error / abs(truth),
        "suspicious_hi_equals_lo": suspicious,
        "note": "Compare error_abs/error_rel here against "
                "corrected_apples_to_apples_test.ds_error_* from the "
                "ground_truth stage at the same n -- that figure is the "
                "pass's own normal (convert(h,f64)+convert(l,f64)) "
                "recombination; this one is the host recombining the raw "
                "pair directly. If they match closely, the pass's "
                "recombination is confirmed exact/clean by construction "
                "(both converts are exact widenings) and any residual "
                "predates recombination. If this (pairs) result is "
                "dramatically more accurate than the ground_truth stage's "
                "standard-path result, recombination/output quantization "
                "is where the gap opens up.",
    }
    return _base_json("recombination", gpu, extra)


STAGE_FUNCS = {
    "ground_truth": stage_ground_truth,
    "split_fidelity": stage_split_fidelity,
    "single_op": stage_single_op,
    "length_scan": stage_length_scan,
    "recombination": stage_recombination,
}


def main():
    args = parse_args()
    gpu = common.detect_gpu(required=True)

    result_json = STAGE_FUNCS[args.stage](args, gpu)

    out_path = args.out or str(
        common.default_result_path(gpu, "exp5_f64_reduction_bisection", args.stage))
    common.write_json_atomic(out_path, result_json)

    print("[exp5:{}] wrote {}".format(args.stage, out_path), file=sys.stderr)
    if "verdict" in result_json:
        print("[exp5:{}] {}".format(args.stage, result_json["verdict"]), file=sys.stderr)
    if "methodology_finding" in result_json:
        print("[exp5:{}] {}".format(args.stage, result_json["methodology_finding"]), file=sys.stderr)


if __name__ == "__main__":
    main()

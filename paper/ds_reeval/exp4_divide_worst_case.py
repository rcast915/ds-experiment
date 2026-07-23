#!/usr/bin/env python3
"""
Experiment 4 worker -- worst-case relative accuracy of DS divide, measured
post-fix (the `-t3` correction to emitDsDiv's dropped TwoProd residual --
see stablehlo_pass/DsTransformPass.cpp's emitDsDiv comment and the commit
that added the correction).

Feeds the paper's extended-operations table entry for divide: an expected
bound of ~2^-48 (same double-word class as add/sub/mul/sqrt), needing a
measured worst case now that the omitted residual term has been restored.
Pre-fix, an exhaustive host-side random search found a worst case of
1.16e-7 (f32-ULP class, confirmed by test_ds_divide.py's own commit
history); this experiment re-measures the post-fix worst case on-GPU, with
the same "cast the actual f32 input values up to f64" ground-truth
protocol used throughout this suite (see exp3_pair_accuracy.py).

Two modes, each its own process (DS_RETURN_PAIRS, like DS_BYPASS, is read
at plugin/pass load time and cannot be toggled mid-process):

  --mode internal   : DS_RETURN_PAIRS=1 must be set in the environment.
                       Host recombines (hi, lo) in f64 -- this is divide's
                       *arithmetic* accuracy, the number the paper's table
                       wants (it reports accuracy, not interface
                       quantization).
  --mode observable : ordinary jitted divide, single f32-typed return.
                       Included for comparison; expected to sit at
                       correctly-rounded-f32 level (~2^-23 to 2^-24)
                       regardless of internal accuracy for lo=0 inputs,
                       since an isolated correctly-rounded division's f32
                       output can't be improved on by definition -- see
                       tests/test_ds_divide.py Section 3a's note. Not the
                       number the paper wants, but worth recording so the
                       two don't get conflated.

This process measures four distinct things, all against the same f64
ground truth protocol:

  1. Random sweep: `--seeds` fixed seeds (default 3), `--samples-per-seed`
     samples each (default 100000) per seed. Numerator/denominator
     magnitude exponents drawn uniformly from [-15, 15] decades with
     random sign and mantissa -- comfortably inside both f32 range and the
     Veltkamp-split safe zone (see VELTKAMP_SAFE_LIMIT below), while still
     spanning 30 decades of quotient range. Per-seed AND combined
     max/median are reported so seed-to-seed stability is visible in the
     JSON, not just asserted.
  2. Adversarial cases, individually labeled: every fixed case already in
     tests/test_ds_divide.py's Section 1 (near-1 quotients, wide magnitude
     spread, mixed-decimal, pi/e), plus new cases targeting
     near-power-of-two quotients and explicit many-orders-of-magnitude
     spread.
  3. Exact-power-of-two divisors: dividing by 2^k is exactly representable
     at every step of the algorithm (initial divide, TwoProd, Sterbenz
     subtract, rescale -- see emitDsDiv), so expected rel_err is exactly
     0.0. A strong, cheap sanity check that a real regression would very
     likely violate.
  4. Beyond-safe-range probe: operands just above
     FLT_MAX / 4097 (~8.30e34, ~2^116) -- the split constant used by
     emitSplit's Veltkamp split (4097.0f = 2^12+1) -- past which `c =
     4097*a` can itself overflow f32 even though `a` is finite, corrupting
     the split silently rather than cleanly erroring. NOT included in the
     worst-case figure and NOT asserted against a bound -- just
     characterized (inf, nan, or a silently-wrong finite value), since the
     paper's Limitations section already states this restriction; this
     confirms whether the failure mode is where it's predicted to be.

Ground truth for every case: the actual f32 input values (not the
originating Python/numpy literals) cast up to f64 and divided in f64.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# FLT_MAX / 4097 -- see emitSplit's Veltkamp constant (4097.0f = 2^12+1) in
# DsTransformPass.cpp. Above this, `c = 4097*a` overflows f32 even though
# `a` itself is finite, corrupting the split silently rather than cleanly
# erroring. The main sweep and adversarial cases stay well below this;
# safe_range_probe deliberately crosses it.
VELTKAMP_SAFE_LIMIT = 3.4028235e38 / 4097.0  # ~8.303e34, ~2^116


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["internal", "observable"])
    p.add_argument("--seeds", default="20260716,1,42",
                    help="Comma-separated fixed seeds for the random sweep")
    p.add_argument("--samples-per-seed", type=int, default=100_000)
    p.add_argument("--out", default=None)
    return p.parse_args()


def make_random_sweep(rng, n):
    """n samples of (a, b) f32 pairs: exponents in [-15,15] decades,
    random sign and mantissa. Returns (a_f32, b_f32) numpy arrays."""
    import numpy as np
    exp_a = rng.uniform(-15, 15, size=n)
    exp_b = rng.uniform(-15, 15, size=n)
    mant_a = rng.uniform(1.0, 10.0, size=n)
    mant_b = rng.uniform(1.0, 10.0, size=n)
    sign_a = rng.choice([-1.0, 1.0], size=n)
    sign_b = rng.choice([-1.0, 1.0], size=n)
    a = (sign_a * mant_a * 10.0 ** exp_a).astype(np.float32)
    b = (sign_b * mant_b * 10.0 ** exp_b).astype(np.float32)
    # b_hi == 0 is deliberately exercised elsewhere (edge-semantics tests
    # in test_ds_divide.py) -- not something this worst-case sweep should
    # silently include as an extreme-outlier NaN from underflow.
    zero_mask = (b == 0.0)
    if zero_mask.any():
        b = np.where(zero_mask, np.float32(1.0), b)
    return a, b


def adversarial_cases():
    """(label, a, b) tuples -- fixed, not random. Includes every adversarial
    case in tests/test_ds_divide.py's Section 1, plus new cases targeting
    near-power-of-two quotients and extreme magnitude spread."""
    import numpy as np
    return [
        ("near_one_a", 1.0, 3.0),
        ("near_one_b", 7.0, 2.0),
        ("sign_mix_a", -5.0, 4.0),
        ("sign_mix_b", 5.0, -4.0),
        ("sign_mix_c", -5.0, -4.0),
        ("wide_spread_a", 1e15, 1e-15),
        ("wide_spread_b", 1e-15, 1e15),
        ("mixed_decimal", 123456.789, 0.00012345),
        ("pi_over_e", float(np.pi), float(np.e)),
        ("quotient_near_one", 1.0000001, 0.9999999),
        # Quotient near powers of two -- deliberately perturbed so the
        # numerator does NOT land exactly on 2^k (that would trivially
        # collapse to the power-of-two-divisor sanity check instead).
        ("quotient_near_2^-10", 1.0000003 * 2.0 ** -10, 1.0),
        ("quotient_near_2^0", 1.0000003, 1.0),
        ("quotient_near_2^10", 1.0000003 * 2.0 ** 10, 1.0),
        ("quotient_near_2^20", 1.0000003 * 2.0 ** 20, 1.0),
        # Operands differing by many orders of magnitude, each still well
        # inside the Veltkamp-safe zone individually.
        ("extreme_spread_a", 1e20, 1e-10),
        ("extreme_spread_b", 1e-20, 1e10),
    ]


def power_of_two_divisor_cases():
    """Numerators divided by exact powers of two. Every step of emitDsDiv
    (initial divide, TwoProd, Sterbenz subtract, rescale) is exact for a
    power-of-two divisor, so expected rel_err is exactly 0.0 -- a
    non-zero result here indicates a real regression, not noise."""
    numerators = [1.0, 3.0, 123456.789, -7.5, 1e10, 1e-10]
    exponents = [-20, -10, -1, 0, 1, 10, 20]
    return [(nv, 2.0 ** k) for nv in numerators for k in exponents]


def safe_range_probe_cases():
    """Just above VELTKAMP_SAFE_LIMIT -- characterize, do not assert."""
    return [
        ("just_above_limit_1.5x", VELTKAMP_SAFE_LIMIT * 1.5, 1.0),
        ("just_above_limit_2x", VELTKAMP_SAFE_LIMIT * 2.0, 1.0),
        ("just_above_limit_10x", VELTKAMP_SAFE_LIMIT * 10.0, 1.0),
        ("denominator_above_limit", 1.0, VELTKAMP_SAFE_LIMIT * 2.0),
    ]


def run_case_set(div_fn, a_vals, b_vals, mode):
    """a_vals, b_vals: numpy f32 arrays. Returns (measured_f64, truth_f64)
    numpy arrays, using the mode's evaluation path."""
    import numpy as np
    import jax.numpy as jnp

    a = jnp.asarray(a_vals)
    b = jnp.asarray(b_vals)

    if mode == "internal":
        hi, lo = div_fn(a, b)
        common.block_until_ready((hi, lo))
        hi_np = np.asarray(hi, dtype=np.float32).astype(np.float64)
        lo_np = np.asarray(lo, dtype=np.float32).astype(np.float64)
        measured = hi_np + lo_np
    else:
        r = div_fn(a, b)
        common.block_until_ready(r)
        measured = np.asarray(r, dtype=np.float32).astype(np.float64)

    truth = a_vals.astype(np.float64) / b_vals.astype(np.float64)
    return measured, truth


def rel_err_of(measured, truth):
    import numpy as np
    denom = np.maximum(np.abs(truth), 1e-300)
    return np.abs(measured - truth) / denom


def main():
    args = parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    gpu = common.detect_gpu(required=True)
    # f32-arithmetic experiment with f64 HOST recombination via plain numpy
    # -- does not need JAX-side x64, matching exp3's f32-input pattern
    # (see common.x64_status's docstring). Recorded for transparency only.
    x64_status = common.x64_status()

    if args.mode == "internal":
        @jax.jit
        def div_fn(a, b):
            r = a / b
            return r, r
    else:
        @jax.jit
        def div_fn(a, b):
            return a / b

    # ---- 1. random sweep ------------------------------------------------
    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = []
    all_rel_err = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        a_vals, b_vals = make_random_sweep(rng, args.samples_per_seed)
        measured, truth = run_case_set(div_fn, a_vals, b_vals, args.mode)
        errs = rel_err_of(measured, truth)
        finite = np.isfinite(errs)
        if not finite.all():
            print(
                "[exp4:{}] WARNING: seed={} produced {} non-finite rel_err "
                "values inside the nominally safe sweep range".format(
                    args.mode, seed, int((~finite).sum())),
                file=sys.stderr,
            )
        errs = errs[finite]
        per_seed.append({
            "seed": seed, "n": int(finite.sum()),
            "max_rel_err": float(np.max(errs)) if len(errs) else None,
            "median_rel_err": float(np.median(errs)) if len(errs) else None,
        })
        all_rel_err.append(errs)
    combined = np.concatenate(all_rel_err) if all_rel_err else np.array([])
    random_sweep = {
        "per_seed": per_seed,
        "total_samples": int(len(combined)),
        "combined_max_rel_err": float(np.max(combined)) if len(combined) else None,
        "combined_median_rel_err": float(np.median(combined)) if len(combined) else None,
        "exponent_decade_range": [-15, 15],
    }

    # ---- 2. adversarial cases -------------------------------------------
    adv = adversarial_cases()
    a_adv = np.array([c[1] for c in adv], dtype=np.float32)
    b_adv = np.array([c[2] for c in adv], dtype=np.float32)
    measured_adv, truth_adv = run_case_set(div_fn, a_adv, b_adv, args.mode)
    errs_adv = rel_err_of(measured_adv, truth_adv)
    adversarial = [
        {"label": c[0], "a": float(a_adv[i]), "b": float(b_adv[i]),
         "rel_err": float(errs_adv[i])}
        for i, c in enumerate(adv)
    ]

    # ---- 3. power-of-two divisor sanity check ---------------------------
    pow2 = power_of_two_divisor_cases()
    a_pow2 = np.array([c[0] for c in pow2], dtype=np.float32)
    b_pow2 = np.array([c[1] for c in pow2], dtype=np.float32)
    measured_pow2, truth_pow2 = run_case_set(div_fn, a_pow2, b_pow2, args.mode)
    errs_pow2 = rel_err_of(measured_pow2, truth_pow2)
    power_of_two = {
        "cases": [
            {"a": float(a_pow2[i]), "b": float(b_pow2[i]), "rel_err": float(errs_pow2[i])}
            for i in range(len(pow2))
        ],
        "all_exact": bool(np.all(errs_pow2 == 0.0)),
        "max_rel_err": float(np.max(errs_pow2)),
    }

    # ---- 4. beyond-safe-range probe (characterize, do not assert) -------
    probe = safe_range_probe_cases()
    a_probe = np.array([c[1] for c in probe], dtype=np.float32)
    b_probe = np.array([c[2] for c in probe], dtype=np.float32)
    measured_probe, truth_probe = run_case_set(div_fn, a_probe, b_probe, args.mode)
    safe_range_probe = {
        "veltkamp_safe_limit": VELTKAMP_SAFE_LIMIT,
        "cases": [
            {
                "label": probe[i][0],
                "a": float(a_probe[i]), "b": float(b_probe[i]),
                "a_over_limit": float(a_probe[i]) / VELTKAMP_SAFE_LIMIT,
                "truth": float(truth_probe[i]),
                "measured": float(measured_probe[i]),
                "measured_is_finite": bool(np.isfinite(measured_probe[i])),
                "rel_err_if_finite": (
                    float(rel_err_of(measured_probe[i:i + 1], truth_probe[i:i + 1])[0])
                    if np.isfinite(measured_probe[i]) else None
                ),
            }
            for i in range(len(probe))
        ],
    }

    result_json = {
        "experiment": "exp4_divide_worst_case",
        "mode": args.mode,
        "gpu": gpu,
        "env_relevant": common.snapshot_relevant_env(),
        "x64_status": x64_status,
        "git": common.git_commit_info(),
        "random_sweep": random_sweep,
        "adversarial_cases": adversarial,
        "power_of_two_divisors": power_of_two,
        "safe_range_probe": safe_range_probe,
        "jax_version": jax.__version__,
        "timestamp": common.utc_now_iso(),
    }

    out_path = args.out or str(common.default_result_path(gpu, "exp4_divide_worst_case", args.mode))
    common.write_json_atomic(out_path, result_json)

    worst_overall = max(
        random_sweep["combined_max_rel_err"] or 0.0,
        max((c["rel_err"] for c in adversarial), default=0.0),
    )
    print(
        "[exp4:{}] worst-case rel_err={:.6e} (random sweep max={:.6e} over {} samples, "
        "median={:.6e}; adversarial max={:.6e}; power-of-two all_exact={}); wrote {}".format(
            args.mode, worst_overall,
            random_sweep["combined_max_rel_err"] or float("nan"),
            random_sweep["total_samples"],
            random_sweep["combined_median_rel_err"] or float("nan"),
            max((c["rel_err"] for c in adversarial), default=float("nan")),
            power_of_two["all_exact"], out_path,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

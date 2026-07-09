#!/usr/bin/env python3
"""
Experiment 1 worker -- headline speedup under precision=HIGHEST.

Claim under test: "12.87x speedup over native f64 GEMM at 2048^2 on L40S"
(paper/main.tex, table tab:matmul_l40s / ~line 433). That number was
measured at default matmul precision; the paper's own accuracy validation
uses precision=HIGHEST (paper/main.tex ~line 470-472 says so explicitly).
This worker times A@B under all four combinations of {f64 baseline,
DS-f32} x {default, HIGHEST}, run one CLI invocation at a time so the
caller (run_all.py, or a human) can put each in its own subprocess with
the right environment -- DS_BYPASS is read at plugin-load time and cannot
be toggled mid-process.

Usage (env must already be set correctly by the caller -- see README.md):
    python3 exp1_gemm_highest.py --config ds_highest --precision highest \\
        --sizes 256,512,1024,2048 --warmup 5 --reps 30 --out result.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (stdlib-only import; safe regardless of env-var timing)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                    choices=["f64_default", "f64_highest", "ds_default", "ds_highest"],
                    help="Which env configuration this process was launched under. "
                         "Purely descriptive/for the output JSON -- this script does not "
                         "itself set DS_BYPASS or JAX_ENABLE_X64; the caller must.")
    p.add_argument("--precision", required=True, choices=["default", "highest"])
    p.add_argument("--sizes", default="256,512,1024,2048",
                    help="Comma-separated square GEMM sizes, e.g. 256,512,1024,2048")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Output JSON path (default: computed under results/)")
    return p.parse_args()


def main():
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    # jax is imported only now, after argument parsing -- env vars
    # (JAX_ENABLE_X64, PJRT_NAMES_AND_LIBRARY_PATHS, DS_BYPASS) must already
    # be set in this process's environment by whoever launched it.
    import jax
    import jax.numpy as jnp
    import numpy as np

    common.assert_x64_enabled()

    gpu = common.detect_gpu(required=True)
    precision = jax.lax.Precision.HIGHEST if args.precision == "highest" else jax.lax.Precision.DEFAULT

    @jax.jit
    def matmul(a, b):
        return jnp.dot(a, b, precision=precision)

    cases = []
    rng = np.random.default_rng(args.seed)
    for n in sizes:
        # Seeded random normal inputs. GEMM wall time for well-conditioned
        # random matrices does not depend on the data values (unlike
        # Experiment 3's accuracy test, which is data-sensitive), so this
        # is only about reproducibility of the timing run, not numerics.
        a_np = rng.standard_normal((n, n))
        b_np = rng.standard_normal((n, n))
        a = jnp.asarray(a_np, dtype=jnp.float64)
        b = jnp.asarray(b_np, dtype=jnp.float64)

        timing = common.timed_run(lambda a=a, b=b: matmul(a, b),
                                   warmup=args.warmup, reps=args.reps)
        tflops = common.gemm_tflops(n, timing["median_ms"] / 1000.0)
        cases.append({"n": n, "tflops": tflops, **timing})
        print(
            "[exp1:{}:{}] n={} median={:.4f}ms iqr={:.4f}ms tflops={:.3f} escalated={}".format(
                args.config, args.precision, n, timing["median_ms"], timing["iqr_ms"],
                tflops, timing["escalated"]),
            file=sys.stderr,
        )

    result = {
        "experiment": "exp1_gemm_highest",
        "config_name": args.config,
        "precision": args.precision,
        "gpu": gpu,
        "env_relevant": common.snapshot_relevant_env(),
        "seed": args.seed,
        "protocol": {
            "warmup": args.warmup,
            "reps": args.reps,
            "iqr_escalate_threshold": common.IQR_ESCALATE_THRESHOLD,
            "escalate_reps": common.ESCALATE_REPS,
        },
        "cases": cases,
        "jax_version": jax.__version__,
        "timestamp": common.utc_now_iso(),
    }

    out_path = args.out or str(common.default_result_path(gpu, "exp1_gemm_highest", args.config))
    common.write_json_atomic(out_path, result)
    print("[exp1:{}:{}] wrote {}".format(args.config, args.precision, out_path), file=sys.stderr)


if __name__ == "__main__":
    main()

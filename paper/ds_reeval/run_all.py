#!/usr/bin/env python3
"""
ds_reeval driver -- orchestrates one subprocess per (experiment x
configuration), each with the environment it needs, and collects the JSON
each worker writes.

Env vars (DS_BYPASS, DS_RETURN_PAIRS, JAX_ENABLE_X64, XLA_FLAGS, ...) are
read at plugin/XLA/pass load time, so configurations are never toggled
inside a running process -- every row in the plans below is its own
subprocess.

    python3 run_all.py --dry-run                  # print the plan, touch nothing
    python3 run_all.py --smoke                     # fast reduced-size correctness pass
    python3 run_all.py --experiments 1              # just Experiment 1
    python3 run_all.py --plugin-path /path/to/libds_pjrt_plugin.so \\
        --image-digest sha256:...                   # full real run

See README.md for exact per-cluster commands including SLURM/container setup.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PLUGIN_PATH = "/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so"

# (config_name, env_overrides, precision) -- PJRT_NAMES_AND_LIBRARY_PATHS and
# JAX_ENABLE_X64 defaults are added uniformly in build_env(); overrides here
# only need to state what's different about *this* config.
EXP1_CONFIGS = [
    ("f64_default", {"DS_BYPASS": "1"}, "default"),
    ("f64_highest", {"DS_BYPASS": "1"}, "highest"),
    ("ds_default", {}, "default"),
    ("ds_highest", {}, "highest"),
]

EXP2_MODES = ["default", "highest"]

EXP3_MODES = ["standard", "pairs"]

EXP4_MODES = ["internal", "observable"]

EXP5_STAGES = ["ground_truth", "split_fidelity", "single_op", "length_scan", "recombination"]


def build_env(plugin_path, overrides):
    env = {"PJRT_NAMES_AND_LIBRARY_PATHS": "cuda:{}".format(plugin_path),
           "JAX_ENABLE_X64": "1"}
    env.update(overrides)
    return env


def plan_exp1(args, results_dir):
    plans = []
    sizes_list = ["256"] if args.smoke else args.sizes.split(",")
    if args.include_4096 and not args.smoke and "4096" not in sizes_list:
        sizes_list = sizes_list + ["4096"]
    warmup = 1 if args.smoke else args.warmup
    reps = 5 if args.smoke else args.reps

    for name, overrides, precision in EXP1_CONFIGS:
        out = results_dir / "exp1_gemm_highest_{}.json".format(name)
        env = build_env(args.plugin_path, overrides)
        argv = [
            sys.executable, str(THIS_DIR / "exp1_gemm_highest.py"),
            "--config", name, "--precision", precision,
            "--sizes", ",".join(sizes_list),
            "--warmup", str(warmup), "--reps", str(reps),
            "--out", str(out),
        ]
        plans.append(("exp1 config={} precision={}".format(name, precision), argv, env, out))
    return plans


def plan_exp2(args, results_dir):
    plans = []
    n = 256 if args.smoke else 2048
    for mode in EXP2_MODES:
        out = results_dir / "exp2_tf32_dispatch_{}.json".format(mode)
        dump_dir = results_dir / "xla_dumps" / mode
        env = build_env(args.plugin_path, {})
        argv = [
            sys.executable, str(THIS_DIR / "exp2_tf32_dispatch.py"),
            "--precision", mode, "--n", str(n), "--out", str(out),
            "--dump-dir", str(dump_dir), "--nsys", args.nsys,
        ]
        plans.append(("exp2 precision={}".format(mode), argv, env, out))
    return plans


def plan_exp3(args, results_dir):
    plans = []
    n = 1000 if args.smoke else 10000
    for mode in EXP3_MODES:
        out = results_dir / "exp3_pair_accuracy_{}.json".format(mode)
        overrides = {"DS_RETURN_PAIRS": "1"} if mode == "pairs" else {}
        env = build_env(args.plugin_path, overrides)
        argv = [
            sys.executable, str(THIS_DIR / "exp3_pair_accuracy.py"),
            "--mode", mode, "--n", str(n), "--val", "0.1", "--out", str(out),
        ]
        plans.append(("exp3 mode={}".format(mode), argv, env, out))
    return plans


def plan_exp4(args, results_dir):
    plans = []
    samples = 5000 if args.smoke else args.divide_samples_per_seed
    for mode in EXP4_MODES:
        out = results_dir / "exp4_divide_worst_case_{}.json".format(mode)
        overrides = {"DS_RETURN_PAIRS": "1"} if mode == "internal" else {}
        env = build_env(args.plugin_path, overrides)
        argv = [
            sys.executable, str(THIS_DIR / "exp4_divide_worst_case.py"),
            "--mode", mode, "--samples-per-seed", str(samples),
            "--out", str(out),
        ]
        plans.append(("exp4 mode={}".format(mode), argv, env, out))
    return plans


def plan_exp5(args, results_dir):
    plans = []
    n = 1000 if args.smoke else 10000
    for stage in EXP5_STAGES:
        out = results_dir / "exp5_f64_reduction_bisection_{}.json".format(stage)
        # ground_truth doesn't need DS_RETURN_PAIRS; every other stage does
        # (see exp5_f64_reduction_bisection.py's module docstring).
        overrides = {} if stage == "ground_truth" else {"DS_RETURN_PAIRS": "1"}
        env = build_env(args.plugin_path, overrides)
        argv = [
            sys.executable, str(THIS_DIR / "exp5_f64_reduction_bisection.py"),
            "--stage", stage, "--n", str(n), "--val", "0.1",
            "--out", str(out),
        ]
        plans.append(("exp5 stage={}".format(stage), argv, env, out))
    return plans


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", default="1,2,3,4,5",
                    help="Comma-separated subset of {1,2,3,4,5} to run (default: all)")
    p.add_argument("--divide-samples-per-seed", type=int, default=100_000,
                    help="Experiment 4: random-sweep samples per seed (full run only; "
                         "--smoke always uses a small fixed count)")
    p.add_argument("--smoke", action="store_true",
                    help="Reduced sizes/reps for a fast correctness pass. Writes to "
                         "smoke_out/ instead of results/, so it can never be mistaken "
                         "for a real result.")
    p.add_argument("--dry-run", action="store_true",
                    help="Print the planned subprocess commands and environments; "
                         "execute nothing, write nothing.")
    p.add_argument("--plugin-path", default=DEFAULT_PLUGIN_PATH,
                    help="Path to libds_pjrt_plugin.so (default matches the documented "
                         "in-container build path)")
    p.add_argument("--results-dir", default=None,
                    help="Override the results root (default: ds_reeval/results, or "
                         "ds_reeval/smoke_out under --smoke)")
    p.add_argument("--sizes", default="256,512,1024,2048",
                    help="Comma-separated GEMM sizes for Experiment 1")
    p.add_argument("--include-4096", action="store_true",
                    help="Append 4096 to the Experiment 1 size sweep (memory permitting)")
    p.add_argument("--image-digest", default=None,
                    help="Container image digest to record in manifest.json -- obtain with "
                         "`docker inspect --format='{{index .RepoDigests 0}}' ds-experiment` "
                         "on the host before entering the container. Falls back to "
                         "$DS_IMAGE_DIGEST, then 'unknown'.")
    p.add_argument("--nsys", choices=["auto", "on", "off"], default="auto",
                    help="Experiment 2 profiling: auto-detect nsys (default), require it "
                         "(on), or skip it and rely on the HLO dump alone (off)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    exps = set(x.strip() for x in args.experiments.split(","))
    unknown = exps - {"1", "2", "3", "4", "5"}
    if unknown:
        print("Unknown experiment id(s): {}".format(sorted(unknown)), file=sys.stderr)
        return 2

    gpu = common.detect_gpu(required=not args.dry_run)
    gpu_tag = common.sanitize_gpu_tag(gpu["name"])

    if args.results_dir:
        results_root = Path(args.results_dir)
    else:
        results_root = THIS_DIR / ("smoke_out" if args.smoke else "results")
    results_dir = results_root / gpu_tag

    plans = []
    if "1" in exps:
        plans += plan_exp1(args, results_dir)
    if "2" in exps:
        plans += plan_exp2(args, results_dir)
    if "3" in exps:
        plans += plan_exp3(args, results_dir)
    if "4" in exps:
        plans += plan_exp4(args, results_dir)
    if "5" in exps:
        plans += plan_exp5(args, results_dir)

    if args.dry_run:
        print("# ds_reeval plan -- {} subprocess run(s), GPU tag: {}".format(len(plans), gpu_tag))
        print("# (dry run: nothing will be executed, no files will be written)\n")
        for desc, argv, env, out_path in plans:
            print("## {}".format(desc))
            print("  out = {}".format(out_path))
            print("  env = {}".format(env))
            print("  cmd = {}".format(" ".join(argv)))
            print()
        return 0

    print("==> Writing manifest for GPU '{}' to {}".format(gpu["name"], results_dir / "manifest.json"))
    common.write_json_atomic(results_dir / "manifest.json", common.build_manifest(args, gpu))

    failures = []
    for desc, argv, env, out_path in plans:
        print("==> {}".format(desc))
        full_env = dict(os.environ)
        full_env.update(env)
        proc = subprocess.run(argv, env=full_env)
        if proc.returncode != 0:
            print("    FAILED (exit {}) -- see stderr above; continuing with remaining runs.".format(
                proc.returncode))
            failures.append(desc)
        else:
            print("    ok -> {}".format(out_path))

    if failures:
        print("\n{} of {} run(s) failed:".format(len(failures), len(plans)))
        for f in failures:
            print("  - {}".format(f))
        print("\nRe-run report.py to see what did complete -- it degrades gracefully "
              "over whatever JSON is actually present.")
        return 1

    print("\nAll {} run(s) completed. Next: python3 report.py".format(len(plans)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Experiment 2 worker -- TF32 dispatch verification.

Claim under test: the paper's *inference*, from arithmetic alone, that
default-precision DS sub-matmuls dispatch to TF32 tensor cores on H100:
54.6 effective TFLOPS (denominated against 2*N^3) implies ~218 TFLOPS of
actual f32 arithmetic across the four sub-matmuls, which exceeds H100's
non-tensor-core f32 peak (~67 TFLOPS) (paper/main.tex ~line 462-466). This
worker turns that inference into direct evidence: what kernel actually ran.

Runs the 2048^2 DS matmul once (not a timed loop -- this experiment is
about kernel *identity*, not speed) with XLA's HLO-dump flags set, then:
  1. Greps the dumped, optimized HLO text for the dot/custom-call the four
     sub-GEMMs lowered to (tolerant, grep-style -- HLO dump file naming and
     internal formatting vary across XLA versions, so this does not attempt
     a brittle full parse).
  2. If `nsys` is on PATH, additionally profiles one iteration and
     classifies the GPU kernel names as TF32-family, plain-f32 SGEMM, or
     other. If nsys is unavailable, records that plainly and relies on the
     HLO dump alone.

Each precision mode is its own process (same env-inheritance reasoning as
exp1: XLA_FLAGS must be set before jax/XLA initializes).
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", required=True, choices=["default", "highest"])
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--out", default=None)
    p.add_argument("--dump-dir", required=True, help="Directory for XLA HLO text dump + nsys report")
    p.add_argument("--nsys", choices=["auto", "on", "off"], default="auto",
                    help="auto: use nsys if found on PATH; on: require it and fail loudly if "
                         "missing; off: skip profiling, HLO dump evidence only")
    p.add_argument("--exec-only", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# Grep-style keyword match over dumped HLO text -- intentionally broad and
# line-oriented rather than a structural HLO parse, per the instruction to
# keep this tolerant of format drift across XLA versions.
HLO_KEYWORD_RE = re.compile(
    r"(cublas|custom-call|dot\(|gemm|tf32|algorithm|precision)", re.IGNORECASE
)
TF32_KERNEL_RE = re.compile(r"(tf32|s1688|h1688|xmma\S*tf32)", re.IGNORECASE)
F32_KERNEL_RE = re.compile(r"(sgemm|s884|ssyrk)", re.IGNORECASE)
KERNEL_LINE_RE = re.compile(r"(gemm|xmma|s1688|s884|volta|ampere|hopper|ada|cutlass)", re.IGNORECASE)

# XLA embeds Python source-frame/backtrace annotations in its dumps, shaped
# like `<int> "<path>.py"`, pointing back at whatever script traced the
# jit'd computation -- including, self-referentially, this test harness
# itself (exp2_tf32_dispatch.py, whose own filename contains "tf32" by
# design). Confirmed via a real false positive: the only two "tf32" hits in
# an actual run were both this exact line, quoting this script's own path,
# not any kernel or dispatch evidence. Exclude lines with this shape from
# the verdict-determining text match so the detector can't trigger on its
# own filename -- they stay in the raw hlo_evidence list for transparency.
SOURCE_FRAME_ECHO_RE = re.compile(r'^\d+\s+"[^"]*\.py"\s*$')


def run_single_matmul(n, precision_mode, dump_dir):
    """Sets XLA HLO-dump flags, then imports jax and runs one DS matmul.

    Returns the jax version string (also confirms jax imported successfully).
    Must be called before any other jax import happens in this process.
    """
    os.makedirs(dump_dir, exist_ok=True)
    existing = os.environ.get("XLA_FLAGS", "")
    dump_flags = "--xla_dump_to={} --xla_dump_hlo_as_text".format(dump_dir)
    os.environ["XLA_FLAGS"] = (existing + " " + dump_flags).strip()

    import jax
    import jax.numpy as jnp
    import numpy as np

    common.assert_x64_enabled()

    precision = jax.lax.Precision.HIGHEST if precision_mode == "highest" else jax.lax.Precision.DEFAULT

    @jax.jit
    def fn(a, b):
        return jnp.dot(a, b, precision=precision)

    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((n, n)), dtype=jnp.float64)
    b = jnp.asarray(rng.standard_normal((n, n)), dtype=jnp.float64)
    out = fn(a, b)
    common.block_until_ready(out)
    return jax.__version__


def find_optimized_hlo_files(dump_dir):
    # XLA's dump file naming has changed across releases (e.g.
    # "*after_optimizations.txt" vs "*after-optimizations.txt"); try the
    # specific patterns first and fall back to "everything" rather than
    # hard-coding one and silently finding nothing on a different version.
    patterns = ["*after_optimizations*.txt", "*after-optimizations*.txt", "*.txt"]
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(dump_dir, pat)))
        if matches:
            return matches
    return []


def grep_hlo_evidence(files):
    matched_lines = []
    for path in files:
        try:
            with open(path, "r", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if HLO_KEYWORD_RE.search(line):
                        matched_lines.append({
                            "file": os.path.basename(path),
                            "line": lineno,
                            "text": line.strip(),
                        })
        except OSError:
            continue
    return matched_lines


def classify_kernel_name(name):
    if TF32_KERNEL_RE.search(name):
        return "tf32"
    if F32_KERNEL_RE.search(name):
        return "f32"
    return "other"


def _parse_nsys_kernel_summary(text):
    # Tolerant, grep-style: pull lines that look like a CUDA GEMM kernel
    # name rather than fully parsing nsys's column layout, which varies by
    # nsys version and report type.
    names = []
    for line in text.splitlines():
        if KERNEL_LINE_RE.search(line):
            names.append(line.strip())
    return names


def try_nsys_profile(script_path, n, precision_mode, dump_dir, nsys_mode):
    if nsys_mode == "off":
        return {"attempted": False, "available": None, "reason": "disabled by --nsys off"}
    nsys_bin = shutil.which("nsys")
    if nsys_bin is None:
        reason = "nsys not found on PATH"
        if nsys_mode == "on":
            return {"attempted": True, "available": False, "reason": reason + " (was required by --nsys on)"}
        return {"attempted": False, "available": False, "reason": reason + " (auto mode, skipping)"}

    report_base = os.path.join(dump_dir, "nsys_{}".format(precision_mode))
    # Re-invoke this same script with --exec-only so nsys profiles exactly
    # one clean matmul execution, nothing else.
    cmd = [
        nsys_bin, "profile", "--stats=true", "--force-overwrite=true",
        "-o", report_base,
        sys.executable, script_path,
        "--precision", precision_mode, "--n", str(n),
        "--dump-dir", dump_dir, "--nsys", "off", "--exec-only",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        return {"attempted": True, "available": True, "reason": "nsys invocation failed: {}".format(e)}

    if proc.returncode != 0:
        return {
            "attempted": True, "available": True,
            "reason": "nsys exited with code {}".format(proc.returncode),
            "stderr_tail": proc.stderr[-2000:],
        }

    kernels = _parse_nsys_kernel_summary(proc.stdout)
    if not kernels:
        for ext in (".txt", ".csv"):
            cand = report_base + ext
            if os.path.exists(cand):
                with open(cand, "r", errors="replace") as f:
                    kernels = _parse_nsys_kernel_summary(f.read())
                if kernels:
                    break

    classified = [{"name": k, "class": classify_kernel_name(k)} for k in kernels]
    return {
        "attempted": True,
        "available": True,
        "reason": None,
        "report_file": report_base + ".nsys-rep",
        "kernels": classified,
        "stdout_tail": proc.stdout[-4000:],
    }


def main():
    args = parse_args()

    if args.exec_only:
        # Internal path: this process is the child spawned under `nsys
        # profile` by try_nsys_profile(). Just run the matmul and exit --
        # no HLO grep, no further nsys spawning, no JSON output here.
        run_single_matmul(args.n, args.precision, args.dump_dir)
        return 0

    jax_version = run_single_matmul(args.n, args.precision, args.dump_dir)

    gpu = common.detect_gpu(required=True)
    hlo_files = find_optimized_hlo_files(args.dump_dir)
    hlo_evidence = grep_hlo_evidence(hlo_files)
    nsys_result = try_nsys_profile(str(Path(__file__).resolve()), args.n, args.precision,
                                    args.dump_dir, args.nsys)

    verdict = None
    evidence_quality = "none"
    if nsys_result.get("kernels"):
        classes = {k["class"] for k in nsys_result["kernels"]}
        if "tf32" in classes and "f32" not in classes:
            verdict, evidence_quality = True, "nsys_kernel_names"
        elif "f32" in classes and "tf32" not in classes:
            verdict, evidence_quality = False, "nsys_kernel_names"
        else:
            evidence_quality = "nsys_kernel_names_ambiguous"
    if verdict is None and hlo_evidence:
        kernel_lines = [e["text"] for e in hlo_evidence
                        if not SOURCE_FRAME_ECHO_RE.match(e["text"].strip())]
        joined = "\n".join(kernel_lines).lower()
        if "tf32" in joined:
            verdict, evidence_quality = True, "hlo_dump_text_match"
        elif evidence_quality == "none":
            evidence_quality = "hlo_dump_present_but_inconclusive"

    result = {
        "experiment": "exp2_tf32_dispatch",
        "precision": args.precision,
        "n": args.n,
        "gpu": gpu,
        "env_relevant": common.snapshot_relevant_env(),
        "dump_dir": args.dump_dir,
        "hlo_files_found": [os.path.basename(f) for f in hlo_files],
        "hlo_evidence": hlo_evidence,
        "nsys": nsys_result,
        "verdict_tf32_dispatch": verdict,
        "evidence_quality": evidence_quality,
        "jax_version": jax_version,
        "timestamp": common.utc_now_iso(),
    }

    out_path = args.out or str(common.default_result_path(gpu, "exp2_tf32_dispatch", args.precision))
    common.write_json_atomic(out_path, result)
    print(
        "[exp2:{}] verdict_tf32_dispatch={} (quality={}); wrote {}".format(
            args.precision, verdict, evidence_quality, out_path),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

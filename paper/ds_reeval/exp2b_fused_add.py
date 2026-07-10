#!/usr/bin/env python3
"""
Experiment 2b worker -- did the TwoSum correction survive XLA's optimizer,
or did the algebraic simplifier collapse it into a plain add?

Background: the DS pass's dot_general handler emits a final
(out_hi, out_lo) = TwoSum(p, e1+e2+e3) correction, then recombines via
cast(hi, target) + cast(lo, target). TwoSum's residual computation
(v = s - a; e = (a - (s - v)) + (b - v)) is algebraically zero under real
arithmetic, so a simplifier that doesn't respect IEEE float semantics
could legally-looking-but-incorrectly fold it away, silently discarding
the DS correction on that path. Experiment 2's dumps only grep for a
narrow keyword set (cublas/gemm/precision/etc.) and would not have
noticed this either way -- this script specifically extracts the epilogue
computation body (the one combining the GEMM/fusion outputs into the
final result) and classifies it by structure:

  INTACT       -- the subtract-based residual chain is present (>=4
                  subtracts is the signature of >=1 TwoSum call; see
                  emitTwoSum in DsTransformPass.cpp: 2 adds + 4 subtracts).
  COLLAPSED    -- zero subtracts; body is just add(convert(x), convert(y))
                  or similar -- the correction is gone.
  PARTIAL/OTHER -- something in between; inspect the body.

Also locates the earliest available (pre-optimization) dump of the same
module and counts subtracts there, to distinguish "the simplifier removed
it" (present pre-opt, gone post-opt) from "the pass never emitted it on
this path" (absent pre-opt too) -- these are different bugs.

Measurement only. Does not modify the pass, does not time anything.
Reuses an existing Experiment 2 dump directory
(results/<gpu>/xla_dumps/<precision>/) if one is already there from a
prior run; regenerates by re-running the same 2048^2 DS matmul workload
otherwise (via exp2_tf32_dispatch.run_single_matmul, same env-var
discipline as the rest of the suite: XLA_FLAGS/JAX_ENABLE_X64/
PJRT_NAMES_AND_LIBRARY_PATHS must already be set by the caller before
this process imports jax).

Usage:
    python3 exp2b_fused_add.py --precision default
    python3 exp2b_fused_add.py --precision highest
(dump-dir and out default to the same GPU-tagged results/ path
Experiment 2 already uses/wrote, auto-detected via nvidia-smi/procfs)
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import exp2_tf32_dispatch as exp2  # noqa: E402  (reuses run_single_matmul; safe, no jax at module level)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", required=True, choices=["default", "highest"])
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--dump-dir", default=None,
                    help="Default: results/<gpu-tag>/xla_dumps/<precision> -- same path "
                         "Experiment 2 uses, so its dumps are reused automatically if present.")
    p.add_argument("--out", default=None,
                    help="Default: results/<gpu-tag>/exp2b_fused_add_<precision>.json")
    return p.parse_args()


def dump_has_content(dump_dir):
    return os.path.isdir(dump_dir) and bool(glob.glob(os.path.join(dump_dir, "*.txt")))


def _looks_like_hlo_module(text):
    return bool(re.search(r'^\s*ENTRY\s+%', text, re.MULTILINE))


def _read_if_hlo_module(path):
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    return text if _looks_like_hlo_module(text) else None


def find_after_optimizations_file(dump_dir):
    # Match the filename *suffix* exactly ("...after_optimizations.txt",
    # nothing after it) rather than a substring -- XLA emits auxiliary
    # reports alongside the real dump whose names also contain
    # "after_optimizations" as a substring but aren't HLO text at all
    # (buffer-assignment, live-range, memory-usage-report, and there is no
    # guarantee that list is complete across XLA versions -- e.g.
    # "-memory-usage-report.txt" wasn't anticipated and broke an exact
    # exclusion-list approach here). Content-verify too (must have an
    # ENTRY block) rather than trusting the name alone.
    candidates = sorted(
        p for p in glob.glob(os.path.join(dump_dir, "*jit_fn*after_optimizations*.txt"))
        if p.endswith("after_optimizations.txt")
    )
    for c in candidates:
        if _read_if_hlo_module(c) is not None:
            return c
    return candidates[0] if candidates else None


def find_pre_optimization_file(dump_dir, after_file):
    candidates = sorted(
        p for p in glob.glob(os.path.join(dump_dir, "*jit_fn*before_optimizations*.txt"))
        if p.endswith("before_optimizations.txt")
    )
    for c in candidates:
        if _read_if_hlo_module(c) is not None:
            return c, "matched *before_optimizations.txt (exact suffix, content-verified)"
    # Fallback: any other jit_fn module dump, content-verified as an actual
    # HLO module (not an auxiliary report), preferring the lowest-numbered
    # (earliest) one that isn't the after-optimizations file itself.
    all_jit_fn = sorted(
        p for p in glob.glob(os.path.join(dump_dir, "module_*jit_fn*.txt"))
        if p != after_file
    )
    for c in all_jit_fn:
        if _read_if_hlo_module(c) is not None:
            return c, "fallback: lowest-numbered jit_fn module dump (content-verified) other than after_optimizations"
    return None, "no HLO-module-shaped candidate found"


def _extract_block(text, start_line_re):
    """Extract a `name (...) -> type { ... }`-shaped block: from the first
    line matching start_line_re through the next line that is a bare `}`
    at column 0 (XLA's standard computation-block indentation)."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if start_line_re.match(line):
            start = i
            break
    if start is None:
        return None
    body_lines = [lines[start]]
    for line in lines[start + 1:]:
        body_lines.append(line)
        if line.strip() == "}" and not line.startswith((" ", "\t")):
            break
    return "\n".join(body_lines)


def extract_entry_computation(text):
    return _extract_block(text, re.compile(r'^\s*ENTRY\s+%'))


def extract_named_computation(text, comp_name):
    return _extract_block(text, re.compile(r'^\s*%{}\s*\('.format(re.escape(comp_name))))


def extract_root_and_calls(entry_text):
    m = re.search(r'^\s*ROOT\s+%\S+\s*=.*$', entry_text, re.MULTILINE)
    if not m:
        return None, None
    root_line = m.group(0)
    calls_m = re.search(r'calls=%(\S+)', root_line)
    return root_line, (calls_m.group(1) if calls_m else None)


GEMM_PRODUCER_RE = re.compile(
    r'custom_call_target="__cublas|"kind":"__triton_nested_gemm_fusion"|=\s*\S+\s+dot\('
)


def extract_inline_epilogue(entry_text, root_line):
    """Fallback when ROOT has no calls=%X: take the entry computation's own
    instruction lines from the last GEMM-producing line through ROOT
    (textual/definition-order slice, not full dataflow tracing -- a
    tolerant approximation consistent with this suite's grep-style
    philosophy elsewhere)."""
    lines = entry_text.splitlines()
    root_idx = next((i for i, l in enumerate(lines) if l.strip() == root_line.strip()), None)
    if root_idx is None:
        return None
    last_gemm_idx = next(
        (i for i in range(root_idx, -1, -1) if GEMM_PRODUCER_RE.search(lines[i])), None
    )
    if last_gemm_idx is None:
        return "\n".join(lines[max(0, root_idx - 30):root_idx + 1])
    return "\n".join(lines[last_gemm_idx:root_idx + 1])


def count_ops(text):
    return {
        "adds": len(re.findall(r'=\s*\S+\s+add\(', text)),
        "subtracts": len(re.findall(r'=\s*\S+\s+subtract\(', text)),
        "converts": len(re.findall(r'=\s*\S+\s+convert\(', text)),
        "slices_reshapes_bitcasts": len(re.findall(r'=\s*\S+\s+(?:slice|reshape|bitcast)\(', text)),
    }


def classify(counts):
    if counts["subtracts"] >= 4 and counts["converts"] >= 1:
        return "INTACT"
    if counts["subtracts"] == 0 and counts["converts"] >= 1:
        return "COLLAPSED"
    return "PARTIAL/OTHER"


def main():
    args = parse_args()

    gpu = common.detect_gpu(required=True)
    gpu_tag = common.sanitize_gpu_tag(gpu["name"])
    dump_dir = args.dump_dir or str(common.RESULTS_ROOT / gpu_tag / "xla_dumps" / args.precision)
    out_path = args.out or str(common.RESULTS_ROOT / gpu_tag / "exp2b_fused_add_{}.json".format(args.precision))

    reused = dump_has_content(dump_dir)
    if not reused:
        print("[exp2b:{}] no existing dump at {}, regenerating (running the matmul once, "
              "not timed)...".format(args.precision, dump_dir), file=sys.stderr)
        exp2.run_single_matmul(args.n, args.precision, dump_dir)
    else:
        print("[exp2b:{}] reusing existing dump at {}".format(args.precision, dump_dir),
              file=sys.stderr)

    result = {
        "experiment": "exp2b_fused_add",
        "precision": args.precision,
        "n": args.n,
        "gpu": gpu,
        "dump_dir": dump_dir,
        "reused_existing_dump": reused,
        "timestamp": common.utc_now_iso(),
    }

    after_file = find_after_optimizations_file(dump_dir)
    if not after_file:
        result["error"] = "no *jit_fn*after_optimizations*.txt found in {}".format(dump_dir)
        result["verdict"] = "PARTIAL/OTHER (no dump found)"
        common.write_json_atomic(out_path, result)
        print("[exp2b:{}] ERROR: {}".format(args.precision, result["error"]), file=sys.stderr)
        return 1

    after_text = Path(after_file).read_text(errors="replace")
    entry_text = extract_entry_computation(after_text)
    if entry_text is None:
        result["error"] = "could not find ENTRY computation in {}".format(after_file)
        result["verdict"] = "PARTIAL/OTHER (no ENTRY found)"
        common.write_json_atomic(out_path, result)
        print("[exp2b:{}] ERROR: {}".format(args.precision, result["error"]), file=sys.stderr)
        return 1

    root_line, called_comp = extract_root_and_calls(entry_text)

    body_text, body_source = None, None
    if called_comp:
        body_text = extract_named_computation(after_text, called_comp)
        body_source = "named computation %{}".format(called_comp)
    if not body_text and root_line:
        body_text = extract_inline_epilogue(entry_text, root_line)
        body_source = "inline epilogue (textual slice from last GEMM producer to ROOT)"

    counts = count_ops(body_text) if body_text else {"adds": 0, "subtracts": 0, "converts": 0,
                                                       "slices_reshapes_bitcasts": 0}
    verdict = classify(counts) if body_text else "PARTIAL/OTHER (no body extracted)"

    pre_file, pre_method = find_pre_optimization_file(dump_dir, after_file)
    pre_check = None
    if pre_file:
        pre_text = Path(pre_file).read_text(errors="replace")
        pre_counts = count_ops(pre_text)
        pre_check = {
            "file": pre_file,
            "method": pre_method,
            "subtract_count_in_whole_module": pre_counts["subtracts"],
            "note": "counts the *whole* pre-optimization module, not just the epilogue "
                    "specifically (naming/structure differs before fusion) -- confirms TwoSum "
                    "ops exist *somewhere* in the unoptimized program, distinguishing "
                    "'simplifier removed it' from 'pass never emitted it here'.",
        }

    result.update({
        "after_optimizations_file": after_file,
        "root_line": root_line,
        "called_epilogue_computation": called_comp,
        "body_source": body_source,
        "body_text": body_text,
        "op_counts": counts,
        "verdict": verdict,
        "pre_optimization_check": pre_check,
    })

    common.write_json_atomic(out_path, result)
    print(
        "[exp2b:{}] verdict={} (subtracts={} adds={} converts={}); wrote {}".format(
            args.precision, verdict, counts["subtracts"], counts["adds"], counts["converts"], out_path),
        file=sys.stderr,
    )
    if verdict.startswith("COLLAPSED"):
        print("[exp2b:{}] *** COLLAPSED -- DS correction may be silently discarded on this "
              "path ***".format(args.precision), file=sys.stderr)
    print("--- {} epilogue body ({}) ---".format(args.precision, body_source), file=sys.stderr)
    print(body_text, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

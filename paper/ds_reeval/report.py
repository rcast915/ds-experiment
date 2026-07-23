#!/usr/bin/env python3
"""
Assembles every ds_reeval/results/<gpu>/*.json into ds_reeval/REPORT.md:
per claim, the paper's current published number, the re-measured number,
the config difference, and a PASS/FAIL/CHANGED (or CONFIRMED/REFUTED/
INCONCLUSIVE for Experiment 3) verdict, ending in a summary table.

Degrades gracefully: any missing JSON is reported as "not yet run" rather
than raising, so this can be pointed at a partial results/ tree (e.g. only
one GPU done, or only one experiment finished) and still produce something
useful. This script only *reads* results/*.json and writes REPORT.md -- it
never launches a benchmark itself.

    python3 report.py                       # results/ -> REPORT.md
    python3 report.py --results-dir smoke_out --out SMOKE_REPORT.md
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = THIS_DIR / "results"
REPORT_PATH = THIS_DIR / "REPORT.md"

# Reference values transcribed from paper/main.tex as of the commit noted in
# each GPU's manifest.json at re-measurement time. If the paper changes,
# these will drift -- report.py flags itself as possibly stale by printing
# the line numbers so a human can diff against the current draft.
PAPER_L40S_DEFAULT_MS = {256: 0.07, 512: 0.08, 1024: 0.16, 2048: 0.95}   # tab:matmul_l40s, DS col
PAPER_H100_DEFAULT_MS = {256: 0.05, 512: 0.08, 1024: 0.11, 2048: 0.31}   # tab:matmul_h100, DS col
PAPER_HEADLINE_SPEEDUP_L40S_2048 = 12.87   # paper/main.tex ~line 433, 449
PAPER_EXP3_F32_ERROR = 4.65e-6             # paper/main.tex ~line 597
PAPER_EXP3_PLAIN_F32_ERROR = {"H100": 7.16e-5, "L40S": 7.93e-5}  # ~line 596
PAPER_EXP3_N = 10000                       # paper/main.tex ~line 587 -- only comparable at this n
PAPER_F64_REDUCTION_DS_ERROR = 1.49e-5     # paper/main.tex ~line 605, absolute error
PAPER_F64_REDUCTION_F32_ERROR = 1.22e-4    # paper/main.tex ~line 605-606, absolute error
DIVIDE_PRE_FIX_WORST_CASE = 1.16e-7        # measured before the -t3 correction commit
                                            # (500k random trials, host-side check) -- not
                                            # itself a paper figure, just the number Task 1
                                            # in the request that produced this file names
                                            # as the pre-fix baseline to improve on.

PAPER_CLAIMS = {
    "1": {
        "title": "Headline L40S 2048² speedup (f64 vs DS-f32)",
        "paper_value": "12.87× (f64=12.19ms, DS=0.95ms; f64=1.41 TFLOPS, DS=18.1 TFLOPS)",
        "paper_location": "paper/main.tex table tab:matmul_l40s (~line 449), text ~line 428-433",
        "measured_precision": "default matmul precision (paper text ~line 472: "
                               "\"the performance benchmarks use default precision\")",
        "validated_precision": "precision=HIGHEST (paper text ~line 470: \"we use "
                                "precision=HIGHEST in accuracy tests ... to suppress TF32\")",
    },
    "2": {
        "title": "H100 2048² DS TFLOPS implies TF32 tensor-core dispatch",
        "paper_value": "54.6 TFLOPS effective (2N³ basis) implies ~218 TFLOPS of real f32 "
                        "work, exceeding H100's non-tensor-core f32 peak of ~67 TFLOPS",
        "paper_location": "paper/main.tex ~line 462-466",
    },
    "3": {
        "title": "f32-input reduction (n=10000, val=0.1): DS-f32 error reduction vs plain f32",
        "paper_value": "~15× (plain f32 {:.2e}-{:.2e}, DS-f32 {:.2e})".format(
            min(PAPER_EXP3_PLAIN_F32_ERROR.values()), max(PAPER_EXP3_PLAIN_F32_ERROR.values()),
            PAPER_EXP3_F32_ERROR),
        "paper_location": "paper/main.tex §Reduction Precision (~line 587-600)",
        "note": "Distinct from the *separate* f64-input version of this reduction "
                "(~line 602-606, an 8× figure with different numbers) -- this experiment "
                "re-measures the f32-input/15× claim specifically.",
    },
    "4": {
        "title": "DS divide worst-case relative accuracy, post `-t3` correction",
        "paper_value": "not yet in paper/main.tex as of this writing -- the draft's "
                        "extended-operations table for divide/sqrt (and the Limitations "
                        "section's claim that division is unsupported) predate the "
                        "divide/sqrt port done in this repo's history; this experiment "
                        "supplies the number for whenever that table is added.",
        "paper_location": "none yet -- see paper_value",
        "note": "Pre-fix worst case (500k random trials, host-side check, before the "
                "-t3 correction commit): {:.3e}. Expected post-fix scale: ~2^-48-class "
                "(double-word), same as add/sub/mul/sqrt.".format(DIVIDE_PRE_FIX_WORST_CASE),
    },
    "5": {
        "title": "f64-input reduction (n=10000, val=0.1) residual root cause",
        "paper_value": "paper/main.tex currently states an explanation (input-"
                        "representation error dominates, ~line 608-617) for an 8x error "
                        "reduction figure (DS {:.2e} vs f32 {:.2e}, ~line 602-606) as "
                        "settled fact, not as an open question under investigation -- this "
                        "experiment checks whether that explanation is actually correct.".format(
                            PAPER_F64_REDUCTION_DS_ERROR, PAPER_F64_REDUCTION_F32_ERROR),
        "paper_location": "paper/main.tex §Reduction Precision (~line 602-617)",
    },
}

REL_NOISE_THRESHOLD = 0.20    # exp1 reproduction-check: beyond this, flag CHANGED not PASS
EXP3_STRONG_RATIO = 10.0      # (a)/(b) >= this  ->  contributes to CONFIRMED
EXP3_CLOSE_RATIO = 2.0        # (a)/(b) <  this  ->  REFUTED
EXP3_FLOOR_RATIO = 3.0        # (a) within this factor of (c)  ->  contributes to CONFIRMED


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        return {"_load_error": "invalid JSON in {}: {}".format(path, e)}


def fmt_ms(x):
    return "{:.4f}".format(x) if isinstance(x, (int, float)) else "—"


def fmt_tflops(x):
    return "{:.2f}".format(x) if isinstance(x, (int, float)) else "—"


def exp1_case(doc, n):
    if not doc:
        return None
    for c in doc.get("cases", []):
        if c.get("n") == n:
            return c
    return None


def paper_default_table_for(gpu_tag):
    up = gpu_tag.upper()
    if "L40S" in up:
        return PAPER_L40S_DEFAULT_MS
    if "H100" in up:
        return PAPER_H100_DEFAULT_MS
    return None


def render_exp1(gpu_tag, results_dir):
    configs = {name: load_json(results_dir / "exp1_gemm_highest_{}.json".format(name))
               for name in ["f64_default", "f64_highest", "ds_default", "ds_highest"]}

    lines = ["### Experiment 1 — Headline speedup under precision=HIGHEST", ""]
    if all(v is None for v in configs.values()):
        lines.append("_Not yet run on this GPU._")
        return "\n".join(lines), None

    sizes = sorted({c["n"] for doc in configs.values() if doc for c in doc.get("cases", [])})

    def speedup_table(title, f64_key, ds_key):
        rows = ["**{}**".format(title), "",
                "| Size | f64 (ms) | DS (ms) | Speedup | f64 TFLOPS | DS TFLOPS |",
                "|---|---|---|---|---|---|"]
        for n in sizes:
            f64_case = exp1_case(configs[f64_key], n)
            ds_case = exp1_case(configs[ds_key], n)
            if not f64_case or not ds_case:
                rows.append("| {}² | — | — | — | — | — |".format(n))
                continue
            speedup = (f64_case["median_ms"] / ds_case["median_ms"]) if ds_case["median_ms"] else float("nan")
            rows.append("| {}² | {} | {} | {:.2f}× | {} | {} |".format(
                n, fmt_ms(f64_case["median_ms"]), fmt_ms(ds_case["median_ms"]), speedup,
                fmt_tflops(f64_case.get("tflops")), fmt_tflops(ds_case.get("tflops"))))
        rows.append("")
        return "\n".join(rows)

    lines.append(speedup_table("default precision vs. default precision", "f64_default", "ds_default"))
    lines.append(speedup_table("HIGHEST vs. HIGHEST (headline candidate)", "f64_highest", "ds_highest"))

    paper_default = paper_default_table_for(gpu_tag)
    if paper_default and configs["ds_default"]:
        rows = ["**Reproduction check** (measured ds_default vs. paper's published "
                "default-precision table):", "",
                "| Size | Paper DS (ms) | Measured DS (ms) | Rel. diff | Verdict |",
                "|---|---|---|---|---|"]
        for n, paper_ms in sorted(paper_default.items()):
            case = exp1_case(configs["ds_default"], n)
            if not case:
                rows.append("| {}² | {} | — | — | NOT RUN |".format(n, paper_ms))
                continue
            diff = abs(case["median_ms"] - paper_ms) / paper_ms
            verdict = "PASS" if diff <= REL_NOISE_THRESHOLD else "CHANGED"
            rows.append("| {}² | {} | {} | {:.1f}% | {} |".format(
                n, paper_ms, fmt_ms(case["median_ms"]), diff * 100, verdict))
        rows.append("")
        lines.append("\n".join(rows))

    headline_highest = None
    f64h, dsh = exp1_case(configs["f64_highest"], 2048), exp1_case(configs["ds_highest"], 2048)
    if f64h and dsh and dsh["median_ms"]:
        headline_highest = f64h["median_ms"] / dsh["median_ms"]

    lines.append(
        "**Interpretation:** the paper's published headline (12.87× at 2048², "
        "paper/main.tex ~line 433/449) was measured at default matmul precision; the "
        "paper's own text (~line 470-472) already notes accuracy tests use "
        "precision=HIGHEST while performance benchmarks use default precision -- this "
        "experiment closes that gap. HIGHEST-vs-HIGHEST speedup at 2048² measured "
        "here: **{}**. This is the number that should replace or accompany 12.87× as "
        "the headline, since it is measured under the same precision setting used to "
        "validate accuracy.\n".format(
            "{:.2f}×".format(headline_highest) if headline_highest else "not yet available")
    )
    return "\n".join(lines), headline_highest


def render_exp2(results_dir):
    modes = {mode: load_json(results_dir / "exp2_tf32_dispatch_{}.json".format(mode))
             for mode in ["default", "highest"]}

    lines = ["### Experiment 2 — TF32 dispatch verification", ""]
    if all(v is None for v in modes.values()):
        lines.append("_Not yet run on this GPU._")
        return "\n".join(lines)

    rows = ["| Precision | TF32 dispatch verdict | Evidence quality | nsys available |",
            "|---|---|---|---|"]
    verdict_str = {True: "YES", False: "NO", None: "UNKNOWN (insufficient evidence)"}
    for mode, doc in modes.items():
        if not doc:
            rows.append("| {} | NOT RUN | — | — |".format(mode))
            continue
        v = doc.get("verdict_tf32_dispatch")
        nsys_avail = doc.get("nsys", {}).get("available")
        rows.append("| {} | {} | {} | {} |".format(
            mode, verdict_str.get(v, v), doc.get("evidence_quality", "—"), nsys_avail))
    lines.append("\n".join(rows) + "\n")
    lines.append("**Paper's inference being checked** ({}): {}\n".format(
        PAPER_CLAIMS["2"]["paper_location"], PAPER_CLAIMS["2"]["paper_value"]))

    for mode, doc in modes.items():
        evidence = doc.get("hlo_evidence") if doc else None
        if evidence:
            lines.append("<details><summary>{}: matched HLO dump lines ({})</summary>\n".format(
                mode, len(evidence)))
            lines.append("```")
            for e in evidence[:40]:
                lines.append("{}:{}: {}".format(e["file"], e["line"], e["text"]))
            if len(evidence) > 40:
                lines.append("... ({} more, see the JSON)".format(len(evidence) - 40))
            lines.append("```\n</details>\n")
    return "\n".join(lines)


def render_exp2b(results_dir):
    modes = {mode: load_json(results_dir / "exp2b_fused_add_{}.json".format(mode))
             for mode in ["default", "highest"]}

    lines = ["### Experiment 2b — Did the TwoSum correction survive XLA's optimizer?", ""]
    if all(v is None for v in modes.values()):
        return "\n".join(lines)  # not run -- omit the section entirely rather than clutter with "not run"

    lines.append(
        "Checks whether the DS pass's final `TwoSum(p, e1+e2+e3)` correction "
        "(4 subtracts + 2 adds per call, see `emitTwoSum` in "
        "`DsTransformPass.cpp`) survives in the matmul epilogue after XLA's "
        "optimizer runs, or whether the algebraic simplifier folded the "
        "subtract-based residual chain away (legal under real-number algebra, "
        "not under float rounding).\n"
    )

    rows = ["| Precision | Verdict | Subtracts | Adds | Converts | Pre-opt module had subtracts? |",
            "|---|---|---|---|---|---|"]
    collapsed_modes = []
    extraction_warnings = []
    for mode, doc in modes.items():
        if not doc:
            rows.append("| {} | NOT RUN | — | — | — | — |".format(mode))
            continue
        counts = doc.get("op_counts", {})
        pre = doc.get("pre_optimization_check")
        pre_str = "{} (whole pre-opt module)".format(
            pre["subtract_count_in_whole_module"]) if pre else "not found"
        verdict = doc.get("verdict", "?")
        if str(verdict).startswith("COLLAPSED"):
            collapsed_modes.append(mode)
        if doc.get("extraction_warning"):
            extraction_warnings.append((mode, doc["extraction_warning"]))
        rows.append("| {} | **{}** | {} | {} | {} | {} |".format(
            mode, verdict, counts.get("subtracts", "?"), counts.get("adds", "?"),
            counts.get("converts", "?"), pre_str))
    lines.append("\n".join(rows) + "\n")

    if extraction_warnings:
        lines.append("**⚠ Extraction warning(s) -- verdict(s) above are not trustworthy as-is:**\n")
        for mode, warning in extraction_warnings:
            lines.append("- {}: {}\n".format(mode, warning))

    if collapsed_modes:
        lines.append(
            "**⚠ COLLAPSED on: {}** -- the DS correction may be silently "
            "discarded on this path. This changes what the paper can claim "
            "about that precision mode; see the body below and "
            "`results/<gpu>/exp2b_fused_add_*.json`.\n".format(", ".join(collapsed_modes))
        )

    for mode, doc in modes.items():
        if doc and doc.get("body_text"):
            lines.append("<details><summary>{}: extracted epilogue body ({})</summary>\n".format(
                mode, doc.get("body_source", "?")))
            lines.append("```")
            lines.append(doc["body_text"])
            lines.append("```\n</details>\n")
    return "\n".join(lines)


def render_exp3(results_dir):
    standard = load_json(results_dir / "exp3_pair_accuracy_standard.json")
    pairs = load_json(results_dir / "exp3_pair_accuracy_pairs.json")

    lines = ["### Experiment 3 — Unrecombined accuracy: measure the (h, l) pair directly", ""]
    if not standard and not pairs:
        lines.append("_Not yet run on this GPU._")
        return "\n".join(lines), None

    a = standard.get("error") if standard else None
    b = pairs.get("error") if pairs else None
    c = (standard or pairs or {}).get("half_ulp_f32_floor")

    verdict = "INCONCLUSIVE (missing data)"
    if a is not None and b is not None and c:
        floor_ok = c > 0 and (1.0 / EXP3_FLOOR_RATIO) <= (a / c) <= EXP3_FLOOR_RATIO
        if b > 0 and (a / b) >= EXP3_STRONG_RATIO and floor_ok:
            verdict = "CONFIRMED"
        elif b > 0 and (a / b) < EXP3_CLOSE_RATIO:
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"

    rows = ["| Quantity | Value | Description |", "|---|---|---|"]
    rows.append("| (a) recombined-in-function f32 error | {} | should reproduce paper's {:.2e} |".format(
        "{:.3e}".format(a) if a is not None else "NOT RUN", PAPER_EXP3_F32_ERROR))
    rows.append("| (b) host-recombined f64 error (hi+lo) | {} | via DS_RETURN_PAIRS=1 |".format(
        "{:.3e}".format(b) if b is not None else "NOT RUN"))
    rows.append("| (c) half-ulp f32 floor at result magnitude | {} | analytic |".format(
        "{:.3e}".format(c) if c else "—"))
    lines.append("\n".join(rows) + "\n")
    lines.append("**Verdict: {}**\n".format(verdict))
    lines.append(
        "Verdict logic: CONFIRMED requires (b) at least {:.0f}× smaller than (a) *and* (a) "
        "within {:.0f}× of the analytic half-ulp floor (c); REFUTED if (a) and (b) are within "
        "{:.0f}× of each other; otherwise INCONCLUSIVE.\n".format(
            EXP3_STRONG_RATIO, EXP3_FLOOR_RATIO, EXP3_CLOSE_RATIO)
    )
    n_used = (standard or pairs or {}).get("n")
    if a is not None and n_used == PAPER_EXP3_N:
        dev = abs(a - PAPER_EXP3_F32_ERROR) / PAPER_EXP3_F32_ERROR
        repro = "reproduces" if dev <= REL_NOISE_THRESHOLD else "**DEVIATES FROM**"
        lines.append("(a) = {:.3e} {} the paper's published f32-input DS-f32 error ({:.3e}, "
                     "{}; {:.1f}% relative difference).\n".format(
            a, repro, PAPER_EXP3_F32_ERROR, PAPER_CLAIMS["3"]["paper_location"], dev * 100))
    elif a is not None:
        lines.append(
            "(a) = {:.3e} measured at n={} -- not compared against the paper's published "
            "{:.3e} (measured at n={}); these are different problem sizes, not a "
            "reproduction check. This is expected under --smoke (which intentionally uses "
            "a smaller n for speed) and is not itself a finding.\n".format(
                a, n_used, PAPER_EXP3_F32_ERROR, PAPER_EXP3_N)
        )
    return "\n".join(lines), verdict


def render_exp4(results_dir):
    modes = {mode: load_json(results_dir / "exp4_divide_worst_case_{}.json".format(mode))
             for mode in ["internal", "observable"]}

    lines = ["### Experiment 4 — DS divide worst-case relative accuracy (post `-t3` fix)", ""]
    if all(v is None for v in modes.values()):
        lines.append("_Not yet run on this GPU._")
        return "\n".join(lines), None

    lines.append(
        "Measures divide's worst-case relative accuracy now that emitDsDiv's "
        "dropped TwoProd residual (`t3`) has been restored (see "
        "stablehlo_pass/DsTransformPass.cpp's emitDsDiv comment). Pre-fix "
        "worst case (500k random trials, host-side check): **{:.3e}** "
        "(f32-ULP class). `internal` is divide's own arithmetic accuracy "
        "(via DS_RETURN_PAIRS=1, host-recombined in f64) -- this is the "
        "number for the paper's table. `observable` is the ordinary "
        "f32-typed return, included for comparison but expected to sit at "
        "correctly-rounded-f32 level regardless of internal accuracy for "
        "lo=0 inputs (an isolated correctly-rounded division can't be "
        "improved on by definition).\n".format(DIVIDE_PRE_FIX_WORST_CASE)
    )

    rows = ["| Mode | Random-sweep max | Random-sweep median | Samples | "
            "Adversarial max | Power-of-two all exact? |",
            "|---|---|---|---|---|---|"]
    worst_internal = None
    for mode, doc in modes.items():
        if not doc:
            rows.append("| {} | NOT RUN | — | — | — | — |".format(mode))
            continue
        rs = doc.get("random_sweep", {})
        adv = doc.get("adversarial_cases", [])
        adv_max = max((c["rel_err"] for c in adv), default=None)
        pow2 = doc.get("power_of_two_divisors", {})
        rows.append("| {} | {} | {} | {} | {} | {} |".format(
            mode,
            "{:.3e}".format(rs["combined_max_rel_err"]) if rs.get("combined_max_rel_err") is not None else "—",
            "{:.3e}".format(rs["combined_median_rel_err"]) if rs.get("combined_median_rel_err") is not None else "—",
            rs.get("total_samples", "—"),
            "{:.3e}".format(adv_max) if adv_max is not None else "—",
            pow2.get("all_exact", "—"),
        ))
        if mode == "internal" and rs.get("combined_max_rel_err") is not None:
            worst_internal = max(
                rs["combined_max_rel_err"],
                adv_max if adv_max is not None else 0.0,
            )
    lines.append("\n".join(rows) + "\n")

    if worst_internal is not None:
        improvement = (DIVIDE_PRE_FIX_WORST_CASE / worst_internal) if worst_internal > 0 else float("inf")
        double_word_class = worst_internal < 2.0 ** -40
        lines.append(
            "**Internal worst-case: {:.3e}** ({} pre-fix {:.3e}, ~{:.1f}x "
            "{}). This {} the ~2^-48-class double-word bound the paper's "
            "extended-operations table wants.\n".format(
                worst_internal,
                "improves on" if worst_internal < DIVIDE_PRE_FIX_WORST_CASE else "DOES NOT improve on",
                DIVIDE_PRE_FIX_WORST_CASE, improvement,
                "better" if worst_internal < DIVIDE_PRE_FIX_WORST_CASE else "worse",
                "is consistent with" if double_word_class else "is NOT consistent with (still f32-ULP class)",
            )
        )

    for mode, doc in modes.items():
        if not doc:
            continue
        probe = doc.get("safe_range_probe", {})
        if probe.get("cases"):
            lines.append("<details><summary>{}: beyond-safe-range probe "
                          "(characterization only, not asserted)</summary>\n".format(mode))
            lines.append("| Case | a/limit | Measured finite? | rel_err (if finite) |")
            lines.append("|---|---|---|---|")
            for c in probe["cases"]:
                lines.append("| {} | {:.2f}x | {} | {} |".format(
                    c["label"], c["a_over_limit"], c["measured_is_finite"],
                    "{:.3e}".format(c["rel_err_if_finite"]) if c["rel_err_if_finite"] is not None else "—"))
            lines.append("\n</details>\n")

    return "\n".join(lines), worst_internal


def render_exp5(results_dir):
    stages = {s: load_json(results_dir / "exp5_f64_reduction_bisection_{}.json".format(s))
              for s in ["ground_truth", "split_fidelity", "single_op", "length_scan", "recombination"]}

    lines = ["### Experiment 5 — f64-input reduction residual: root cause bisection", ""]
    if all(v is None for v in stages.values()):
        lines.append("_Not yet run on this GPU._")
        return "\n".join(lines), None

    gt = stages["ground_truth"]
    if gt and gt.get("methodology_finding"):
        lines.append("**Stage 1 (ground truth / methodology) finding:**\n")
        lines.append("> {}\n".format(gt["methodology_finding"]))
        existing = gt.get("existing_test_reproduction", {})
        corrected = gt.get("corrected_apples_to_apples_test", {})
        lines.append(
            "| | Operation | DS result | Truth | DS error (abs) | DS error (rel) |\n"
            "|---|---|---|---|---|---|\n"
            "| Existing test (as currently written) | {} | {} | {} | {} | {} |\n"
            "| Corrected (apples-to-apples with f32-input claim) | {} | {} | {} | {} | {} |\n".format(
                existing.get("operation", "—"), fmt_ms(existing.get("ds_result")),
                fmt_ms(existing.get("truth_f64")),
                "{:.3e}".format(existing["ds_error_abs"]) if "ds_error_abs" in existing else "—",
                "{:.3e}".format(existing["ds_error_rel"]) if "ds_error_rel" in existing else "—",
                corrected.get("operation", "—"), fmt_ms(corrected.get("ds_result")),
                fmt_ms(corrected.get("truth_f64_fsum")),
                "{:.3e}".format(corrected["ds_error_abs"]) if "ds_error_abs" in corrected else "—",
                "{:.3e}".format(corrected["ds_error_rel"]) if "ds_error_rel" in corrected else "—",
            )
        )

    rows = ["| Stage | Key result | Verdict |", "|---|---|---|"]
    for name in ["ground_truth", "split_fidelity", "single_op", "length_scan", "recombination"]:
        doc = stages[name]
        if not doc:
            rows.append("| {} | NOT RUN | — |".format(name))
            continue
        if name == "ground_truth":
            key = "see table above"
            verdict = "METHODOLOGY MISMATCH" if doc.get("same_operation_as_f32_input_claim") is False else "—"
        elif name == "split_fidelity":
            split2 = doc.get("compiled_split_times_two_via_DS_RETURN_PAIRS", {})
            key = "max rel_err {:.3e}".format(split2.get("max_rel_err", float("nan")))
            verdict = doc.get("verdict", "—")
        elif name == "single_op":
            key = "max rel_err {:.3e}".format(doc.get("max_rel_err", float("nan")))
            verdict = doc.get("verdict", "—")
        elif name == "length_scan":
            key = doc.get("growth_classification", "—")
            verdict = "n/a (see rows in JSON)"
        else:  # recombination
            key = "error_rel {:.3e}".format(doc.get("error_rel", float("nan")))
            verdict = "cross-reference vs. ground_truth's corrected test (see note)"
        rows.append("| {} | {} | {} |".format(name, key, verdict))
    lines.append("\n".join(rows) + "\n")

    # Synthesize a bracket from whichever stages actually ran.
    clean_stages = []
    dirty_stages = []
    for name in ["split_fidelity", "single_op"]:
        doc = stages[name]
        if not doc:
            continue
        v = str(doc.get("verdict", ""))
        (clean_stages if v.startswith("CLEAN") else dirty_stages).append(name)

    if gt and gt.get("same_operation_as_f32_input_claim") is False:
        bracket = (
            "**Root cause: methodology mismatch, not a pass defect.** The "
            "existing f64-input reduction test measures a different "
            "operation (plain sum) than the f32-input claim it was being "
            "compared against (sum of squares) -- see Stage 1 above. "
            + (
                "Stages run beyond ground_truth ({}) found no defect in "
                "the f64 ingestion path itself.".format(", ".join(clean_stages))
                if clean_stages and not dirty_stages else
                "Further stages ({}) below are still worth reading for "
                "the f64 path's own accuracy, independent of this finding.".format(
                    ", ".join(s for s in stages if stages[s]) or "none run yet")
            )
        )
    elif dirty_stages:
        bracket = "**Bracket: error absent before {}, present at/after {}.**".format(
            ", ".join(clean_stages) or "no clean stage observed",
            ", ".join(dirty_stages))
    elif clean_stages:
        bracket = ("**All bisected stages ({}) are clean (double-word-class "
                    "accuracy).** No pass defect isolated in the f64 ingestion "
                    "path by this bisection.").format(", ".join(clean_stages))
    else:
        bracket = "**Insufficient stages run to state a bracket.**"
    lines.append(bracket + "\n")

    return "\n".join(lines), bracket


def render_manifest_summary(results_dir):
    manifest = load_json(results_dir / "manifest.json")
    if not manifest:
        return ("_No manifest.json found in this directory -- provenance (commit, "
                "container digest, JAX/CUDA versions) is unknown for this run._\n")
    git = manifest.get("git", {})
    gpu = manifest.get("gpu", {})
    dirty_note = " (dirty tree)" if git.get("dirty") else ""
    return (
        "- Generated: {}\n"
        "- Pass commit: `{}`{}\n"
        "- Container image digest: {}\n"
        "- JAX: {} · CUDA toolkit: {}\n"
        "- GPU: {} (driver {}, {})\n".format(
            manifest.get("generated_at", "?"), git.get("commit", "?"), dirty_note,
            manifest.get("container_image_digest", "?"), manifest.get("jax_version", "?"),
            manifest.get("cuda_toolkit_version", "?"), gpu.get("name", "?"),
            gpu.get("driver_version", "?"), gpu.get("arch", "?"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    gpus = sorted(p.name for p in results_root.iterdir() if p.is_dir()) if results_root.exists() else []

    out = ["# DS-f32 Paper Re-evaluation Report", "",
           "_Generated by `ds_reeval/report.py` from `{}/<gpu>/*.json`. This file is "
           "regenerated wholesale on every run of report.py -- do not hand-edit it._".format(
               results_root), ""]

    if not gpus:
        out.append(
            "No results found under `{}`. Run `python3 run_all.py --dry-run` to see the "
            "launch plan, then see README.md for exact commands to run on each cluster.\n".format(
                results_root)
        )
        Path(args.out).write_text("\n".join(out))
        print("Wrote {} (no results present yet)".format(args.out))
        return 0

    summary_rows = []
    for gpu_tag in gpus:
        gdir = results_root / gpu_tag
        out.append("## {}\n".format(gpu_tag))
        out.append(render_manifest_summary(gdir))
        exp1_md, headline_highest = render_exp1(gpu_tag, gdir)
        out.append(exp1_md)
        out.append(render_exp2(gdir))
        out.append(render_exp2b(gdir))
        exp3_md, exp3_verdict = render_exp3(gdir)
        out.append(exp3_md)
        exp4_md, divide_worst_internal = render_exp4(gdir)
        out.append(exp4_md)
        exp5_md, exp5_bracket = render_exp5(gdir)
        out.append(exp5_md)
        summary_rows.append((gpu_tag, headline_highest, exp3_verdict, divide_worst_internal, exp5_bracket))

    out.append("## Summary\n")
    out.append("| # | Claim | Paper says | Measured | Verdict |")
    out.append("|---|---|---|---|---|")
    for gpu_tag, headline_highest, exp3_verdict, divide_worst_internal, exp5_bracket in summary_rows:
        if headline_highest:
            h_str = "{:.2f}× (HIGHEST precision)".format(headline_highest)
            h_verdict = ("CHANGED" if abs(headline_highest - PAPER_HEADLINE_SPEEDUP_L40S_2048)
                         / PAPER_HEADLINE_SPEEDUP_L40S_2048 > REL_NOISE_THRESHOLD else "PASS")
        else:
            h_str, h_verdict = "not yet run", "—"
        out.append("| 1 ({}) | {} | {:.2f}× (default precision) | {} | {} |".format(
            gpu_tag, PAPER_CLAIMS["1"]["title"], PAPER_HEADLINE_SPEEDUP_L40S_2048, h_str, h_verdict))
        out.append("| 2 ({}) | {} | inferred from TFLOPS, not directly observed | "
                   "see Experiment 2 section above | see above |".format(
                       gpu_tag, PAPER_CLAIMS["2"]["title"]))
        plain_f32_ref = next(
            (v for k, v in PAPER_EXP3_PLAIN_F32_ERROR.items() if k in gpu_tag.upper()),
            PAPER_EXP3_F32_ERROR * 15,
        )
        out.append("| 3 ({}) | {} | ~15× ({:.2e} vs ~{:.2e}) | see Experiment 3 section above | {} |".format(
            gpu_tag, PAPER_CLAIMS["3"]["title"], PAPER_EXP3_F32_ERROR, plain_f32_ref,
            exp3_verdict or "not yet run"))
        out.append("| 4 ({}) | {} | not yet in paper draft (pre-fix baseline {:.2e}) | {} | {} |".format(
            gpu_tag, PAPER_CLAIMS["4"]["title"], DIVIDE_PRE_FIX_WORST_CASE,
            "{:.3e}".format(divide_worst_internal) if divide_worst_internal is not None else "not yet run",
            ("IMPROVED" if divide_worst_internal is not None and divide_worst_internal < DIVIDE_PRE_FIX_WORST_CASE
             else ("REGRESSED" if divide_worst_internal is not None else "—"))))
        out.append("| 5 ({}) | {} | states an explanation as settled fact | see Experiment 5 section above | {} |".format(
            gpu_tag, PAPER_CLAIMS["5"]["title"],
            "see bracket above" if exp5_bracket else "not yet run"))

    Path(args.out).write_text("\n".join(out) + "\n")
    print("Wrote {} covering GPU(s): {}".format(args.out, ", ".join(gpus)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

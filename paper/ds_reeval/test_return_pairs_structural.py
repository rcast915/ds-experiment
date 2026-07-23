#!/usr/bin/env python3
"""
Structural (CPU-only, no GPU/PJRT execution) test for the DS_RETURN_PAIRS=1
flag added to stablehlo_pass/DsTransformPass.cpp for Experiment 3, plus its
later extension to f64-typed returns (added for
exp5_f64_reduction_bisection.py, which needs to observe a raw f64-sourced
DS pair -- the original f32-only version of this flag could not do that).

Drives `mlir-ds-opt` directly on a hand-written, minimal StableHLO function
rather than going through JAX tracing, so the exact input op sequence is
known and the expected output op counts are exact, not approximate --
matching this project's existing "MLIR structural" test style (see
python/test_pass_correctness.py, described in handoff.md as counting
expanded DS ops in the lowered MLIR).

Input function (deliberately the simplest possible case that exercises
func.return's dsMap handling: a float function argument is *always*
dsMap-tracked by convertFuncArgs, no arithmetic op is needed to set up the
test):

    func.func @ret_twice(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
      %unused = stablehlo.constant dense<0> : tensor<i32>
      func.return %arg0, %arg0 : tensor<f32>, tensor<f32>
    }

(The dead i32 constant only exists so the input text contains a
`stablehlo.` op and mlir-ds-opt's parser loads the stablehlo dialect into
the MLIRContext before the pass runs -- otherwise the pass's pre-existing,
unrelated-to-this-flag emitFromFloat() crashes the moment it tries to
programmatically build a stablehlo.constant into a context that never
loaded that dialect. i32 keeps it outside isFloatTensor(), so the pass
skips it via `continue` and it doesn't affect any count below.)

For an f32 argument, emitFromFloat makes hi == %arg0 itself (no new op) and
lo == a new zero constant. So:

  DS_RETURN_PAIRS unset (default): both return operands are independently
    recombined via emitToFloat (2 stablehlo.convert + 1 stablehlo.add each)
    -> exactly 4 converts, 2 adds in the output, and the two returned SSA
    values are different (each its own independent recombination), even
    though they are numerically redundant copies of the same value. This is
    exactly the pre-existing (pre-this-flag) behavior -- proving it is
    unchanged is the "flag off must be a no-op" regression guard.

  DS_RETURN_PAIRS=1: recombination is skipped for this doubled-return
    pattern; the first operand becomes `hi` (= %arg0 directly, an existing
    value, no new op) and the second becomes `lo` (the existing zero
    constant, no new op) -> exactly 0 converts, 0 adds, and the two
    returned SSA values are provably different (hi != lo).

This does not require a GPU, PJRT, or JAX -- only the mlir-ds-opt binary
built from stablehlo_pass/. Run this before attempting
ds_reeval/exp3_pair_accuracy.py --mode pairs on a GPU node, so a broken
build of the flag is caught in seconds, not after burning GPU allocation
time.

Usage:
    python3 test_return_pairs_structural.py
    python3 test_return_pairs_structural.py --opt-binary /path/to/mlir-ds-opt
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MLIR_INPUT = """\
func.func @ret_twice(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %unused = stablehlo.constant dense<0> : tensor<i32>
  func.return %arg0, %arg0 : tensor<f32>, tensor<f32>
}
"""

# f64 variant of the same test, added alongside DS_RETURN_PAIRS's extension
# to f64-typed returns (see DsTransformPass.cpp's func.return comment).
# emitFromFloat's f64 path additionally does the argument split itself (2
# converts + 1 subtract, at arg-processing time, independent of
# DS_RETURN_PAIRS) before func.return's own handling runs -- see the op
# counts below, which account for both.
MLIR_INPUT_F64 = """\
func.func @ret_twice_f64(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
  %unused = stablehlo.constant dense<0> : tensor<i32>
  func.return %arg0, %arg0 : tensor<f64>, tensor<f64>
}
"""

# Single (non-doubled) f64 return -- the exact shape of the bug found via
# null_test_return_pairs_noop.py on real hardware: substitution used to be
# decided per-operand from "1st or 2nd occurrence seen so far" with no
# check on the TOTAL occurrence count, so a value returned exactly once
# also hit "1st occurrence" and got wrongly substituted with hi alone,
# dropping lo. This case must produce IDENTICAL op counts under
# DS_RETURN_PAIRS=1 and unset -- if it doesn't, the single-return case is
# incorrectly being treated as a pair.
MLIR_INPUT_F64_SINGLE = """\
func.func @ret_once_f64(%arg0: tensor<f64>) -> tensor<f64> {
  %unused = stablehlo.constant dense<0> : tensor<i32>
  func.return %arg0 : tensor<f64>
}
"""
# The dead %unused i32 constant above is not arithmetic significant -- it
# exists purely so the input text contains a `stablehlo.` op, which is what
# makes mlir-ds-opt's parser load the stablehlo dialect into the
# MLIRContext. Without it, the parser only ever sees `func` ops, the
# stablehlo dialect never gets loaded, and the pass's *pre-existing*
# emitFromFloat/convertFuncArgs (unrelated to DS_RETURN_PAIRS, unchanged by
# this feature) crashes the first time it programmatically builds a
# stablehlo.constant -- "isn't known in this MLIRContext: the dialect may
# not be loaded". i32 (not f32/f64) means isFloatTensor() is false, so the
# pass's ConstantOp handler skips it via `continue` -- it passes through
# untouched and does not affect the convert/add counts asserted below.

DEFAULT_OPT_BINARY = (
    Path(__file__).resolve().parent.parent.parent / "stablehlo_pass" / "build" / "mlir-ds-opt"
)


def run_pass(opt_binary, mlir_text, return_pairs):
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(mlir_text)
        in_path = f.name
    try:
        env = dict(os.environ)
        if return_pairs:
            env["DS_RETURN_PAIRS"] = "1"
        else:
            env.pop("DS_RETURN_PAIRS", None)
        proc = subprocess.run(
            [str(opt_binary), "--pass-pipeline=builtin.module(func.func(ds-transform))", in_path],
            capture_output=True, text=True, env=env, timeout=30,
        )
    finally:
        os.unlink(in_path)
    if proc.returncode != 0:
        raise RuntimeError(
            "mlir-ds-opt failed (return_pairs={}):\n{}".format(return_pairs, proc.stderr)
        )
    return proc.stdout


def count_ops(mlir_text, short_mnemonic):
    # StableHLO's pretty text form prints ops as `%N = stablehlo.<mnemonic> ...`.
    return len(re.findall(r"=\s*stablehlo\.{}\b".format(re.escape(short_mnemonic)), mlir_text))


def extract_return_operands(mlir_text):
    # MLIR's printer emits func.return in its short assembly form -- bare
    # `return ...`, no `func.` prefix -- so match either spelling. Anchored
    # to line-start (after whitespace) so this can't accidentally match an
    # unrelated `foo.return` inside a nested region.
    for line in mlir_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("return ") or stripped.startswith("func.return "):
            m = re.match(r"(?:func\.)?return\s+(.+?)\s*:", stripped)
            if m:
                return [tok.strip() for tok in m.group(1).split(",")]
    raise RuntimeError("could not find func.return in pass output:\n{}".format(mlir_text))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opt-binary", default=str(DEFAULT_OPT_BINARY))
    args = parser.parse_args()

    opt_binary = Path(args.opt_binary)
    if not opt_binary.exists():
        print(
            "SKIP: mlir-ds-opt not found at {} -- build stablehlo_pass first:\n"
            "  cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build "
            "-DCMAKE_CXX_FLAGS=\"-fno-rtti\" && ninja -C stablehlo_pass/build".format(opt_binary),
            file=sys.stderr,
        )
        return 1

    default_out = run_pass(opt_binary, MLIR_INPUT, return_pairs=False)
    pairs_out = run_pass(opt_binary, MLIR_INPUT, return_pairs=True)
    default_out_f64 = run_pass(opt_binary, MLIR_INPUT_F64, return_pairs=False)
    pairs_out_f64 = run_pass(opt_binary, MLIR_INPUT_F64, return_pairs=True)
    single_default_f64 = run_pass(opt_binary, MLIR_INPUT_F64_SINGLE, return_pairs=False)
    single_pairs_f64 = run_pass(opt_binary, MLIR_INPUT_F64_SINGLE, return_pairs=True)

    failures = []

    # --- DS_RETURN_PAIRS unset: must reproduce pre-existing recombination
    # behavior exactly -- 2 independent emitToFloat expansions.
    n_convert_default = count_ops(default_out, "convert")
    n_add_default = count_ops(default_out, "add")
    if n_convert_default != 4:
        failures.append("default mode: expected 4 stablehlo.convert, got {}".format(n_convert_default))
    if n_add_default != 2:
        failures.append("default mode: expected 2 stablehlo.add, got {}".format(n_add_default))
    ops_default = extract_return_operands(default_out)
    if len(ops_default) != 2 or ops_default[0] == ops_default[1]:
        failures.append(
            "default mode: expected 2 distinct SSA names from two independent "
            "recombinations, got {}".format(ops_default)
        )

    # --- DS_RETURN_PAIRS=1: recombination must be skipped for the doubled
    # return -- 0 converts, 0 adds, two genuinely different return operands
    # (hi, then lo).
    n_convert_pairs = count_ops(pairs_out, "convert")
    n_add_pairs = count_ops(pairs_out, "add")
    if n_convert_pairs != 0:
        failures.append("pairs mode: expected 0 stablehlo.convert, got {}".format(n_convert_pairs))
    if n_add_pairs != 0:
        failures.append("pairs mode: expected 0 stablehlo.add, got {}".format(n_add_pairs))
    ops_pairs = extract_return_operands(pairs_out)
    if len(ops_pairs) != 2 or ops_pairs[0] == ops_pairs[1]:
        failures.append(
            "pairs mode: expected 2 distinct SSA values (hi, lo), got {}".format(ops_pairs)
        )

    # --- f64 argument, DS_RETURN_PAIRS unset: emitFromFloat's f64 split (3
    # converts -- hi=convert(v,f32), hi_as_f64=convert(hi,f64),
    # lo=convert(diff,f32) -- + 1 subtract, once) plus two independent
    # emitToFloat recombinations (4 converts + 2 adds) -- unchanged by this
    # feature.
    n_convert_default_f64 = count_ops(default_out_f64, "convert")
    n_add_default_f64 = count_ops(default_out_f64, "add")
    n_sub_default_f64 = count_ops(default_out_f64, "subtract")
    if n_convert_default_f64 != 7:
        failures.append("f64 default mode: expected 7 stablehlo.convert (3 split + 4 recombine), "
                         "got {}".format(n_convert_default_f64))
    if n_add_default_f64 != 2:
        failures.append("f64 default mode: expected 2 stablehlo.add, got {}".format(n_add_default_f64))
    if n_sub_default_f64 != 1:
        failures.append("f64 default mode: expected 1 stablehlo.subtract (split), got {}".format(
            n_sub_default_f64))
    ops_default_f64 = extract_return_operands(default_out_f64)
    if len(ops_default_f64) != 2 or ops_default_f64[0] == ops_default_f64[1]:
        failures.append(
            "f64 default mode: expected 2 distinct SSA names from two independent "
            "recombinations, got {}".format(ops_default_f64)
        )

    # --- f64 argument, DS_RETURN_PAIRS=1: split still happens (3 converts +
    # 1 subtract, unrelated to this flag), but return-time recombination is
    # replaced by widening hi/lo to f64 directly -- 2 more converts, 0 adds,
    # for a total of 5 converts, 0 adds, 1 subtract. This is the exact new
    # code path added for exp5_f64_reduction_bisection.py.
    n_convert_pairs_f64 = count_ops(pairs_out_f64, "convert")
    n_add_pairs_f64 = count_ops(pairs_out_f64, "add")
    n_sub_pairs_f64 = count_ops(pairs_out_f64, "subtract")
    if n_convert_pairs_f64 != 5:
        failures.append("f64 pairs mode: expected 5 stablehlo.convert (3 split + 2 widen), "
                         "got {}".format(n_convert_pairs_f64))
    if n_add_pairs_f64 != 0:
        failures.append("f64 pairs mode: expected 0 stablehlo.add, got {}".format(n_add_pairs_f64))
    if n_sub_pairs_f64 != 1:
        failures.append("f64 pairs mode: expected 1 stablehlo.subtract (split), got {}".format(
            n_sub_pairs_f64))
    ops_pairs_f64 = extract_return_operands(pairs_out_f64)
    if len(ops_pairs_f64) != 2 or ops_pairs_f64[0] == ops_pairs_f64[1]:
        failures.append(
            "f64 pairs mode: expected 2 distinct SSA values (widened hi, widened lo), "
            "got {}".format(ops_pairs_f64)
        )

    # --- f64 SINGLE (non-doubled) return: op counts under DS_RETURN_PAIRS=1
    # must be IDENTICAL to unset -- this is the exact regression guard for
    # the fixed bug (see DsTransformPass.cpp's func.return comment). Split
    # (3 converts + 1 subtract) + emitToFloat recombination (2 converts +
    # 1 add) = 5 converts, 1 add, 1 subtract, for BOTH modes.
    n_convert_single_default = count_ops(single_default_f64, "convert")
    n_add_single_default = count_ops(single_default_f64, "add")
    n_sub_single_default = count_ops(single_default_f64, "subtract")
    n_convert_single_pairs = count_ops(single_pairs_f64, "convert")
    n_add_single_pairs = count_ops(single_pairs_f64, "add")
    n_sub_single_pairs = count_ops(single_pairs_f64, "subtract")
    if (n_convert_single_default, n_add_single_default, n_sub_single_default) != (5, 1, 1):
        failures.append(
            "f64 single-return default mode: expected (5 converts, 1 add, 1 subtract), "
            "got ({}, {}, {})".format(n_convert_single_default, n_add_single_default,
                                       n_sub_single_default)
        )
    if (n_convert_single_pairs, n_add_single_pairs, n_sub_single_pairs) != \
       (n_convert_single_default, n_add_single_default, n_sub_single_default):
        failures.append(
            "f64 single-return: DS_RETURN_PAIRS=1 must be a no-op here (op counts must "
            "match the default-mode counts exactly) -- default=({}, {}, {}) convert/add/"
            "subtract, pairs=({}, {}, {}). A mismatch means a single-return value is "
            "being incorrectly treated as a doubled pair (the bug this test exists to "
            "catch).".format(n_convert_single_default, n_add_single_default,
                              n_sub_single_default, n_convert_single_pairs,
                              n_add_single_pairs, n_sub_single_pairs)
        )

    if failures:
        print("FAIL:")
        for f in failures:
            print("  - {}".format(f))
        print("\n--- default-mode (DS_RETURN_PAIRS unset) output ---\n" + default_out)
        print("\n--- pairs-mode (DS_RETURN_PAIRS=1) output ---\n" + pairs_out)
        print("\n--- f64 default-mode output ---\n" + default_out_f64)
        print("\n--- f64 pairs-mode output ---\n" + pairs_out_f64)
        print("\n--- f64 single-return default-mode output ---\n" + single_default_f64)
        print("\n--- f64 single-return pairs-mode output ---\n" + single_pairs_f64)
        return 1

    print("PASS: DS_RETURN_PAIRS structural test (doubled-return x 2 dtypes, "
          "plus single-return f64 no-op regression guard, all as expected)")
    print("  f32 default: {} converts, {} adds, return={}".format(
        n_convert_default, n_add_default, ops_default))
    print("  f32 pairs:   {} converts, {} adds, return={}".format(
        n_convert_pairs, n_add_pairs, ops_pairs))
    print("  f64 default: {} converts, {} adds, {} subtracts, return={}".format(
        n_convert_default_f64, n_add_default_f64, n_sub_default_f64, ops_default_f64))
    print("  f64 pairs:   {} converts, {} adds, {} subtracts, return={}".format(
        n_convert_pairs_f64, n_add_pairs_f64, n_sub_pairs_f64, ops_pairs_f64))
    print("  f64 single-return default: {} converts, {} adds, {} subtracts".format(
        n_convert_single_default, n_add_single_default, n_sub_single_default))
    print("  f64 single-return pairs:   {} converts, {} adds, {} subtracts".format(
        n_convert_single_pairs, n_add_single_pairs, n_sub_single_pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

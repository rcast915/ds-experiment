#!/usr/bin/env bash
# run_tests.sh — DS compiler pass test suite
#
# Runs accuracy tests and (optionally) benchmarks for the dot product and
# matrix multiply DS transformations.
#
# Usage (from /src/ds_experiment inside the container):
#
#   bash tests/run_tests.sh              # accuracy only (auto-detects GPU)
#   bash tests/run_tests.sh --bench      # accuracy + benchmarks
#   bash tests/run_tests.sh --cpu-only   # skip GPU tests even if plugin is built
#   bash tests/run_tests.sh --bench --cpu-only
#
# Prerequisites:
#   ds_setup.sh must have been run this session to build the PJRT plugin.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse flags ───────────────────────────────────────────────────────────────
RUN_BENCH=0
CPU_ONLY=0
F64_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --bench)     RUN_BENCH=1 ;;
        --cpu-only)  CPU_ONLY=1  ;;
        --f64-only)  F64_ONLY=1  ;;
        -h|--help)
            sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg  (use --bench, --cpu-only, --f64-only, --help)"
            exit 1
            ;;
    esac
done

# ── Detect plugin ─────────────────────────────────────────────────────────────
PLUGIN_SO="$PROJECT_ROOT/pjrt_plugin/build/libds_pjrt_plugin.so"
OPT_BINARY="$PROJECT_ROOT/stablehlo_pass/build/mlir-ds-opt"

USE_PLUGIN=0
if [[ "$CPU_ONLY" -eq 0 && -f "$PLUGIN_SO" ]]; then
    USE_PLUGIN=1
    PJRT_ENV="PJRT_NAMES_AND_LIBRARY_PATHS=cuda:$PLUGIN_SO"
else
    PJRT_ENV=""
fi

# ── Pretty-printing helpers ───────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

banner() { echo -e "\n${BOLD}$1${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET}  $1"; }
warn()   { echo -e "  ${YELLOW}!${RESET}  $1"; }
fail()   { echo -e "  ${RED}✗${RESET}  $1"; }

# ── Sanity checks ─────────────────────────────────────────────────────────────
banner "DS Compiler Pass — Test Suite"
echo "  Project root : $PROJECT_ROOT"
echo "  mlir-ds-opt  : $([ -f "$OPT_BINARY" ] && echo "found" || echo "NOT FOUND — structural tests will skip")"
echo "  PJRT plugin  : $([ -f "$PLUGIN_SO" ]  && echo "found" || echo "NOT FOUND — GPU tests will skip")"
echo "  GPU mode     : $([ "$USE_PLUGIN" -eq 1 ] && echo "YES" || echo "no (CPU only)")"
echo "  Benchmarks   : $([ "$RUN_BENCH" -eq 1 ] && echo "yes" || echo "no (pass --bench to enable)")"

if [[ ! -f "$OPT_BINARY" ]]; then
    warn "mlir-ds-opt not built — build it with:"
    warn "  cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS=-fno-rtti"
    warn "  ninja -C stablehlo_pass/build"
fi

if [[ "$USE_PLUGIN" -eq 0 && "$CPU_ONLY" -eq 0 && ! -f "$PLUGIN_SO" ]]; then
    warn "PJRT plugin not built — build it with:"
    warn "  cmake -GNinja -S pjrt_plugin -B pjrt_plugin/build && ninja -C pjrt_plugin/build"
fi

# ── Run a test script, capturing exit code ────────────────────────────────────
PASS_COUNT=0
FAIL_COUNT=0
FAILED_SCRIPTS=()

run_test() {
    local label="$1"
    local script="$2"

    banner "Running: $label"

    local cmd
    if [[ "$USE_PLUGIN" -eq 1 ]]; then
        cmd="env $PJRT_ENV python3 $SCRIPT_DIR/$script"
    else
        cmd="env JAX_PLATFORMS=cpu python3 $SCRIPT_DIR/$script"
    fi

    set +e
    eval "$cmd"
    local rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
        ok "$label — PASSED"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        fail "$label — FAILED (exit $rc)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_SCRIPTS+=("$label")
    fi
}

run_bench() {
    local label="$1"
    local script="$2"

    banner "Benchmark: $label"

    local cmd
    if [[ "$USE_PLUGIN" -eq 1 ]]; then
        cmd="env $PJRT_ENV python3 $SCRIPT_DIR/$script"
    else
        cmd="env JAX_PLATFORMS=cpu python3 $SCRIPT_DIR/$script"
    fi

    set +e
    eval "$cmd"
    set -e
    # Benchmarks are informational — don't count them as pass/fail.
}

# ── Accuracy tests ────────────────────────────────────────────────────────────
if [[ "$F64_ONLY" -eq 0 ]]; then
    run_test "Dot Product Accuracy (f32)" "test_dot_product.py"
    run_test "Matrix Multiply Accuracy (f32)" "test_matmul.py"
fi

# f64 test needs JAX_ENABLE_X64=1 in the environment.
if [[ "$USE_PLUGIN" -eq 1 ]]; then
    f64_cmd="env $PJRT_ENV JAX_ENABLE_X64=1 python3 $SCRIPT_DIR/test_f64_ds.py"
else
    f64_cmd="env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 $SCRIPT_DIR/test_f64_ds.py"
fi

banner "Running: Float64 → DS Accuracy"
set +e
eval "$f64_cmd"
f64_rc=$?
set -e
if [[ $f64_rc -eq 0 ]]; then
    ok "Float64 → DS Accuracy — PASSED"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    fail "Float64 → DS Accuracy — FAILED (exit $f64_rc)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_SCRIPTS+=("Float64 → DS Accuracy")
fi

# ── Benchmarks (optional) ─────────────────────────────────────────────────────
if [[ "$RUN_BENCH" -eq 1 ]]; then
    if [[ "$USE_PLUGIN" -eq 0 ]]; then
        warn "Benchmarks in CPU mode — timing reflects JAX CPU, not GPU throughput"
    fi
    if [[ "$F64_ONLY" -eq 0 ]]; then
        run_bench "Dot Product Performance (f32)" "bench_dot_product.py"
        run_bench "Matrix Multiply Performance (f32)" "bench_matmul.py"
    fi
    # f64 vs DS benchmark — always needs JAX_ENABLE_X64=1.
    banner "Benchmark: Float64 vs DS-f32 Performance"
    if [[ "$USE_PLUGIN" -eq 1 ]]; then
        eval "env $PJRT_ENV JAX_ENABLE_X64=1 python3 $SCRIPT_DIR/bench_f64_vs_ds.py" || true
    else
        eval "env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 $SCRIPT_DIR/bench_f64_vs_ds.py" || true
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
banner "Summary"
echo "  Passed : $PASS_COUNT"
echo "  Failed : $FAIL_COUNT"

if [[ ${#FAILED_SCRIPTS[@]} -gt 0 ]]; then
    fail "Failed suites:"
    for s in "${FAILED_SCRIPTS[@]}"; do
        echo "    - $s"
    done
    exit 1
else
    ok "All test suites passed."
    exit 0
fi

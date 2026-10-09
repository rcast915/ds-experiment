#!/usr/bin/env bash
# run_full.sh — everything to run on a new GPU, in one go.
#
# Run from /src/ds_experiment inside the container (interactively, or from a
# batch script such as submit_l40s.sh):
#
#   bash tests/run_full.sh [label]
#
# Steps:
#   1. ds_setup.sh (disable JAX CUDA auto-registration, build the PJRT plugin)
#   2. rebuild mlir-ds-opt from the current source
#   3. tests/run_tests.sh --bench       (accuracy suites + all benchmarks,
#                                        including Black-Scholes)
#   4. pair-accuracy sections           (DS_RETURN_PAIRS=1: divide, sqrt, exp/log)
#   5. tests/diag_f64_ops.py            (per-op accuracy on f64 inputs)
#   6. tests/diag_f64_matmul.py         (DS matmul accuracy and time vs f64 / f32)
#
# Everything is also written to results/<gpu>_<label>.txt. A failing step does
# not stop the later ones; the exit code is non-zero if any step failed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── GPU name ──────────────────────────────────────────────────────────────────
# nvidia-smi is not always on PATH inside Apptainer --nv containers; the kernel
# procfs entry needs no userspace tooling.
gpu_name() {
    local name=""
    if command -v nvidia-smi > /dev/null 2>&1; then
        name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    fi
    if [[ -z "$name" ]]; then
        name="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                | head -1 | sed 's/^Model:[[:space:]]*//')"
    fi
    echo "${name:-unknown-gpu}"
}

GPU="$(gpu_name)"
LABEL="${1:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p results
OUT="results/$(echo "$GPU" | tr -c 'A-Za-z0-9\n' '-')_${LABEL}.txt"

PLUGIN_SO="$PROJECT_ROOT/pjrt_plugin/build/libds_pjrt_plugin.so"
PJRT="PJRT_NAMES_AND_LIBRARY_PATHS=cuda:$PLUGIN_SO"

FAILED=()

step() {
    local name="$1"; shift
    echo
    echo "################################################################################"
    echo "# $name"
    echo "################################################################################"
    if "$@"; then
        echo "[run_full] OK: $name"
    else
        echo "[run_full] FAILED (exit $?): $name"
        FAILED+=("$name")
    fi
}

build_pass() {
    cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build \
          -DCMAKE_CXX_FLAGS="-fno-rtti" > /dev/null \
        && ninja -C stablehlo_pass/build
}

main() {
    echo "GPU     : $GPU"
    echo "Date    : $(date -Is)"
    echo "Host    : $(hostname)"
    echo "Commit  : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
         "$(git diff --quiet HEAD 2>/dev/null || echo '(with uncommitted changes)')"

    step "setup: ds_setup.sh" bash ds_setup.sh
    step "build: mlir-ds-opt" build_pass

    # Without the plugin and the pass nothing below is meaningful.
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo; echo "[run_full] Setup or build failed — stopping."
        return 1
    fi

    step "tests + benchmarks: run_tests.sh --bench" bash tests/run_tests.sh --bench

    for t in test_ds_divide.py test_ds_sqrt.py test_ds_exp_log.py; do
        step "pair accuracy (DS_RETURN_PAIRS=1): $t" \
            env "$PJRT" DS_RETURN_PAIRS=1 python3 "tests/$t"
    done

    step "per-op f64 accuracy: diag_f64_ops.py" \
        env "$PJRT" python3 tests/diag_f64_ops.py

    step "matmul accuracy and cost: diag_f64_matmul.py" \
        env "$PJRT" python3 tests/diag_f64_matmul.py

    echo
    echo "################################################################################"
    echo "# Summary — $GPU"
    echo "################################################################################"
    if [[ ${#FAILED[@]} -eq 0 ]]; then
        echo "All steps OK."
    else
        echo "Failed steps:"
        printf '  - %s\n' "${FAILED[@]}"
        return 1
    fi
}

main 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[run_full] Full output saved to $OUT"
exit "$rc"

#!/usr/bin/env bash
# run_accuracy.sh — setup, rebuild, then the two accuracy diagnostics:
#   tests/diag_ds_ops_mp.py   per-op accuracy (add, mul, div, sqrt, exp, log) vs 50-digit reference
#   tests/diag_bs_floor.py    Black-Scholes accuracy by stage, constants check, constants-as-arguments
#
# Run from /src/ds_experiment inside the container, or from a batch script:
#   bash tests/run_accuracy.sh [label]
#   DS_JOB_SCRIPT=tests/run_accuracy.sh bash submit_blackwell.sh      # Stravinsky
#
# Takes ~15-30 minutes, mostly the 50-digit reference on the CPU.
# Output: results/<gpu>_accuracy_<label>.txt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$gpu" ]] && gpu="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                          | head -1 | sed 's/^Model:[[:space:]]*//')"
gpu="$(echo "${gpu:-unknown-gpu}" | tr -c 'A-Za-z0-9\n' '-')"
mkdir -p results
OUT="results/${gpu}_accuracy_${1:-$(date +%Y%m%d-%H%M%S)}.txt"
PJRT="cuda:$PWD/pjrt_plugin/build/libds_pjrt_plugin.so"

{
    bash tests/run_metadata.sh
    bash ds_setup.sh || exit 1
    cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" > /dev/null \
        && ninja -C stablehlo_pass/build || exit 1
    echo
    PJRT_NAMES_AND_LIBRARY_PATHS="$PJRT" python3 -u tests/diag_ds_ops_mp.py "${DS_OPS_N:-1000000}"
    echo
    PJRT_NAMES_AND_LIBRARY_PATHS="$PJRT" python3 -u tests/diag_bs_floor.py 1024 65536
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[run_accuracy] Output saved to $OUT"
exit "$rc"

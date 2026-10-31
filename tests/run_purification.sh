#!/usr/bin/env bash
# run_purification.sh — setup, rebuild, then the density-matrix purification benchmark.
#
# Run from /src/ds_experiment inside the container, or from a batch script:
#   bash tests/run_purification.sh [label]
#   sbatch --export=ALL,DS_JOB_SCRIPT=tests/run_purification.sh submit_l40s.sh
#
# Output is also written to results/<gpu>_purification_<label>.txt.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$gpu" ]] && gpu="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                          | head -1 | sed 's/^Model:[[:space:]]*//')"
gpu="${gpu:-unknown-gpu}"
mkdir -p results
OUT="results/$(echo "$gpu" | tr -c 'A-Za-z0-9\n' '-')_purification_${1:-$(date +%Y%m%d-%H%M%S)}.txt"

{
    echo "GPU    : $gpu"
    echo "Date   : $(date -Is)"
    echo "Commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    bash ds_setup.sh || exit 1
    cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" > /dev/null \
        && ninja -C stablehlo_pass/build || exit 1
    PJRT_NAMES_AND_LIBRARY_PATHS="cuda:$PWD/pjrt_plugin/build/libds_pjrt_plugin.so" \
        python3 -u tests/bench_purification.py
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[run_purification] Output saved to $OUT"
exit "$rc"

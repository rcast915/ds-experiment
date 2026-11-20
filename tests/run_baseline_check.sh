#!/usr/bin/env bash
# run_baseline_check.sh — setup, rebuild, then tests/bench_f64_baseline.py:
# is the native f64 baseline the same with no proxy, with the proxy bypassed
# (DS_BYPASS=1), and serialized at the pass's StableHLO 1.16.3?
#
#   bash tests/run_baseline_check.sh [label]
#   DS_JOB_SCRIPT=tests/run_baseline_check.sh nohup bash submit_blackwell.sh \
#       > logs/output/baseline_$(hostname -s).out 2>&1 &
#
# Output is also written to results/<gpu>_baseline_<label>.txt.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$gpu" ]] && gpu="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                          | head -1 | sed 's/^Model:[[:space:]]*//')"
gpu="${gpu:-unknown-gpu}"
mkdir -p results
OUT="results/$(echo "$gpu" | tr -c 'A-Za-z0-9\n' '-')_baseline_${1:-$(date +%Y%m%d-%H%M%S)}.txt"

{
    bash tests/run_metadata.sh
    bash ds_setup.sh || exit 1
    cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" > /dev/null \
        && ninja -C stablehlo_pass/build || exit 1
    python3 -u tests/bench_f64_baseline.py
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[run_baseline_check] Output saved to $OUT"
exit "$rc"

#!/usr/bin/env bash
# run_migration_check.sh — does the existing pass + ds_gemm work in the CUDA 13 /
# JAX 0.12.1 image? Runs the full suite (tests/run_full.sh: unit tests,
# Black-Scholes and the other benchmarks, pair-accuracy tests, matmul
# diagnostics) and the purification benchmark with DS_STRICT=1, so a pass
# failure aborts instead of silently running native f64, then checks the logs
# for the StableHLO version negotiation.
#
#   DS_JOB_SCRIPT=tests/run_migration_check.sh nohup bash submit_blackwell.sh \
#       > logs/output/migration_$(hostname -s).out 2>&1 &
#
# Output: results/<gpu>_<label>.txt (run_full), results/<gpu>_purification_<label>.txt,
# and a verdict at the end of this script's own output.

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

#   bash tests/run_migration_check.sh --check-only <label>   re-reads an earlier
#                                                            run's logs, no rerun
if [[ "${1:-}" == "--check-only" ]]; then
    label="${2:?usage: --check-only <label>}"; rc_full=0; rc_pur=0
else
    label="${1:-migration-$(date +%m%d-%H%M)}"
    export DS_STRICT=1
    bash tests/run_full.sh "$label"
    rc_full=$?
    bash tests/run_purification.sh "$label"
    rc_pur=$?
fi

# Find the logs by label: nvidia-smi is not always on PATH in the container,
# so the GPU part of the file name is not recomputed here.
shopt -s nullglob
logs=(results/*_"${label}".txt)
shopt -u nullglob
echo "  logs read: ${logs[*]:-NONE FOUND for label $label}"
[[ ${#logs[@]} -gt 0 ]] || { echo "  VERDICT: FAIL -- no result files"; exit 1; }

echo
echo "================================ Migration check ================================"
count() { cat "${logs[@]}" 2>/dev/null | grep -c "$1"; }
echo "  backend StableHLO / reported : $(cat "${logs[@]}" 2>/dev/null | grep -m1 -o 'StableHLO: backend.*' || echo 'not logged')"
echo "  programs serialized by       : $(cat "${logs[@]}" 2>/dev/null | grep -o 'serialized by StableHLO_v[0-9.]*' | sort | uniq -c | tr '\n' ' ')"
echo "  programs transformed         : $(count '\[ds_pjrt\] transformed')"
echo "  pass failures / aborts       : $(count 'Pass failed')"
echo "  DS matmul kernel registered  : $(count 'DS matmul kernel registered')"
echo "  run_full.sh exit             : $rc_full"
echo "  run_purification.sh exit     : $rc_pur"
ok=1
[[ $rc_full -eq 0 && $rc_pur -eq 0 ]] || ok=0
[[ $(count 'Pass failed') -eq 0 ]] || ok=0
[[ $(count '\[ds_pjrt\] transformed') -gt 0 ]] || ok=0
cat "${logs[@]}" 2>/dev/null | grep -o 'serialized by StableHLO_v[0-9.]*' | grep -qv 'v1\.16\.3$' && ok=0
echo "  VERDICT: $([[ $ok -eq 1 ]] && echo 'PASS -- pass and ds_gemm work on the new image' || echo 'FAIL -- see above and the logs')"
[[ $ok -eq 1 ]]

#!/usr/bin/env bash
# run_nsight.sh — profile Black-Scholes, native f64 vs DS, with NVIDIA Nsight.
#
# Run from /src/ds_experiment inside the container, or from a batch script:
#   bash tests/run_nsight.sh [label]
#   sbatch --export=ALL,DS_JOB_SCRIPT=tests/run_nsight.sh submit_l40s.sh
#
# Question it answers: at 16M options, is DS slow because XLA splits it into
# many kernels that pass intermediate arrays through memory (suspected), or
# because one huge kernel spills registers, or something else?
#
#   1. Nsight Systems (nsys): kernels launched per call and GPU time per kernel,
#      for f64 and DS at 1M and 16M options. Needs no special permissions.
#   2. Nsight Compute (ncu): per-kernel memory traffic, registers and spills for
#      one 16M call in each mode. Needs GPU performance-counter access, which
#      some clusters restrict; if refused, step 1 still stands on its own.
#
# Output goes to results/nsight_<gpu>_<label>/ and a summary to
# results/<gpu>_nsight_<label>.txt.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$gpu" ]] && gpu="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                          | head -1 | sed 's/^Model:[[:space:]]*//')"
gpu="$(echo "${gpu:-unknown-gpu}" | tr -c 'A-Za-z0-9\n' '-')"
label="${1:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="results/nsight_${gpu}_${label}"
SUMMARY="results/${gpu}_nsight_${label}.txt"
mkdir -p "$OUT_DIR"

REPS=10
PLUGIN="cuda:$PWD/pjrt_plugin/build/libds_pjrt_plugin.so"

# Find the Nsight tools: on PATH, or in the usual CUDA / Nsight install locations.
find_tool() {
    local name="$1" p
    p="$(command -v "$name" 2>/dev/null)" && { echo "$p"; return; }
    for p in /usr/local/cuda/bin/"$name" /opt/nvidia/nsight-*/*/bin/"$name" \
             /opt/nvidia/nsight-*/bin/"$name" /usr/local/cuda/nsight-*/bin/"$name" \
             /usr/local/cuda-*/bin/"$name"; do
        [[ -x "$p" ]] && { echo "$p"; return; }
    done
    return 1
}

run_mode() {   # run_mode <f64|DS> <cmd...>
    local mode="$1"; shift
    if [[ "$mode" == f64 ]]; then
        env DS_BYPASS=1 PJRT_NAMES_AND_LIBRARY_PATHS="$PLUGIN" JAX_ENABLE_X64=1 "$@"
    else
        env -u DS_BYPASS PJRT_NAMES_AND_LIBRARY_PATHS="$PLUGIN" JAX_ENABLE_X64=1 "$@"
    fi
}

main() {
    echo "GPU    : $gpu"
    echo "Date   : $(date -Is)"
    echo "Commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

    bash ds_setup.sh || return 1
    cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" > /dev/null \
        && ninja -C stablehlo_pass/build || return 1

    NSYS="$(find_tool nsys)" || NSYS=""
    NCU="$(find_tool ncu)" || NCU=""
    echo "nsys   : ${NSYS:-NOT FOUND}"
    echo "ncu    : ${NCU:-NOT FOUND}"

    # ── 1. Nsight Systems ────────────────────────────────────────────────────
    if [[ -n "$NSYS" ]]; then
        for n in 1048576 16777216; do
            for mode in f64 DS; do
                tag="${mode}_${n}"
                echo; echo "=== nsys: $tag ==="
                run_mode "$mode" "$NSYS" profile --trace=cuda \
                    --capture-range=cudaProfilerApi --capture-range-end=stop \
                    --force-overwrite=true -o "$OUT_DIR/$tag" \
                    python3 tests/profile_blackscholes.py "$n" "$REPS" 2>&1 | tail -5
                "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
                    --output "$OUT_DIR/$tag" --force-export=true "$OUT_DIR/$tag.nsys-rep" > /dev/null 2>&1 \
                || "$NSYS" stats --report gpukernsum --format csv \
                    --output "$OUT_DIR/$tag" "$OUT_DIR/$tag.nsys-rep" > /dev/null 2>&1 \
                || echo "  [nsys stats failed for $tag]"
            done
        done
        python3 tests/summarize_nsys.py "$OUT_DIR" "$REPS"
    else
        echo
        echo "Nsight Systems (nsys) is not in this container; skipping it (ncu below gives the kernel list)."
    fi

    # ── 2. Nsight Compute ────────────────────────────────────────────────────
    if [[ -n "$NCU" ]]; then
        for mode in f64 DS; do
            tag="ncu_${mode}_16777216"
            echo; echo "=== ncu: $tag (one call; kernels are replayed, so this is slow) ==="
            run_mode "$mode" "$NCU" --profile-from-start off --target-processes all \
                --section SpeedOfLight --section LaunchStats --section MemoryWorkloadAnalysis \
                python3 tests/profile_blackscholes.py 16777216 1 > "$OUT_DIR/$tag.txt" 2>&1
            if grep -q "ERR_NVGPUCTRPERM" "$OUT_DIR/$tag.txt"; then
                echo "  ncu refused: this cluster restricts GPU performance counters. Skipping ncu."
                break
            fi
        done
        python3 tests/summarize_ncu.py "$OUT_DIR"
    fi

    if [[ -z "$NSYS" && -z "$NCU" ]]; then
        echo
        echo "Neither nsys nor ncu was found inside the container."
        echo "Report this output; the tools then have to come from the cluster (a module) instead."
        return 1
    fi
    echo
    echo "Raw profiles in $OUT_DIR (open the .nsys-rep files in the Nsight Systems GUI for a timeline)."
}

main 2>&1 | tee "$SUMMARY"
rc=${PIPESTATUS[0]}
echo "[run_nsight] Summary saved to $SUMMARY"
exit "$rc"

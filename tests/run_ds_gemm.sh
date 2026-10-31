#!/usr/bin/env bash
# run_ds_gemm.sh — build and run the standalone double-single matmul benchmark.
#
# Run from /src/ds_experiment inside the container, or from a batch script:
#   bash tests/run_ds_gemm.sh [label]
#   sbatch --export=ALL,DS_JOB_SCRIPT=tests/run_ds_gemm.sh submit_l40s.sh
#
# Needs only nvcc and cuBLAS (both in the image); it does not use JAX, the
# plugin or the pass. Output is also written to results/<gpu>_ds-gemm_<label>.txt.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$gpu" ]] && gpu="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                          | head -1 | sed 's/^Model:[[:space:]]*//')"
gpu="$(echo "${gpu:-unknown-gpu}" | tr -c 'A-Za-z0-9\n' '-')"
mkdir -p results cpp/ds_gemm/build
OUT="results/${gpu}_ds-gemm_${1:-$(date +%Y%m%d-%H%M%S)}.txt"

NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"

{
    echo "GPU    : $gpu"
    echo "Date   : $(date -Is)"
    echo "Commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo

    echo "=== CPU check of the DS arithmetic ==="
    g++ -O2 -ffp-contract=off -o cpp/ds_gemm/build/test_ds_gemm_math \
        cpp/ds_gemm/test_ds_gemm_math.cpp || exit 1
    cpp/ds_gemm/build/test_ds_gemm_math
    echo

    echo "=== Building the GPU benchmark ==="
    "$NVCC" -O3 -std=c++17 -arch=native -lineinfo -Xptxas -v \
        -o cpp/ds_gemm/build/ds_gemm_bench cpp/ds_gemm/ds_gemm_bench.cu -lcublas 2>&1 \
        | grep -E "error|warning|registers|spill" || true
    [[ -x cpp/ds_gemm/build/ds_gemm_bench ]] || { echo "Build failed."; exit 1; }
    echo

    echo "=== Benchmark ==="
    cpp/ds_gemm/build/ds_gemm_bench 256 512 1024 2048 4096
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[run_ds_gemm] Output saved to $OUT"
exit "$rc"

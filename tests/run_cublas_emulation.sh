#!/usr/bin/env bash
# run_cublas_emulation.sh — DS matmul kernel vs native cuBLAS DGEMM vs cuBLAS
# FP64 emulation (Ozaki scheme, CUDA 13.0 Update 2+), standalone, one node.
#
# Needs a CUDA 13.0.2-or-newer toolkit (nvcc + cuBLAS); our main image has
# CUDA 12.9, which lacks the emulation. Run it in a stock CUDA 13 container
# through the host wrapper:
#   bash tests/run_cublas_emulation_host.sh            # on Stravinsky, from the repo root
# or directly, inside any CUDA 13.0.2+ devel container with the repo at cwd:
#   bash tests/run_cublas_emulation.sh [label]
#
# Runs the benchmark twice: cuBLAS's default emulation strategy, and
# CUBLAS_EMULATION_STRATEGY=eager (emulate whenever possible). In both, cuBLAS
# keeps FP64-equivalent accuracy by default and may fall back to native FP64.
# Output: results/<gpu>_cublas-emu_<label>.txt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$gpu" ]] && gpu="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                          | head -1 | sed 's/^Model:[[:space:]]*//')"
gpu="$(echo "${gpu:-unknown-gpu}" | tr -c 'A-Za-z0-9\n' '-')"
label="${1:-$(date +%Y%m%d-%H%M%S)}"
BUILD=cpp/ds_gemm/build/emu          # separate from the CUDA 12.9 build
mkdir -p results "$BUILD"
OUT="results/${gpu}_cublas-emu_${label}.txt"
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"

{
    echo "GPU    : $gpu"
    echo "Date   : $(date -Is)"
    echo "Commit : $(git describe --always --dirty 2>/dev/null || echo unknown)"
    echo "nvcc   : $("$NVCC" --version | tail -2 | head -1)"
    grep -h -m1 "Driver Version" /proc/driver/nvidia/version 2>/dev/null || true
    echo

    "$NVCC" -O3 -std=c++17 -arch=native -DWITH_FP64_EMULATION \
        -o "$BUILD/ds_gemm_bench_emu" cpp/ds_gemm/ds_gemm_bench.cu -lcublas \
        || { echo "Build failed (needs cuBLAS from CUDA 13.0 Update 2 or newer)."; exit 1; }

    echo "=== cuBLAS default emulation strategy ==="
    stdbuf -oL "$BUILD/ds_gemm_bench_emu" 1024 2048 4096
    echo
    echo "=== CUBLAS_EMULATION_STRATEGY=eager ==="
    CUBLAS_EMULATION_STRATEGY=eager stdbuf -oL "$BUILD/ds_gemm_bench_emu" 1024 2048 4096
    echo
    echo "Reading the emulated rows: DGEMM/emu > 1 means emulation is faster than native"
    echo "DGEMM; 'emu vs DS' > 1 means emulation is faster than the DS kernel. A DGEMM/emu"
    echo "near 1.00x means cuBLAS fell back to native FP64."
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[run_cublas_emulation] Output saved to $OUT"
exit "$rc"

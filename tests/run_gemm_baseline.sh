#!/usr/bin/env bash
# run_gemm_baseline.sh — the native f64 matmul baselines, measured side by side
# on one node, plus the GPU's own FP64 figures.
#
# Answers:
#   1. What FP32:FP64 ratio does the driver report for this GPU, and what
#      f64 and f32 GEMM rates do we actually measure (SGEMM vs DGEMM)?
#   2. Why does JAX/XLA's f64 matmul (12.6 ms at 2048^2 on L40S) differ from
#      cuBLAS DGEMM in our standalone benchmark (15.2 ms)? Both are timed here
#      in the same job, and XLA's compiled program is dumped to show which
#      GEMM routine it calls, with cuBLASLt on and off.
#   3. The standalone DS kernel vs DGEMM table (same as run_ds_gemm.sh).
#
# Run from /src/ds_experiment inside the container, or from a batch script:
#   bash tests/run_gemm_baseline.sh [label]
#   sbatch --export=ALL,DS_JOB_SCRIPT=tests/run_gemm_baseline.sh submit_l40s.sh
#   DS_JOB_SCRIPT=tests/run_gemm_baseline.sh bash submit_blackwell.sh   # Stravinsky
#
# Output: results/<gpu>_gemm-baseline_<label>.txt (and the run_ds_gemm.sh file).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$gpu" ]] && gpu="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                          | head -1 | sed 's/^Model:[[:space:]]*//')"
gpu="$(echo "${gpu:-unknown-gpu}" | tr -c 'A-Za-z0-9\n' '-')"
label="${1:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p results cpp/ds_gemm/build
OUT="results/${gpu}_gemm-baseline_${label}.txt"
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
PLUGIN="cuda:$PWD/pjrt_plugin/build/libds_pjrt_plugin.so"

{
    bash tests/run_metadata.sh
    nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.sm,power.limit,persistence_mode \
        --format=csv 2>/dev/null
    echo

    echo "=== 1. Driver-reported device properties ==="
    cat > cpp/ds_gemm/build/devprops.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int ratio = 0, clk = 0;
    cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, 0);
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("name                        %s\n", p.name);
    printf("compute capability          %d.%d\n", p.major, p.minor);
    printf("SMs                         %d\n", p.multiProcessorCount);
    printf("max SM clock (driver)       %.0f MHz\n", clk / 1000.0);
    printf("FP32:FP64 perf ratio        %d:1   (cudaDevAttrSingleToDoublePrecisionPerfRatio)\n", ratio);
    return 0;
}
EOF
    "$NVCC" -o cpp/ds_gemm/build/devprops cpp/ds_gemm/build/devprops.cu && cpp/ds_gemm/build/devprops
    echo

    echo "=== 2. Standalone: cuBLAS SGEMM / DGEMM / DS kernel (run_ds_gemm.sh) ==="
    bash tests/run_ds_gemm.sh "baseline-${label}" | sed -n '/=== Benchmark ===/,$p'
    echo

    echo "=== 3. JAX/XLA native f64 matmul, same node (plugin bypassed) ==="
    bash ds_setup.sh > /dev/null 2>&1 || { echo "ds_setup.sh failed"; exit 1; }
    for lt in default true false; do
        flags=""
        [[ "$lt" != default ]] && flags="--xla_gpu_enable_cublaslt=$lt"
        dump="cpp/ds_gemm/build/xla_dump_lt_$lt"
        rm -rf "$dump"
        echo "--- XLA_FLAGS: ${flags:-(none)}"
        DS_BYPASS=1 JAX_ENABLE_X64=1 PJRT_NAMES_AND_LIBRARY_PATHS="$PLUGIN" \
        XLA_FLAGS="$flags --xla_dump_to=$dump --xla_dump_hlo_as_text" python3 - 2>/dev/null <<'EOF'
import time, numpy as np, jax, jax.numpy as jnp
f = jax.jit(lambda a, b: a @ b)
for n in (1024, 2048, 4096):
    rng = np.random.default_rng(n)
    a = jnp.array(rng.standard_normal((n, n))); b = jnp.array(rng.standard_normal((n, n)))
    for _ in range(3): f(a, b).block_until_ready()
    ts = []
    for _ in range(15):
        t = time.perf_counter(); f(a, b).block_until_ready(); ts.append(time.perf_counter() - t)
    ms = float(np.median(ts)) * 1e3
    print(f"  n={n:5d}  f64 matmul {ms:9.3f} ms   {2*n**3/ms/1e9:7.3f} TFLOP/s   (dtype {a.dtype})")
EOF
        # What XLA runs for each matmul: the executed kernel (thunk sequence)
        # and, if autotuned, the chosen Triton tiling or cuBLAS algorithm.
        for m in "$dump"/*jit__lambda.thunk_sequence.txt; do
            [[ -f "$m" ]] || continue
            base="${m%.thunk_sequence.txt}"
            shape="$(grep -o 'f64\[[0-9]*,[0-9]*\]' "$base".sm_*after_optimizations.txt 2>/dev/null | head -1)"
            kern="$(grep -o 'kernel=[^,]*\|kGemm[^ ]*\|kCublas[^ ]*' "$m" | tr '\n' ' ')"
            tune="$(tr -d ' ' < "$base".autotune_results.pbtxt 2>/dev/null \
                    | grep -o -e '^triton' -e '^gemm' -e 'block_[mnk]:[0-9]*' -e 'num_stages:[0-9]*' -e 'num_warps:[0-9]*' -e 'algorithm:[0-9-]*' | tr '\n' ' ')"
            echo "  XLA ran for ${shape:-?}: ${kern:-?}  autotune: ${tune:-none}"
        done
    done
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[run_gemm_baseline] Output saved to $OUT"
exit "$rc"

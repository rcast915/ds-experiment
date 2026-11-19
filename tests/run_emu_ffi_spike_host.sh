#!/usr/bin/env bash
# run_emu_ffi_spike_host.sh — feasibility spike: cuBLAS FP64 emulation from a
# jitted JAX program through our own FFI call. From the repo root on the GPU node:
#
#   bash tests/run_emu_ffi_spike_host.sh [label]
#
# Builds cpp/emu_gemm/emu_dgemm_ffi.cc in the CUDA 13 devel image
# (cuda13-devel.sif, pulled if missing), against JAX's FFI headers from
# .venv-cuda13 (made by tests/run_xla_emulation_probe_host.sh), then runs
# tests/spike_emu_ffi.py twice in python312.sif: XLA preallocation on, then off.
# At run time the library uses the cuBLAS from JAX's pip wheel.
# Output: results/<gpu>_emu-ffi-spike_<label>.txt

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

CT="$(command -v apptainer || command -v singularity)" \
    || { echo "Neither apptainer nor singularity is installed on $(hostname)."; exit 1; }
export APPTAINER_CACHEDIR=/tmp/$USER-apptainer-cache APPTAINER_TMPDIR=/tmp/$USER-apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" results cpp/emu_gemm/build
for need in python312.sif .venv-cuda13/bin/python; do
    [ -e "$need" ] || { echo "Missing $need: run tests/run_xla_emulation_probe_host.sh first."; exit 1; }
done
[ -f cuda13-devel.sif ] || "$CT" pull cuda13-devel.sif docker://nvidia/cuda:13.0.2-devel-ubuntu24.04 || exit 1

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
gpu="$(echo "${gpu:-unknown-gpu}" | tr -c 'A-Za-z0-9\n' '-')"
OUT="results/${gpu}_emu-ffi-spike_${1:-$(hostname -s)-$(date +%m%d-%H%M)}.txt"
B="--bind $(pwd):/src/ds_experiment"
PY=".venv-cuda13/bin/python"

{
    echo "GPU    : $gpu"
    echo "Date   : $(date -Is)"
    echo "Host   : $(hostname)"
    echo "Commit : $(git describe --always --dirty --abbrev=12 2>/dev/null || echo unknown)"
    echo

    INC="$("$CT" exec $B python312.sif bash -c "cd /src/ds_experiment && $PY -c 'import jax.ffi; print(jax.ffi.include_dir())'")" \
        || { echo "Could not locate JAX FFI headers."; exit 1; }
    echo "Building emu_dgemm_ffi.so (FFI headers: $INC)"
    # static libstdc++: built on Ubuntu 24.04, run in the Debian-based Python image
    "$CT" exec $B cuda13-devel.sif bash -c "
        cd /src/ds_experiment &&
        g++ -O2 -std=c++17 -shared -fPIC -static-libstdc++ -static-libgcc \
            -I'$INC' -I/usr/local/cuda/include cpp/emu_gemm/emu_dgemm_ffi.cc \
            -L/usr/local/cuda/lib64 -lcublas -o cpp/emu_gemm/build/libemu_dgemm_ffi.so
    " || { echo "Build failed."; exit 1; }

    for pre in true false; do
        echo
        echo "=================== XLA_PYTHON_CLIENT_PREALLOCATE=$pre ==================="
        "$CT" exec --nv $B python312.sif bash -c "
            cd /src/ds_experiment
            CUB=\$(dirname \$(find .venv-cuda13 -name 'libcublas.so.13*' | head -1))
            LD_LIBRARY_PATH=\$PWD/\$CUB:\${LD_LIBRARY_PATH:-} XLA_PYTHON_CLIENT_PREALLOCATE=$pre \
                $PY -u tests/spike_emu_ffi.py cpp/emu_gemm/build/libemu_dgemm_ffi.so
        "
    done
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[emu_ffi_spike] Output saved to $OUT"
exit "$rc"

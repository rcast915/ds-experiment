#!/usr/bin/env bash
# run_xla_emulation_probe_host.sh — Step 1 go/no-go: do JAX f64 matmuls reach
# cuBLAS FP64 emulation? Runs tests/probe_xla_emulation.py with jax[cuda13]
# (its own pip-wheel CUDA 13 libraries, no toolkit needed) in a stock Python
# container. No DS plugin involved. From the repo root on the GPU node:
#
#   bash tests/run_xla_emulation_probe_host.sh [label]
#
# First run pulls python312.sif and creates .venv-cuda13/ in the repo
# (a few minutes); later runs reuse both. JAX_PIP overrides the install spec.
# Output: results/<gpu>_xla-emu-probe_<label>.txt

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

SIF="python312.sif"
JAX_PIP="${JAX_PIP:-jax[cuda13]}"
CT="$(command -v apptainer || command -v singularity)" \
    || { echo "Neither apptainer nor singularity is installed on $(hostname)."; exit 1; }
export APPTAINER_CACHEDIR=/tmp/$USER-apptainer-cache APPTAINER_TMPDIR=/tmp/$USER-apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" results
[ -f "$SIF" ] || "$CT" pull "$SIF" docker://python:3.12-slim || exit 1

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
gpu="$(echo "${gpu:-unknown-gpu}" | tr -c 'A-Za-z0-9\n' '-')"
OUT="results/${gpu}_xla-emu-probe_${1:-$(hostname -s)-$(date +%m%d-%H%M)}.txt"

{
    echo "GPU    : $gpu"
    echo "Date   : $(date -Is)"
    echo "Host   : $(hostname)"
    echo "Commit : $(git describe --always --dirty --abbrev=12 2>/dev/null || echo unknown)"
    echo "Driver : $(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -1)"
    echo
    "$CT" exec --nv --bind "$(pwd)":/src/ds_experiment "$SIF" bash -c "
        set -e
        cd /src/ds_experiment
        if [ ! -x .venv-cuda13/bin/python ]; then
            python3 -m venv .venv-cuda13
            .venv-cuda13/bin/pip install -q '$JAX_PIP' numpy
        fi
        .venv-cuda13/bin/python -u tests/probe_xla_emulation.py
    "
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
echo "[xla_emulation_probe] Output saved to $OUT"
exit "$rc"

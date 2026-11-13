#!/usr/bin/env bash
# run_cublas_emulation_host.sh — run tests/run_cublas_emulation.sh in a stock
# NVIDIA CUDA 13 devel container (Apptainer/Singularity), from the repo root.
#
#   bash tests/run_cublas_emulation_host.sh [label]
#   CUDA13_IMAGE=docker://nvidia/cuda:13.0.2-devel-ubuntu24.04 bash tests/run_cublas_emulation_host.sh
#
# The image is pulled once to cuda13-devel.sif in the repo directory. The host
# driver must support the image's CUDA version (Stravinsky: driver 610.43.02,
# CUDA 13.3).

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

IMAGE="${CUDA13_IMAGE:-docker://nvidia/cuda:13.0.2-devel-ubuntu24.04}"
SIF="cuda13-devel.sif"
CT="$(command -v apptainer || command -v singularity)" \
    || { echo "Neither apptainer nor singularity is installed on $(hostname)."; exit 1; }

export APPTAINER_CACHEDIR=/tmp/$USER-apptainer-cache
export APPTAINER_TMPDIR=/tmp/$USER-apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [ ! -f "$SIF" ]; then
    echo "Pulling $IMAGE ..."
    "$CT" pull "$SIF" "$IMAGE" || exit 1
fi

"$CT" exec --nv --bind "$(pwd)":/src/ds_experiment "$SIF" bash -c "
    cd /src/ds_experiment
    bash tests/run_cublas_emulation.sh '${1:-$(hostname -s)-$(date +%m%d-%H%M)}'
"

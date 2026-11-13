#!/usr/bin/env bash
# run_metadata.sh — print the reproducibility header every runner writes at the
# top of its results file. Run from the repo root inside the container.
#
#   bash tests/run_metadata.sh
#
# DS_IMAGE_ID, if set by the batch script (submit_*.sh), names the container
# image and its checksum; otherwise the header says the image is unrecorded.

py() { python3 -c "$1" 2>/dev/null || echo "unavailable"; }

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$gpu" ]] && gpu="$(grep -h -m1 '^Model:' /proc/driver/nvidia/gpus/*/information 2>/dev/null \
                          | head -1 | sed 's/^Model:[[:space:]]*//')"

echo "GPU      : ${gpu:-unknown}"
echo "Date     : $(date -Is)"
echo "Host     : $(hostname)"
echo "Commit   : $(git describe --always --dirty --abbrev=12 2>/dev/null || echo unknown)"
if ! git diff --quiet HEAD 2>/dev/null; then
    echo "WARNING  : uncommitted changes in: $(git diff --name-only HEAD | tr '\n' ' ')"
fi
echo "Driver   : $(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -1)"
echo "CUDA     : $( (command -v nvcc >/dev/null && nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p') || echo unknown) (toolkit in container)"
echo "Python   : $(python3 --version 2>&1 | cut -d' ' -f2)"
echo "JAX      : jax $(py 'import jax; print(jax.__version__)'), jaxlib $(py 'import jaxlib; print(jaxlib.__version__)')"
echo "NumPy    : $(py 'import numpy; print(numpy.__version__)')"
echo "cuBLAS   : $(py 'import importlib.metadata as m; print(m.version("nvidia-cublas-cu12"))') (pip wheel used by JAX)"
echo "cuDNN    : $(py 'import importlib.metadata as m; print(m.version("nvidia-cudnn-cu12"))')"
echo "Image    : ${DS_IMAGE_ID:-unrecorded}"
echo

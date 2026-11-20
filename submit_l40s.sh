#!/usr/bin/env bash
#SBATCH --job-name=ds-l40s
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --time=3:00:00
#SBATCH --account=cis260064p
#SBATCH --output=logs/output/l40s_%j.out
#SBATCH --error=logs/error/l40s_%j.err
#
# Bridges-2: run the full DS suite (tests, benchmarks including
# Black-Scholes, pair-accuracy sections, per-op f64 diagnostic) on one GPU.
#
# Submit from the repo root on Bridges-2:
#
#   cd ~/ds-experiment && git pull
#   mkdir -p logs/output logs/error      # sbatch needs these to exist
#   sbatch submit_l40s.sh
#
# The #SBATCH lines above select an L40S. For another GPU type, override them
# on the command line instead of editing this file, e.g.:
#
#   sbatch --gres=gpu:v100-32:1 --job-name=ds-v100 \
#          --output=logs/output/v100_%j.out --error=logs/error/v100_%j.err \
#          submit_l40s.sh
#
# (`sinfo -p GPU-shared -o "%G"` lists the GPU types the partition offers.)
#
# Watch:    tail -f logs/output/<name>_<JOBID>.out
# Results:  results/<gpu-model>_<JOBID>.txt  (same content as the .out file's
#           test section, named by the GPU the job actually got)
#
# To run a different job script (e.g. only the FMA upper-bound benchmark):
#   sbatch --export=ALL,DS_JOB_SCRIPT=tests/run_fma_bound.sh submit_l40s.sh
#
# The job runs whatever is checked out in $PROJECT_DIR — it does not pull.

set -euo pipefail

PROJECT_DIR="$HOME/ds-experiment"
# Image tag: cuda13 (CUDA 13.0.2, JAX 0.12.1) by default; DS_TAG=latest selects
# the earlier CUDA 12.9 / JAX 0.10.1 image. Each tag gets its own .sif/sandbox.
DS_TAG="${DS_TAG:-cuda13}"
DS_IMAGE="docker://rcast915/ds-experiment:$DS_TAG"
SIF="$LOCAL/ds-experiment-$DS_TAG.sif"

echo "=== [1/3] Pulling Singularity image to \$LOCAL ==="
export APPTAINER_CACHEDIR=$LOCAL/.apptainer
export APPTAINER_TMPDIR=$LOCAL/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

singularity pull "$SIF" $DS_IMAGE

echo "=== [2/3] Building writable sandbox on \$LOCAL ==="
SANDBOX="$LOCAL/ds-sandbox-$DS_TAG"
# Built from the registry, not from the .sif: Singularity CE 4.x fails on
# the CUDA 13 image's .sif with "symlinkat /.singularity.d/runscript
# singularity: file exists". The .sif is still pulled for the checksum.
singularity build --sandbox "$SANDBOX" $DS_IMAGE

# Create Bridges-2 filesystem mount points that Apptainer expects
mkdir -p "$SANDBOX/jet" "$SANDBOX/ocean"

echo "=== [3/3] Running setup, tests and benchmarks inside container ==="
# run_full.sh rebuilds the plugin and mlir-ds-opt first (the image's own build
# artifacts are shadowed by the bind mount), and keeps going past a failing
# test step so the benchmarks still run. Its exit code is this job's.
# Record which image this run used (checksum cached next to the .sif).
if [ ! -f "$SIF.sha256" ] || [ "$SIF" -nt "$SIF.sha256" ]; then
    sha256sum "$SIF" | cut -c1-16 > "$SIF.sha256"
fi
export DS_IMAGE_ID="$DS_IMAGE, pulled $(date -r "$SIF" +%F), sif sha256 $(cat "$SIF.sha256")..."
export APPTAINERENV_DS_IMAGE_ID="$DS_IMAGE_ID" SINGULARITYENV_DS_IMAGE_ID="$DS_IMAGE_ID"

singularity exec --nv --writable \
  --bind "$PROJECT_DIR":/src/ds_experiment \
  "$SANDBOX" bash -c "
    cd /src/ds_experiment
    bash ${DS_JOB_SCRIPT:-tests/run_full.sh} '${SLURM_JOB_ID:-manual}'
  "

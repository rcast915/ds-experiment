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
# The job runs whatever is checked out in $PROJECT_DIR — it does not pull.

set -euo pipefail

PROJECT_DIR="$HOME/ds-experiment"
SIF="$LOCAL/ds-experiment.sif"

echo "=== [1/3] Pulling Singularity image to \$LOCAL ==="
export APPTAINER_CACHEDIR=$LOCAL/.apptainer
export APPTAINER_TMPDIR=$LOCAL/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

singularity pull "$SIF" docker://rcast915/ds-experiment:latest

echo "=== [2/3] Building writable sandbox on \$LOCAL ==="
SANDBOX="$LOCAL/ds-sandbox"
singularity build --sandbox "$SANDBOX" "$SIF"

# Create Bridges-2 filesystem mount points that Apptainer expects
mkdir -p "$SANDBOX/jet" "$SANDBOX/ocean"

echo "=== [3/3] Running setup, tests and benchmarks inside container ==="
# run_full.sh rebuilds the plugin and mlir-ds-opt first (the image's own build
# artifacts are shadowed by the bind mount), and keeps going past a failing
# test step so the benchmarks still run. Its exit code is this job's.
singularity exec --nv --writable \
  --bind "$PROJECT_DIR":/src/ds_experiment \
  "$SANDBOX" bash -c "
    cd /src/ds_experiment
    bash tests/run_full.sh '${SLURM_JOB_ID:-manual}'
  "

#!/bin/bash
#SBATCH --job-name=ds-blackwell
#SBATCH --output=logs/output/blackwell_%j.out
#SBATCH --error=logs/error/blackwell_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=stravinsky
##SBATCH --gres=gpu:1
#
# orthus.nic.uoregon.edu (University of Oregon NIC), partition `stravinsky`:
# run the full DS suite (tests, benchmarks including Black-Scholes,
# pair-accuracy sections, per-op f64 and matmul diagnostics).
#
# Same Slurm settings and container flow as the earlier
# run_blackwell_reeval.sh on that machine; only the payload differs.
#
# Submit from the repo root on orthus:
#
#   cd ~/ds-experiment && git pull
#   mkdir -p logs/output logs/error      # sbatch needs these to exist
#   sbatch submit_blackwell.sh
#
# Watch:    tail -f logs/output/blackwell_<JOBID>.out
# Results:  results/<gpu-model>_<JOBID>.txt
#
# The image (ds-experiment.sif) and the writable sandbox (ds-sandbox/) live in
# the repo directory and are reused if present. Delete both to force a fresh
# pull, e.g. after the Docker image has been rebuilt.
#
# The job runs whatever is checked out — it does not pull.

set -uo pipefail
mkdir -p logs/output logs/error

echo "Job started: $(date) on $(hostname)"

cd "$SLURM_SUBMIT_DIR"

# 1. Temporary directories, to avoid permission/space issues
export APPTAINER_CACHEDIR=/tmp/$USER-apptainer-cache
export APPTAINER_TMPDIR=/tmp/$USER-apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# 2. Pull the pre-built image from Docker Hub
SIF="ds-experiment.sif"
if [ ! -f "$SIF" ]; then
    echo "Pulling image from Docker Hub..."
    singularity pull "$SIF" docker://rcast915/ds-experiment:latest || exit 1
fi

# 3. Build a writable sandbox
SANDBOX="ds-sandbox"
if [ ! -d "$SANDBOX" ]; then
    echo "Building writable sandbox..."
    singularity build --sandbox "$SANDBOX" "$SIF" || exit 1
fi

# 4. Run everything inside the container. run_full.sh rebuilds the plugin and
#    mlir-ds-opt first (the image's own build artifacts are shadowed by the
#    bind mount) and keeps going past a failing step. Its exit code is the
#    job's.
echo "Starting full DS run..."
singularity exec --nv --writable \
  --bind "$(pwd)":/src/ds_experiment \
  "$SANDBOX" bash -c "
    cd /src/ds_experiment
    bash tests/run_full.sh '${SLURM_JOB_ID:-manual}'
  "
rc=$?

echo "Job finished: $(date) (exit $rc)"
exit $rc

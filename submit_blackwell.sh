#!/bin/bash
#SBATCH --job-name=ds-blackwell
#SBATCH --output=logs/output/blackwell_%j.out
#SBATCH --error=logs/error/blackwell_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# No partition here: orthus has no default GPU partition, so pass one.
#
# University of Oregon NIC (orthus.nic.uoregon.edu). Runs the full DS suite
# (tests, Black-Scholes, diagnostics), or another job script, in the
# container. Works two ways:
#
# 1. Through Slurm, for GPU nodes that are Slurm partitions:
#      godzilla   T4 (Turing)          picard   2 x A2000 (Ampere)
#
#      cd ~/ds-experiment && git pull
#      mkdir -p logs/output logs/error      # sbatch needs these to exist
#      sbatch -p godzilla submit_blackwell.sh
#      sbatch -p godzilla --dependency=singleton \
#             --export=ALL,DS_JOB_SCRIPT=tests/run_purification.sh submit_blackwell.sh
#
#    --dependency=singleton waits for the earlier job of the same name, so
#    two jobs never rebuild the pass at the same time.
#
# 2. Directly, on servers outside Slurm (Stravinsky: RTX PRO 6000 Blackwell;
#    Icarus: 3 x RTX A6000). ssh to the server, then from the repo root:
#      nohup bash submit_blackwell.sh > logs/output/direct_$(hostname -s).out 2>&1 &
#      DS_JOB_SCRIPT=tests/run_purification.sh nohup bash submit_blackwell.sh \
#            > logs/output/purif_$(hostname -s).out 2>&1 &     # after the first one ends
#
# Watch:    tail -f logs/output/<file>.out
# Results:  results/<gpu-model>_<label>.txt
#
# The image (ds-experiment-<tag>.sif) and the writable sandbox (ds-sandbox-<tag>/) live in
# the repo directory and are reused if present. Delete both to force a fresh
# pull, e.g. after the Docker image has been rebuilt. They are x86 only: do
# not use the grace / hopper (ARM) nodes.
#
# The job runs whatever is checked out — it does not pull.

set -uo pipefail
mkdir -p logs/output logs/error

echo "Job started: $(date) on $(hostname)"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"

# 1. Temporary directories, to avoid permission/space issues
export APPTAINER_CACHEDIR=/tmp/$USER-apptainer-cache
export APPTAINER_TMPDIR=/tmp/$USER-apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# 2. Pull the pre-built image from Docker Hub
# Apptainer and Singularity take the same commands; use whichever exists.
CT="$(command -v apptainer || command -v singularity)" \
    || { echo "Neither apptainer nor singularity is installed on $(hostname)."; exit 1; }

# Image tag: cuda13 (CUDA 13.0.2, JAX 0.12.1) by default; DS_TAG=latest selects
# the earlier CUDA 12.9 / JAX 0.10.1 image. Each tag gets its own .sif/sandbox.
DS_TAG="${DS_TAG:-cuda13}"
DS_IMAGE="docker://rcast915/ds-experiment:$DS_TAG"
SIF="ds-experiment-$DS_TAG.sif"
if [ ! -f "$SIF" ]; then
    echo "Pulling image from Docker Hub..."
    "$CT" pull "$SIF" $DS_IMAGE || exit 1
fi

# 3. Build a writable sandbox
SANDBOX="ds-sandbox-$DS_TAG"
if [ ! -d "$SANDBOX" ]; then
    echo "Building writable sandbox..."
    "$CT" build --sandbox "$SANDBOX" "$SIF" || exit 1
fi

# 4. Run everything inside the container. run_full.sh rebuilds the plugin and
#    mlir-ds-opt first (the image's own build artifacts are shadowed by the
#    bind mount) and keeps going past a failing step. Its exit code is the
#    job's.
# Record which image this run used (checksum cached next to the .sif).
if [ ! -f "$SIF.sha256" ] || [ "$SIF" -nt "$SIF.sha256" ]; then
    sha256sum "$SIF" | cut -c1-16 > "$SIF.sha256"
fi
export DS_IMAGE_ID="$DS_IMAGE, pulled $(date -r "$SIF" +%F), sif sha256 $(cat "$SIF.sha256")..."
export APPTAINERENV_DS_IMAGE_ID="$DS_IMAGE_ID" SINGULARITYENV_DS_IMAGE_ID="$DS_IMAGE_ID"

echo "Starting full DS run..."
"$CT" exec --nv --writable \
  --bind "$(pwd)":/src/ds_experiment \
  "$SANDBOX" bash -c "
    cd /src/ds_experiment
    bash ${DS_JOB_SCRIPT:-tests/run_full.sh} '${SLURM_JOB_ID:-$(hostname -s)-$(date +%m%d-%H%M)}'
  "
rc=$?

echo "Job finished: $(date) (exit $rc)"
exit $rc

#!/usr/bin/env bash
#SBATCH --job-name=ds-full
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/output/ds_%j.out
#SBATCH --error=logs/error/ds_%j.err
#
# Run the full DS suite on any Slurm cluster that has Apptainer/Singularity.
# Cluster-neutral sibling of submit_l40s.sh (which is Bridges-2 specific):
# the account, partition and GPU request are NOT set above — pass them on the
# command line, since they differ on every system.
#
# Submit from the repo root:
#
#   mkdir -p logs/output logs/error      # sbatch needs these to exist
#   sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> --gres=gpu:<TYPE>:1 \
#          submit_apptainer.sh
#
# To find the right values on a new cluster:
#
#   sinfo -o "%P %G %l"                                   # partitions, GPU types, time limits
#   sacctmgr -nP show assoc user=$USER format=account,partition,qos
#   module avail 2>&1 | grep -i -E "apptainer|singularity"   # may need `module load`
#
# Watch:    tail -f logs/output/ds_<JOBID>.out
# Results:  results/<gpu-model>_<JOBID>.txt
#
# Optional environment (set with `sbatch --export=ALL,NAME=value ...`):
#   DS_SCRATCH      where to put the image and sandbox (several GB).
#                   Default: node-local scratch if the cluster provides one.
#   DS_MODULES      space-separated modules to load first, e.g. "apptainer".
#   DS_MOUNT_STUBS  space-separated absolute paths to create inside the
#                   sandbox, for clusters whose Apptainer config binds
#                   filesystems the image has no mount point for
#                   (Bridges-2 needs "/jet /ocean").
#
# The job runs whatever is checked out in the submit directory — it does not pull.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$PROJECT_DIR/tests/run_full.sh" ]]; then
    echo "Submit this from the repo root (no tests/run_full.sh in $PROJECT_DIR)." >&2
    exit 1
fi

for m in ${DS_MODULES:-}; do
    module load "$m"
done

if command -v apptainer > /dev/null 2>&1; then
    CT=apptainer
elif command -v singularity > /dev/null 2>&1; then
    CT=singularity
else
    echo "Neither apptainer nor singularity is on PATH." \
         "Find the module name and resubmit with --export=ALL,DS_MODULES=<name>." >&2
    exit 1
fi

SCRATCH_BASE="${DS_SCRATCH:-${LOCAL:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}}"
WORK="$SCRATCH_BASE/ds-${SLURM_JOB_ID:-manual}"
# Image tag: cuda13 (CUDA 13.0.2, JAX 0.12.1) by default; DS_TAG=latest selects
# the earlier CUDA 12.9 / JAX 0.10.1 image. Each tag gets its own .sif/sandbox.
DS_TAG="${DS_TAG:-cuda13}"
DS_IMAGE="docker://rcast915/ds-experiment:$DS_TAG"
SIF="$WORK/ds-experiment-$DS_TAG.sif"
SANDBOX="$WORK/ds-sandbox-$DS_TAG"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

echo "Container runtime : $CT ($($CT --version 2>&1 | head -1))"
echo "Project dir       : $PROJECT_DIR"
echo "Scratch           : $WORK ($(df -h --output=avail "$WORK" | tail -1 | tr -d ' ') free)"
echo "Node              : $(hostname)"

echo "=== [1/3] Pulling image ==="
export APPTAINER_CACHEDIR="$WORK/.apptainer"
export APPTAINER_TMPDIR="$WORK/.apptainer/tmp"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_TMPDIR"

"$CT" pull "$SIF" $DS_IMAGE

echo "=== [2/3] Building writable sandbox ==="
# ds_setup.sh renames JAX's CUDA plugin directories inside the image, so the
# container filesystem has to be writable.
"$CT" build --sandbox "$SANDBOX" "$SIF"

for d in ${DS_MOUNT_STUBS:-}; do
    mkdir -p "$SANDBOX$d"
done

# Record which image this run used (checksum cached next to the .sif).
if [ ! -f "$SIF.sha256" ] || [ "$SIF" -nt "$SIF.sha256" ]; then
    sha256sum "$SIF" | cut -c1-16 > "$SIF.sha256"
fi
export DS_IMAGE_ID="$DS_IMAGE, pulled $(date -r "$SIF" +%F), sif sha256 $(cat "$SIF.sha256")..."
export APPTAINERENV_DS_IMAGE_ID="$DS_IMAGE_ID" SINGULARITYENV_DS_IMAGE_ID="$DS_IMAGE_ID"

echo "=== [3/3] Running setup, tests and benchmarks inside container ==="
"$CT" exec --nv --writable \
  --bind "$PROJECT_DIR":/src/ds_experiment \
  "$SANDBOX" bash -c "
    cd /src/ds_experiment
    bash tests/run_full.sh '${SLURM_JOB_ID:-manual}'
  "

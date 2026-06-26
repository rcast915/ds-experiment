#!/usr/bin/env bash
#SBATCH --job-name=ds-l40s
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --time=3:00:00
#SBATCH --account=cis260064p
#SBATCH --output=logs/output/l40s_%j.out
#SBATCH --error=logs/error/l40s_%j.err

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

echo "=== [3/4] Running setup and tests inside container ==="
singularity exec --nv --writable \
  --bind "$PROJECT_DIR":/src/ds_experiment \
  "$SANDBOX" bash -c '
    set -e
    cd /src/ds_experiment
    bash ds_setup.sh
    bash tests/run_tests.sh --bench
  '

echo "=== [4/4] Done ==="

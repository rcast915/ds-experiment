#!/usr/bin/env bash
#SBATCH --job-name=ds-l40s-exp2b
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --time=0:30:00
#SBATCH --account=cis260064p
#SBATCH --output=logs/output/l40s_exp2b_%j.out
#SBATCH --error=logs/error/l40s_exp2b_%j.err
#
# One-off inspection (not a benchmark): did the DS pass's TwoSum correction
# survive XLA's optimizer in the matmul epilogue, or did the algebraic
# simplifier collapse the subtract-based residual chain into a plain add?
# Measurement only -- does not modify the pass, does not time anything.
#
# Reuses paper/ds_reeval/results/NVIDIA-L40S/xla_dumps/{default,highest}/
# from the prior full run if still present (it should be -- results/ isn't
# touched by git pull, it's gitignored and lives on the bind-mounted host
# filesystem, not the ephemeral sandbox); regenerates by re-running the
# matmul once per mode otherwise.
#
# Submit from the repo root: cd ~/ds-experiment && sbatch submit_l40s_exp2b.sh

set -uo pipefail

PROJECT_DIR="$HOME/ds-experiment"
SIF="$LOCAL/ds-experiment.sif"

echo "=== [1/3] Pulling latest code ==="
git -C "$PROJECT_DIR" pull

echo "=== [2/3] Pulling Singularity image + building sandbox ==="
export APPTAINER_CACHEDIR=$LOCAL/.apptainer
export APPTAINER_TMPDIR=$LOCAL/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
singularity pull "$SIF" docker://rcast915/ds-experiment:latest
SANDBOX="$LOCAL/ds-sandbox"
singularity build --sandbox "$SANDBOX" "$SIF"
mkdir -p "$SANDBOX/jet" "$SANDBOX/ocean"

echo "=== [3/3] Running exp2b inside container ==="
singularity exec --nv --writable \
  --bind "$PROJECT_DIR":/src/ds_experiment \
  "$SANDBOX" bash -c '
    set -e
    cd /src/ds_experiment
    bash ds_setup.sh
    cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" > /dev/null
    ninja -C stablehlo_pass/build

    cd paper/ds_reeval
    export PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so"
    export JAX_ENABLE_X64=1

    python3 exp2b_fused_add.py --precision default
    python3 exp2b_fused_add.py --precision highest
    python3 report.py
  '

echo "=== Done. Check paper/ds_reeval/REPORT.md for the Experiment 2b section ==="

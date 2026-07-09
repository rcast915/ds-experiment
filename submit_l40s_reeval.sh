#!/usr/bin/env bash
#SBATCH --job-name=ds-l40s-reeval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --time=3:00:00
#SBATCH --account=cis260064p
#SBATCH --output=logs/output/l40s_reeval_%j.out
#SBATCH --error=logs/error/l40s_reeval_%j.err
#
# Sibling of submit_l40s.sh -- same image pull / sandbox / bind-mount /
# rebuild flow (including the /jet and /ocean mount-point fix and the
# writable-sandbox fix already proven out there), but runs the ds_reeval
# suite instead of tests/run_tests.sh --bench.
#
# Submit from the repo root on Bridges-2 (same convention as submit_l40s.sh):
#   cd ~/ds-experiment && sbatch submit_l40s_reeval.sh
# Watch progress:
#   tail -f logs/output/l40s_reeval_<JOBID>.out
# Results land in paper/ds_reeval/results/<gpu>/ and
# paper/ds_reeval/REPORT.md (plus smoke_out/ and SMOKE_REPORT.md from the
# smoke pass that runs first as a sanity gate).

set -uo pipefail
# Deliberately NOT `set -e` for the whole script: an individual ds_reeval
# experiment failing should not prevent report.py from running afterward
# to show whatever *did* succeed (report.py degrades gracefully over
# partial results by design). Steps where failure really should stop
# everything (repo pull, container setup, pass rebuild, the structural
# test) are inside the inner `bash -c '... set -e ...'` block below, which
# aborts on the first failure among those specifically.

PROJECT_DIR="$HOME/ds-experiment"
SIF="$LOCAL/ds-experiment.sif"

echo "=== [1/4] Pulling latest code (needs the DS_RETURN_PAIRS commit) ==="
git -C "$PROJECT_DIR" pull

echo "=== [2/4] Pulling Singularity image to \$LOCAL ==="
export APPTAINER_CACHEDIR=$LOCAL/.apptainer
export APPTAINER_TMPDIR=$LOCAL/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

singularity pull "$SIF" docker://rcast915/ds-experiment:latest

echo "=== [3/4] Building writable sandbox on \$LOCAL ==="
SANDBOX="$LOCAL/ds-sandbox"
singularity build --sandbox "$SANDBOX" "$SIF"

# Bridges-2 filesystem mount points Apptainer expects (same fix as submit_l40s.sh)
mkdir -p "$SANDBOX/jet" "$SANDBOX/ocean"

echo "=== [4/4] Running ds_reeval inside container ==="
singularity exec --nv --writable \
  --bind "$PROJECT_DIR":/src/ds_experiment \
  "$SANDBOX" bash -c '
    set -e
    cd /src/ds_experiment
    bash ds_setup.sh

    echo "[setup] Rebuilding mlir-ds-opt (shadowed by bind mount, must rebuild)..."
    cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" > /dev/null
    ninja -C stablehlo_pass/build

    cd paper/ds_reeval

    echo "--- [a] structural test (CPU-only, must pass before spending GPU time) ---"
    python3 test_return_pairs_structural.py

    echo "--- [b] dry run (sanity-check the plan) ---"
    python3 run_all.py --dry-run

    echo "--- [c] smoke test ---"
    set +e
    python3 run_all.py --smoke
    set -e
    python3 report.py --results-dir smoke_out --out SMOKE_REPORT.md

    echo "--- [d] full run ---"
    set +e
    python3 run_all.py --image-digest "rcast915/ds-experiment:latest-pulled-$(date -I)"
    set -e
    python3 report.py
  '

echo "=== Done. Check paper/ds_reeval/REPORT.md and SMOKE_REPORT.md ==="

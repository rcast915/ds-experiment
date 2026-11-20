#!/bin/bash
#SBATCH --job-name=ds-docker-build
#SBATCH --output=logs/output/build_%j.out
#SBATCH --error=logs/error/build_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=dgx
#SBATCH --qos=punakha_dgx2_general

set -euo pipefail
mkdir -p logs/output logs/error

echo "Job started: $(date)"
module load docker/27.3.1/rootless-docker
start_rootless_docker.sh --quiet
sleep 10

cd /o_home/racastaneda3/ds_experiment
# Tag cuda13: CUDA 13.0.2 / JAX 0.12.1. The earlier CUDA 12.9 image stays on
# Docker Hub as :latest, which produced the results recorded before 2026-11-20.
docker build --network=host -t ds-experiment:cuda13 -t rcast915/ds-experiment:cuda13 .
# DS_PUSH=1 pushes to Docker Hub from this job (run `docker login` beforehand
# in an interactive session; credentials are read from ~/.docker/config.json).
if [ "${DS_PUSH:-0}" = 1 ]; then
    docker push rcast915/ds-experiment:cuda13
else
    echo "Not pushed. Push with: docker push rcast915/ds-experiment:cuda13 (or rerun with DS_PUSH=1)"
fi

echo "Build finished: $(date)"
echo "Exit code: $?"

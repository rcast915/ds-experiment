#!/bin/bash
#SBATCH --job-name=ds-docker-build
#SBATCH --output=logs/output/build_%j.out
#SBATCH --error=logs/error/build_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=dgx
#SBATCH --qos=punakha_dgx2_general

set -euo pipefail
mkdir -p logs/output logs/error

echo "Job started: $(date)"
module load docker/27.3.1/rootless-docker
start_rootless_docker.sh --quiet
sleep 10

cd /o_home/racastaneda3/ds_experiment
docker build --network=host -t ds-experiment .

echo "Build finished: $(date)"
echo "Exit code: $?"

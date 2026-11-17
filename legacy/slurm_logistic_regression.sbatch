#!/bin/bash
#SBATCH --job-name=logreg_baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logreg_%j.out
#SBATCH --error=logreg_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python/3.10 || true
uv sync --frozen --python /usr/bin/python3.10 --no-progress

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

LOGREG_CONFIG="${LOGREG_CONFIG:-legacy/configs/logistic_regression.yaml}"

echo "Running logistic-regression baseline: ${LOGREG_CONFIG}"
echo "Host: $(hostname)"
echo "Sample folder: ${WILDFIRE_SAMPLE_FOLDER:-configs/settings.py:SAMPLE_FOLDER}"
echo "SCRATCH: ${SCRATCH:-}"
echo "SLURM_TMP_DIR: ${SLURM_TMP_DIR:-}"
if [ -n "${SLURM_TMP_DIR:-}" ]; then
  df -h "${SLURM_TMP_DIR}"
fi

.venv/bin/python -u -m legacy.logistic_regression_main --config "${LOGREG_CONFIG}"

#!/bin/bash
# Train the three seeds (runs 1-3) of one architecture as a SLURM array job.
# Usage: sbatch scripts/parallel_script.sh <architecture> [config]
#SBATCH --job-name=dailywildfirecan
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-3

ARCH=${1:-unet}
CONFIG=${2:-configs/default.yaml}

module load python/3.10
source /PATH/TO/VENV/bin/activate
cd /PATH/TO/PROJECT/FOLDER

# export COMET_API_KEY=""   # only if comet.enabled is true in the config

python -m scripts.parallel_main --config "$CONFIG" --arch "$ARCH" --run_id "$SLURM_ARRAY_TASK_ID"

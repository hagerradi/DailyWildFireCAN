#!/bin/bash
# Evaluate the checkpoints of all architectures and runs on the three test splits.
# Usage: sbatch scripts/evaluate_script.sh [checkpoint_dir]
#SBATCH --job-name=dailywildfirecan_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

CHECKPOINT_DIR=${1:-.}

module load python/3.10
source /PATH/TO/VENV/bin/activate
cd /PATH/TO/PROJECT/FOLDER

python -m src.evaluate --config configs/default.yaml --checkpoint_dir "$CHECKPOINT_DIR"

#!/bin/bash
#SBATCH -J path_patch_country_capital
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

MODEL="meta-llama/Llama-3.2-1B-Instruct"

echo "=== running path-patching sweep (sender -> one receiver at a time) ==="
python src/shared_utils/path_patching.py \
    --model_name "$MODEL" \
    --d_name "country-capital" \
    --prompt_type "EP" \
    --component_type "lexical_task" \
    --save_root "output_1" \
    --project_root "$REPO_ROOT" \
    --dataset_folder "$REPO_ROOT/datasets/abstractive" \
    --k 10 \
    --threshold 0.1 \
    --remote False
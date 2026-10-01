#!/bin/bash
#SBATCH -J pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Full pipeline for one task:
#   behavior_variance -> identify_heads_nn -> shared_heads_nn -> path_patching -> plot_utils
#
# Usage (from the repo root):
#   sbatch -J pipeline_country-capital oscar/run_pipeline.sbatch country-capital
#   sbatch -J pipeline_park-country    oscar/run_pipeline.sbatch park-country all shuffle_input 5
#
# Args: TASK [POSITIONS=last] [CORRUPTION=shuffle_input] [INPUT_LENGTH=1]
# Skip finished steps with START_AT, e.g.:  START_AT=4 sbatch oscar/run_pipeline.sbatch antonym
# One-off receivers instead of the shared-LTH list (space-separated LAYER,HEAD):
# RECEIVERS="7,5" START_AT=4 sbatch -J pp_antonym_L7H5 oscar/run_pipeline.sbatch antonym all shuffle_input

set -euo pipefail
mkdir -p logs

TASK="${1:?usage: sbatch oscar/run_pipeline.sbatch TASK [POSITIONS] [CORRUPTION] [INPUT_LENGTH]}"
POSITIONS="${2:-last}"          # last | all
CORRUPTION="${3:-shuffle_input}" # zs | shuffle_input | shuffle_output (zs only with POSITIONS=last)
INPUT_LENGTH="${4:-1}"           # used only with POSITIONS=all (e.g. 5 for park-country)
START_AT="${START_AT:-1}"
RECEIVER_ARGS=()
if [[ -n "${RECEIVERS:-}" ]]; then RECEIVER_ARGS=(--receivers $RECEIVERS); fi

MODEL="meta-llama/Llama-3.2-1B-Instruct"
PROMPT_TYPE="EP"
EXP_SIZE=300   # path patching only: max prompts; tasks with fewer use all they have
SEED=42
K=10
THRESHOLD=0.1
BATCH_SIZE=8

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

MODEL_SHORT="${MODEL##*/}"
COMMON=(--model_name "$MODEL" --d_name "$TASK" --prompt_type "$PROMPT_TYPE"
        --save_root output --project_root "$REPO_ROOT" --dataset_folder datasets/abstractive)

echo "=== $TASK | positions=$POSITIONS corruption=$CORRUPTION input_length=$INPUT_LENGTH | start at step $START_AT ==="

if (( START_AT <= 1 )); then
    echo "=== [1/5] behavior_variance ==="
    # every task shares one behavior JSON (read-modify-write), so serialize this step across jobs
    mkdir -p "output/$MODEL_SHORT/across_tasks/Behavior"
    flock "output/$MODEL_SHORT/across_tasks/Behavior/.lock" \
        python src/behavior_variance.py "${COMMON[@]}" --batch_size 20
fi

if (( START_AT <= 2 )); then
    echo "=== [2/5] identify_heads_nn ==="
    # original identification: first --exp_size correct prompts (script default, 100) per shot count
    python src/identify_heads_nn.py "${COMMON[@]}" --component_type lexical_task \
        --k "$K" --batch_size 20
fi

if (( START_AT <= 3 )); then
    echo "=== [3/5] shared_heads_nn ==="
    python src/shared_heads_nn.py "${COMMON[@]}" --component_type lexical_task \
        --k "$K" --threshold "$THRESHOLD"
fi

if (( START_AT <= 4 )); then
    echo "=== [4/5] path_patching ==="
    python src/shared_utils/path_patching.py "${COMMON[@]}" --component_type lexical_task \
        --k "$K" --threshold "$THRESHOLD" --corruption_type "$CORRUPTION" \
        --positions "$POSITIONS" --input_length "$INPUT_LENGTH" \
        --exp_size "$EXP_SIZE" --seed "$SEED" --batch_size "$BATCH_SIZE" "${RECEIVER_ARGS[@]}"
fi

if (( START_AT <= 5 )); then
    echo "=== [5/5] plot_utils ==="
    # must match path_patching.py's save folder naming
    if [[ "$CORRUPTION" == "zs" ]]; then SUBDIR="zs"; else SUBDIR="${CORRUPTION}_seed${SEED}"; fi
    if [[ "$POSITIONS" == "all" ]]; then SUBDIR+="_allpos"; else SUBDIR+="_lastpos"; fi
    python src/shared_utils/plot_utils.py --model_name "$MODEL" --d_name "$TASK" \
        --prompt_type "$PROMPT_TYPE" --save_root output \
        --subdir "path_patching_per_receiver/$SUBDIR"
fi

echo "=== done: $TASK ==="

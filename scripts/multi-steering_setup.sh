#!/bin/bash
#SBATCH --job-name=multi-steering-clipping
#SBATCH --partition=granite-gpu-guest
#SBATCH --account=phillipsj
#SBATCH --qos=granite-gpu-guest

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:a800:1

#SBATCH --time=24:00:00
#SBATCH --array=0-13%4

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# -------------------------
# Environment setup
# -------------------------
source /uufs/chpc.utah.edu/sys/installdir/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate steering

set -euo pipefail  # stop the whole chain if any stage fails — later stages depend on earlier outputs

source ~/.hf_token
# -------------------------
# Run config
# -------------------------
DATE="$(date +%-m-%-d-%Y)"
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"
LAYER=13
BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH myopic-reward survival-instinct power-seeking-inclination wealth-seeking-inclination)

ACTIVATIONS_NAME="${DATE}_$(IFS=_; echo "${BEHAVIORS[*]}")"
ALPHA_TRAIN_NAME="${DATE}_alpha-iterative_training_experiment_caa"
PARAM_TRAIN_NAME="${DATE}_parameterized_training_experiment_caa"
INTERVAL_MAP_NAME="${DATE}_interval_map"
EXPERIMENT_NAME="${DATE}_$(IFS=_; echo "${BEHAVIORS[*]}")_multi_attribute_experiment"

echo "=== Stage 1: compute activations ==="
python -u ../experiments/new_get_activations.py "$MODEL_PATH" \
--behaviors "${BEHAVIORS[@]}" \
--dataset-subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--save-name "$ACTIVATIONS_NAME"

echo "=== Stage 2: single-behavior alpha sweep (alpha-iterative) ==="
python -u ../experiments/train_single_behavior.py \
--model_path "$MODEL_PATH" \
--activations_name "$ACTIVATIONS_NAME" \
--dataset_subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--steering_type alpha-iterative \
--save_name "$ALPHA_TRAIN_NAME"

echo "=== Stage 2: single-behavior alpha sweep (parameterized) ==="
python -u ../experiments/train_single_behavior.py \
--model_path "$MODEL_PATH" \
--activations_name "$ACTIVATIONS_NAME" \
--dataset_subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--steering_type parameterized \
--save_name "$PARAM_TRAIN_NAME"

echo "=== Stage 3: fit intervals ==="
python -u ../experiments/compute_intervals.py \
--training_experiments "alpha-iterative:${ALPHA_TRAIN_NAME}" "parameterized:${PARAM_TRAIN_NAME}" \
--behaviors "${BEHAVIORS[@]}" \
--layer "$LAYER" \
--gamma 0.1 \
--save_name "$INTERVAL_MAP_NAME"

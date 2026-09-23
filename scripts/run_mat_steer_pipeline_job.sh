#!/bin/bash
#SBATCH --job-name=mat_steer_pipeline
#SBATCH --partition=utahdb-gpu-np
#SBATCH --account=utahdb-gpu-np

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --time=96:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# -------------------------
# Environment setup
# -------------------------
set -eo pipefail

set +u
source /uufs/chpc.utah.edu/sys/installdir/r8/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/general/vast/u1591141/conda/envs/steering
set -u

python --version
which python

source ~/.hf_token

# -------------------------
DATE="$(date +%-m-%-d-%Y)"
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3.1-8B-Instruct"
LAYER=14

DATASET_SUBFOLDER="mat_steer"
BEHAVIORS=(bbq truthfulqa toxigen)
TARGET_CLASSES=(bbq truthfulqa toxigen)

ACTIVATIONS_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}")"
ALPHA_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_alpha-iterative_training_experiment")"
PARAM_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_parameterized_training_experiment")"
INTERVAL_MAP_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_interval_map")"

# echo "=== Stage 1: compute activations ==="
# python -u ../experiments/new_get_activations.py "$MODEL_PATH" \
#   --behaviors "${BEHAVIORS[@]}" \
#   --dataset-subfolder "$DATASET_SUBFOLDER" \
#   --layer "$LAYER" \
#   --save-name "$ACTIVATIONS_NAME"

# echo "=== Stage 2a: alpha-iterative single-behavior sweep ==="
# python -u ../experiments/train_single_behavior.py \
#   --model_path "$MODEL_PATH" \
#   --activations_name "$ACTIVATIONS_NAME" \
#   --dataset_subfolder "$DATASET_SUBFOLDER" \
#   --layer "$LAYER" \
#   --steering_type alpha-iterative \
#   --save_name "$ALPHA_TRAIN_NAME"

echo "=== Stage 2b: parameterized single-behavior sweep ==="
python -u ../experiments/train_single_behavior.py \
  --model_path "$MODEL_PATH" \
  --activations_name "$ACTIVATIONS_NAME" \
  --dataset_subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --steering_type parameterized \
  --save_name "$PARAM_TRAIN_NAME"

echo "=== Stage 3: fit intervals ==="
python -u ../experiments/compute_intervals.py \
  --training_experiments \
    "alpha-iterative:${ALPHA_TRAIN_NAME}" \
    "parameterized:${PARAM_TRAIN_NAME}" \
  --behaviors "${BEHAVIORS[@]}" \
  --layer "$LAYER" \
  --gamma 0.1 \
  --save_name "$INTERVAL_MAP_NAME"

echo "=== Stage 4: joint steering at normalized alpha=1 ==="
EXPERIMENT_NAME="${DATE}_mat_multi_attribute_alpha_one"
python -u ../experiments/run_multi_attribute_alpha_one.py \
  --model_path "$MODEL_PATH" \
  --activations_name "$ACTIVATIONS_NAME" \
  --dataset_subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --target_classes "${TARGET_CLASSES[@]}" \
  --test_behaviors "${BEHAVIORS[@]}" \
  --interval_map_name "$INTERVAL_MAP_NAME" \
  --alpha --parameterized \
  --save_name "$EXPERIMENT_NAME"

echo "=== Done ==="

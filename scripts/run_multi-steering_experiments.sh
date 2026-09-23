#!/bin/bash
#SBATCH --job-name=multi_steering_pipeline
#SBATCH --partition=utahdb-gpu-np
#SBATCH --account=utahdb-gpu-np

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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
DATASET_SUBFOLDER="agreeable_sycophancy"
LAYER=13
BEHAVIORS=(sycophancy agreeableness)
TEST_BEHAVIOR1=sycophancy   # which behavior in BEHAVIORS to sweep alpha/evaluate on in stage 4
TEST_BEHAVIOR2=agreeableness

ACTIVATIONS_NAME="${DATE}_$(IFS=_; echo "${BEHAVIORS[*]}")"
ALPHA_TRAIN_NAME="${DATE}_alpha-iterative_training_experiment"
PARAM_TRAIN_NAME="${DATE}_parameterized_training_experiment"
INTERVAL_MAP_NAME="${DATE}_interval_map"
EXPERIMENT_NAME="${DATE}_$(IFS=_; echo "${BEHAVIORS[*]}")_multi_attribute_experiment"


echo "=== Stage 4a: multi-attribute experiment behavior 1==="
python -u ../experiments/run_multi_attribute_experiment.py \
--model_path "$MODEL_PATH" \
--activations_name "$ACTIVATIONS_NAME" \
--dataset_subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--target_classes "${BEHAVIORS[@]}" \
--test_behaviors "$TEST_BEHAVIOR1" \
--interval_map_name "$INTERVAL_MAP_NAME" \
--alpha --parameterized \
--save_name "$EXPERIMENT_NAME"

echo "=== Stage 4b: multi-attribute experiment behavior 2==="
python -u ../experiments/run_multi_attribute_experiment.py \
--model_path "$MODEL_PATH" \
--activations_name "$ACTIVATIONS_NAME" \
--dataset_subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--target_classes "${BEHAVIORS[@]}" \
--test_behaviors "$TEST_BEHAVIOR2" \
--interval_map_name "$INTERVAL_MAP_NAME" \
--alpha --parameterized \
--save_name "$EXPERIMENT_NAME"

echo "=== Done. Final experiment: results/logit_results/${EXPERIMENT_NAME}.pkl ==="

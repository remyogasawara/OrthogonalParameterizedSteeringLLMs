#!/bin/bash
#SBATCH --job-name=
#SBATCH --partition=
#SBATCH --account=

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --time=96:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# -------------------------
# Environment setup
# -------------------------
source 
conda activate steering

set -euo pipefail  # stop the whole chain if any stage fails — later stages depend on earlier outputs

source ~/.hf_token
# -------------------------
# Run config
# -------------------------
DATE="$(date +%-m-%-d-%Y)"

MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="Llama-2-7b-chat-hf" 
LAYER=13

DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"

BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH myopic-reward survival-instinct power-seeking-inclination wealth-seeking-inclination)
TEST_BEHAVIOR1=coordinate-other-ais
TEST_BEHAVIOR2=power-seeking-inclination


ACTIVATIONS_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}")"
ALPHA_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_alpha-iterative_training_experiment_large_intervals")"
PARAM_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_parameterized_training_experiment_large_intervals")"
INTERVAL_MAP_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_large_interval_map")"

EXPERIMENT_NAME1="${DATE}_$(IFS=_; echo "${TEST_BEHAVIOR1}_${TEST_BEHAVIOR2}")_multi_attribute_experiment"
EXPERIMENT_NAME2="${DATE}_$(IFS=_; echo "${TEST_BEHAVIOR2}_${TEST_BEHAVIOR1}")_multi_attribute_experiment"

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
--alpha-min -2.0 \
--alpha-max 2.0 \
--alpha-step 0.25 \
--save_name "$ALPHA_TRAIN_NAME"

echo "=== Stage 2: single-behavior alpha sweep (parameterized) ==="
python -u ../experiments/train_single_behavior.py \
--model_path "$MODEL_PATH" \
--activations_name "$ACTIVATIONS_NAME" \
--dataset_subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--steering_type parameterized \
--alpha-min -2.0 \
--alpha-max 2.0 \
--alpha-step 0.25 \
--save_name "$PARAM_TRAIN_NAME"

echo "=== Stage 3: fit intervals ==="
python -u ../experiments/compute_intervals.py \
--training_experiments "alpha-iterative:${ALPHA_TRAIN_NAME}" "parameterized:${PARAM_TRAIN_NAME}" \
--behaviors "${BEHAVIORS[@]}" \
--layer "$LAYER" \
--step 0.25 \
--save_name "$INTERVAL_MAP_NAME"


echo "=== Stage 4a: multi-attribute experiment behavior 1==="
python -u ../experiments/run_multi_attribute_experiment.py \
--model_path "$MODEL_PATH" \
--activations_name "$ACTIVATIONS_NAME" \
--dataset_subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--target_classes "${TARGET_CLASSES[@]}" \
--test_behaviors "$TEST_BEHAVIOR1" \
--interval_map_name "$INTERVAL_MAP_NAME" \
--alpha --parameterized \
--save_name "$EXPERIMENT_NAME1"

echo "=== Stage 4b: multi-attribute experiment behavior 2==="
python -u ../experiments/run_multi_attribute_experiment.py \
--model_path "$MODEL_PATH" \
--activations_name "$ACTIVATIONS_NAME" \
--dataset_subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--target_classes "${TARGET_CLASSES[@]}" \
--test_behaviors "$TEST_BEHAVIOR2" \
--interval_map_name "$INTERVAL_MAP_NAME" \
--alpha --parameterized \
--save_name "$EXPERIMENT_NAME2"

echo "=== Done. Experiment 1: results/logit_results/${EXPERIMENT_NAME1}.pkl ==="
echo "=== Done. Experiment 2: results/logit_results/${EXPERIMENT_NAME2}.pkl ==="

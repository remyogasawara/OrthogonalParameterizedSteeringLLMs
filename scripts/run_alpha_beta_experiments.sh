#!/bin/bash
#SBATCH --job-name=alpha-beta_experiments
#SBATCH --partition=utahdb-gpu-np
#SBATCH --account=utahdb-gpu-np

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --time=72:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# -------------------------
# Environment setup
# -------------------------
source /uufs/chpc.utah.edu/sys/installdir/r8/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate steering

set -euo pipefail  # stop the whole chain if any stage fails — later stages depend on earlier outputs

source ~/.hf_token
# -------------------------
# Run config
# -------------------------
DATE="$(date +%-m-%-d-%Y)"
# MODEL_PATH="Qwen/Qwen3-8B"
# MODEL_NAME="Qwen3-8B"
# LAYER=22
# MODEL_PATH="meta-llama/Llama-3.1-8B" 
# MODEL_NAME="Llama-3.1-8B"
# LAYER=14

MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="Llama-2-7b-chat-hf" 
LAYER=13

DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"

BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH myopic-reward survival-instinct power-seeking-inclination wealth-seeking-inclination)
TEST_BEHAVIOR1=wealth-seeking-inclination
TEST_BEHAVIOR2=corrigible-neutral-HHH

TARGET_CLASSES=(${TEST_BEHAVIOR1} ${TEST_BEHAVIOR2})


ACTIVATIONS_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}")"
ALPHA_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_alpha-iterative_training_experiment")"
PARAM_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_parameterized_training_experiment")"
INTERVAL_MAP_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_interval_map")"

EXPERIMENT_NAME1="${DATE}_$(IFS=_; echo "${TEST_BEHAVIOR1}_${TEST_BEHAVIOR2}")_multi_attribute_experiment"
EXPERIMENT_NAME2="${DATE}_$(IFS=_; echo "${TEST_BEHAVIOR2}_${TEST_BEHAVIOR1}")_multi_attribute_experiment"

# SKIP STEPS 1-3 (activations, single-behavior alpha sweep, and parameterized training) if the corresponding files already exist (run in run_multi_steering_pipeline.sh)
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

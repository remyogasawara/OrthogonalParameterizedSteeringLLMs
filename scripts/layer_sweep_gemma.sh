#!/bin/bash
#SBATCH --job-name=gemma_layer_sweep
#SBATCH --partition=utahdb-gpu-np
#SBATCH --account=utahdb-gpu-np

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=60G
#SBATCH --gres=gpu:1

#SBATCH --time=72:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# -------------------------
# Environment setup
# -------------------------
set -eo pipefail   # -u added back in below, after conda's activation hooks run

source /uufs/chpc.utah.edu/sys/installdir/miniconda3/23.11.0/etc/profile.d/conda.sh

set +u
conda activate steering
set -u

which python || { echo "ERROR: python not found after activation" >&2; exit 1; }

source ~/.hf_token

# -------------------------
# Run config
# -------------------------
DATE="$(date +%-m-%-d-%Y)"
MODEL_PATH="google/gemma-3-27b-it"
MODEL_NAME="Gemma-3-27b-it"


DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"
BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH myopic-reward survival-instinct power-seeking-inclination wealth-seeking-inclination)


# Layers to sweep. Qwen3-8B has 36 layers; literature on this model tends to
# land steering effects deeper than proportional-to-Llama estimates would
# suggest (roughly layers 14-25), so we sweep a wider band around that.
LAYERS=(16 18 20 22 24 26 28 30 32 34)

BEHAVIORS_TAG="$(IFS=_; echo "${BEHAVIORS[*]}")"

# Where per-layer summary rows get appended so you can compare afterwards
# without having to reload every pickle by hand.
SWEEP_DIR="../results/layer_sweep"
mkdir -p "$SWEEP_DIR"
SUMMARY_CSV="${SWEEP_DIR}/${DATE}_${BEHAVIORS_TAG}_layer_sweep_summary.csv"
echo "layer,activations_name,alpha_train_name,param_train_name" > "$SUMMARY_CSV"

for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "##########################################################"
    echo "### LAYER ${LAYER}"
    echo "##########################################################"

    ACTIVATIONS_NAME="${DATE}_${MODEL_NAME}_${BEHAVIORS_TAG}_layer${LAYER}"
    ALPHA_TRAIN_NAME="${DATE}_${MODEL_NAME}_alpha-iterative_training_experiment_layer${LAYER}"
    PARAM_TRAIN_NAME="${DATE}_${MODEL_NAME}_parameterized_training_experiment_layer${LAYER}"

    echo "=== [layer ${LAYER}] Stage 1: compute activations ==="
    python -u ../experiments/new_get_activations.py "$MODEL_PATH" \
        --behaviors "${BEHAVIORS[@]}" \
        --dataset-subfolder "$DATASET_SUBFOLDER" \
        --layer "$LAYER" \
        --save-name "$ACTIVATIONS_NAME"

    echo "=== [layer ${LAYER}] Stage 2: single-behavior alpha sweep (alpha-iterative) ==="
    python -u ../experiments/test_single_behavior.py \
        --model_path "$MODEL_PATH" \
        --activations_name "$ACTIVATIONS_NAME" \
        --dataset_subfolder "$DATASET_SUBFOLDER" \
        --layer "$LAYER" \
        --steering_type alpha-iterative \
        --save_name "$ALPHA_TRAIN_NAME"

    echo "=== [layer ${LAYER}] Stage 2: single-behavior alpha sweep (parameterized) ==="
    python -u ../experiments/test_single_behavior.py \
        --model_path "$MODEL_PATH" \
        --activations_name "$ACTIVATIONS_NAME" \
        --dataset_subfolder "$DATASET_SUBFOLDER" \
        --layer "$LAYER" \
        --steering_type parameterized \
        --save_name "$PARAM_TRAIN_NAME"

    echo "${LAYER},${ACTIVATIONS_NAME},${ALPHA_TRAIN_NAME},${PARAM_TRAIN_NAME}" >> "$SUMMARY_CSV"

    echo "=== [layer ${LAYER}] Done. Stage 2 results: ${ALPHA_TRAIN_NAME}.pkl, ${PARAM_TRAIN_NAME}.pkl ==="
done

echo ""
echo "=== Layer sweep complete. Summary index: ${SUMMARY_CSV} ==="
echo "=== Load each results/logit_results/<alpha_train_name|param_train_name>.pkl to compare layers. ==="
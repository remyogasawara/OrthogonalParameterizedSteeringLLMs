#!/bin/bash
#SBATCH --job-name=multi-steering-clipping
#SBATCH --partition=granite-gpu
#SBATCH --account=phillipsj
#SBATCH --qos=granite-gpu-freecycle

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s"

#SBATCH --time=24:00:00

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# -------------------------
# Environment setup
# -------------------------

source /uufs/chpc.utah.edu/sys/installdir/r8/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
# conda activate /uufs/chpc.utah.edu/common/home/u1591141/.conda/envs/steering
conda activate /uufs/chpc.utah.edu/common/home/u1591141/.conda/envs/steering
which python
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

set -euo pipefail  # stop the whole chain if any stage fails — later stages depend on earlier outputs

source ~/.hf_token
# -------------------------
# Run config
# -------------------------
DATE="$(date +%-m-%-d-%Y)"
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="Llama-2-7b-chat-hf" 

DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"
LAYER=13
BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH myopic-reward survival-instinct power-seeking-inclination wealth-seeking-inclination)

ALPHA_TRAIN_NAME="${DATE}_alpha-iterative_training_experiment_caa"
PARAM_TRAIN_NAME="${DATE}_parameterized_training_experiment_caa"
INTERVAL_MAP_NAME="${DATE}_interval_map"
EXPERIMENT_NAME="${DATE}_$(IFS=_; echo "${BEHAVIORS[*]}")_multi_attribute_experiment"

BEHAVIOR_TAG="$(IFS=_; printf '%s' "${BEHAVIORS[*]}")"
ACTIVATIONS_NAME="${MODEL_NAME}_${BEHAVIOR_TAG}"

ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-../activations}"
ACTIVATIONS_PATH="${ACTIVATIONS_DIR}/${ACTIVATIONS_NAME}"

if [[ -e "${ACTIVATIONS_PATH}" ]]; then
    echo "=== Stage 1: skipped; existing activations found ==="
    echo "${ACTIVATIONS_PATH}"
else
    echo "=== Stage 1: compute activations ==="
    python -u ../experiments/new_get_activations.py "${MODEL_PATH}" \
        --behaviors "${BEHAVIORS[@]}" \
        --dataset-subfolder "${DATASET_SUBFOLDER}" \
        --layer "${LAYER}" \
        --save-name "${ACTIVATIONS_NAME}"
fi

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

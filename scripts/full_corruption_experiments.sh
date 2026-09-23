#!/bin/bash
#SBATCH --job-name=full_corruption_experiments
#SBATCH --partition=granite-gpu
#SBATCH --account=phillipsj
#SBATCH --qos=granite-gpu-freecycle

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s"

#SBATCH --time=8:00:00
#SBATCH --array=0-13%4

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# -------------------------
# Environment setup
# -------------------------
source /uufs/chpc.utah.edu/sys/installdir/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate myenv

python -u experiments/mislabel.py --model-path allenai/OLMo-2-1124-7B-Instruct --activations_name OLMo-2-1124-7B-Instruct_layer13 --alpha 1 --layer 12 --grab-steering-vecs-only

# -------------------------
# Model / activation / alpha arrays
# -------------------------
MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "allenai/OLMo-2-1124-7B-Instruct"
)

ACTIVATION_NAMES=(
    "Llama-3.2-3B-Instruct_layer13"
    "Mistral-7B-Instruct-v0.3_layer13"
    "OLMo-2-1124-7B-Instruct_layer13"
)

ALPHAS=(1 1 1) # NOT USED ANYMORE

LAYER=12

# -------------------------
# Define behavior partitions
# -------------------------
BEHAVIORS=(
  "uncorrigible-neutral-HHH myopic-reward"
  "power-seeking-inclination wealth-seeking-inclination"
  "survival-instinct coordinate-other-ais"
)

# -------------------------
# Build list of experiments
# Each experiment is: script_name model_path activations_name alpha [behaviors]
# -------------------------
EXPERIMENTS=()

# mislabel.py and random_injection.py (2 runs per model)
ONLY_RANDOM_MODEL="meta-llama/Llama-3.2-3B-Instruct"

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    act_name="${ACTIVATION_NAMES[$i]}"
    alpha="${ALPHAS[$i]}"

    if [[ "$model" != "$ONLY_RANDOM_MODEL" ]]; then
        EXPERIMENTS+=(
          "experiments/mislabel.py --model_path $model --activations_name $act_name --alpha $alpha"
        )
    else
        echo "Skipping mislabel for $model"
    fi

    # Always run random injection
    EXPERIMENTS+=(
      "experiments/random_injection.py --model_path $model --activations_name $act_name --alpha $alpha"
    )
done
# behavior_injection.py (3 runs per model, behavior partitions)
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    act_name="${ACTIVATION_NAMES[$i]}"
    alpha="${ALPHAS[$i]}"

    for beh in "${BEHAVIORS[@]}"; do
        EXPERIMENTS+=("experiments/behavior_injection.py \
                      --model_path $model \
                      --activations_name $act_name \
                      --alpha $alpha \
                      --save-name-postfix "${beh// /_}" \
                      --behaviors $beh")
    done
done

# -------------------------
# Pick experiment for this array task
# -------------------------
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "Running: python -u $EXPERIMENT --layer $LAYER"
python -u $EXPERIMENT --layer $LAYER



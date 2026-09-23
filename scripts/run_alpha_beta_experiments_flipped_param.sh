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

# Stop if a command fails, an unset variable is used, or a pipeline fails.
set -euo pipefail

source ~/.hf_token

# -------------------------
# Run config
# -------------------------
DATE="$(date +%-m-%-d-%Y)"

MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="Llama-2-7b-chat-hf"
LAYER=13

DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"

BEHAVIORS=(
    coordinate-other-ais
    corrigible-neutral-HHH
    myopic-reward
    survival-instinct
    power-seeking-inclination
    wealth-seeking-inclination
)

ACTIVATIONS_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}")"
INTERVAL_MAP_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_interval_map")"

# Each entry contains:
# behavior 1|behavior 2
PAIRS=(
    "coordinate-other-ais|power-seeking-inclination"
    "coordinate-other-ais|corrigible-neutral-HHH"
    "corrigible-neutral-HHH|power-seeking-inclination"
    "power-seeking-inclination|wealth-seeking-inclination"
    "coordinate-other-ais|wealth-seeking-inclination"
    "corrigible-neutral-HHH|wealth-seeking-inclination"
    "myopic-reward|wealth-seeking-inclination"
)

# -------------------------
# Run all pairs
# -------------------------
for PAIR in "${PAIRS[@]}"; do
    IFS="|" read -r BEHAVIOR1 BEHAVIOR2 <<< "$PAIR"

    echo
    echo "============================================================"
    echo "Pair: ${BEHAVIOR1} and ${BEHAVIOR2}"
    echo "============================================================"

    # Run both target-class orders:
    #   1. behavior 1 first
    #   2. behavior 2 first
    for FIRST_BEHAVIOR in "$BEHAVIOR1" "$BEHAVIOR2"; do
        if [[ "$FIRST_BEHAVIOR" == "$BEHAVIOR1" ]]; then
            SECOND_BEHAVIOR="$BEHAVIOR2"
        else
            SECOND_BEHAVIOR="$BEHAVIOR1"
        fi

        TARGET_CLASSES=("$FIRST_BEHAVIOR" "$SECOND_BEHAVIOR")

        echo
        echo "Target-class order: ${TARGET_CLASSES[*]}"

        # Evaluate both behaviors under this target-class order.
        for TEST_BEHAVIOR in "$BEHAVIOR1" "$BEHAVIOR2"; do
            EXPERIMENT_NAME="flipped_param/${DATE}_${BEHAVIOR1}_${BEHAVIOR2}_${FIRST_BEHAVIOR}_first_test_${TEST_BEHAVIOR}_multi_attribute_experiment"

            echo
            echo "Running:"
            echo "  First target class: $FIRST_BEHAVIOR"
            echo "  Second target class: $SECOND_BEHAVIOR"
            echo "  Test behavior: $TEST_BEHAVIOR"
            echo "  Save name: $EXPERIMENT_NAME"

            python -u ../experiments/run_multi_attribute_experiment.py \
                --model_path "$MODEL_PATH" \
                --activations_name "$ACTIVATIONS_NAME" \
                --dataset_subfolder "$DATASET_SUBFOLDER" \
                --layer "$LAYER" \
                --target_classes "${TARGET_CLASSES[@]}" \
                --test_behaviors "$TEST_BEHAVIOR" \
                --interval_map_name "$INTERVAL_MAP_NAME" \
                --parameterized \
                --save_name "$EXPERIMENT_NAME"

            echo "Finished: results/logit_results/${EXPERIMENT_NAME}.pkl"
        done
    done
done

echo
echo "=== All parameterized multi-attribute experiments completed ==="
#!/bin/bash
#SBATCH --job-name=run_alpha_beta_pairs_flipped_param
#SBATCH --partition=<SLURM_PARTITION>
#SBATCH --account=<SLURM_ACCOUNT>

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --time=72:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Parameterized (P) and orthogonal parameterized (OP) multi-attribute sweeps
# for the 7 trait pairs of Tables 2 and 3, in BOTH steering orders.
# Parameterized steering depends on which trait is steered first, so every pair
# is run with each trait first and evaluated on each trait:
# 7 pairs x 2 orders x 2 test behaviours = 28 runs.
#
# Paper results this feeds:
#   - Table 2, tab:avg-score-mean-sd-beta-alpha-fixed-full: the OP_alpha,
#     P_alpha, OP_beta and P_beta columns (subscript = trait steered first).
#   - Appendix "Additional Tables with Percent Steered",
#     tab:percent-steered-mean-sd-beta-alpha-fixed-full: the same four columns.
#   The OAI and AI columns come from run_alpha_beta_pairs.sh.
#   These pickles also hold OP and P for both orders, from which Table 3's OP /
#   P / Delta_P columns can be computed (range over alpha in [-1, 1] at
#   beta = -1 and +1). No public script prints those columns.
#
# Prerequisite: Stages 1-3 of run_multi_steering_pipeline.sh, which write the
# activations and interval map for the six xrisk behaviours
# (Llama-2-7b-chat-hf, layer 13).
# Interval map used by the paper: <MODEL>_<BEHAVIORS>_interval_map, fitted on
# the single-behaviour sweep at the Python-default grid
# (train_single_behavior.py --alpha-step 0.1, i.e. 41 alphas in [-2, 2];
# compute_intervals.py --step 0.1 --gamma 0.1). This script defaults to the
# name the public pipeline's Stage 3 writes (..._large_interval_map). If that
# map was fitted at step 0.25, the script runs, but the intervals and hence the
# table numbers differ from the paper's. To match the paper, run Stages 2-3
# with the step-0.1 values and pass the map's name:
#   INTERVAL_MAP_NAME=<name> sbatch run_alpha_beta_pairs_flipped_param.sh
#
# Run from the scripts/ directory:
#   sbatch run_alpha_beta_pairs_flipped_param.sh                  # all 7 pairs (28 runs)
#   bash   run_alpha_beta_pairs_flipped_param.sh                  # same, without Slurm
#   bash   run_alpha_beta_pairs_flipped_param.sh <b1> <b2>        # one pair, both orders
#
# Writes 28 pickles:
#   results/logit_results/alpha_beta/flipped_param/<DATE>_<b1>_<b2>_<first>_first_test_<test>_multi_attribute_experiment.pkl
# Keep this file name: src/parameterized_mean_sd.py reads the "<first>_first"
# part to tell the two steering orders apart.
#
# Sweep settings are the defaults of run_multi_attribute_experiment.py: alpha
# in [-2, 2] at step 0.25, beta in [-2, 2] at step 0.5, 200 held-out test
# questions per behaviour, batch size 4, seed 42.
#
# Turn the pickles into table numbers (command run from src/):
#   python parameterized_mean_sd.py ../results/logit_results/alpha_beta/flipped_param \
#       --pattern "*_first_test_*_multi_attribute_experiment.pkl" --metric avg_score \
#       --output-dir ../results/tables/avg_score_param_mean_sd
# In its output, "alpha first" rows are OP_alpha / P_alpha and "beta first"
# rows are OP_beta / P_beta. Repeat with --metric percent_steered for the
# appendix table.
# =============================================================================

# -------------------------
# Environment setup
# Run from the scripts/ directory (paths below are relative to it).
# -------------------------
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate steering
fi

# Stop if a command fails, an unset variable is used, or a pipeline fails.
set -euo pipefail

# Slurm opens logs/ before this line runs, so create scripts/logs once before
# the first sbatch; this keeps plain `bash` runs working too.
mkdir -p logs
# Optional: a file that exports HF_TOKEN; alternatively run `hf auth login` or export HF_TOKEN yourself
if [ -f "$HOME/.hf_token" ]; then source "$HOME/.hf_token"; fi

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
# Must equal the Stage 3 --save_name in run_multi_steering_pipeline.sh.
# Override with: INTERVAL_MAP_NAME=<name> bash run_alpha_beta_pairs_flipped_param.sh
INTERVAL_MAP_NAME="${INTERVAL_MAP_NAME:-${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}")_large_interval_map}"

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

if [ "$#" -eq 2 ]; then
    PAIRS=("$1|$2")
elif [ "$#" -ne 0 ]; then
    echo "Usage: bash run_alpha_beta_pairs_flipped_param.sh [<behavior1> <behavior2>]" >&2
    exit 2
fi

# -------------------------
# Pre-flight checks
# -------------------------
if [ ! -f ../experiments/run_multi_attribute_experiment.py ]; then
    echo "Run this script from the scripts/ directory of the repository." >&2
    exit 1
fi
for REQUIRED in "../activations/${ACTIVATIONS_NAME}.pkl" "../results/intervals/${INTERVAL_MAP_NAME}.pkl"; do
    if [ ! -f "$REQUIRED" ]; then
        echo "Missing ${REQUIRED}: run Stages 1-3 of run_multi_steering_pipeline.sh first," >&2
        echo "or set INTERVAL_MAP_NAME to the --save_name your Stage 3 used." >&2
        exit 1
    fi
done

# run_multi_attribute_experiment.py creates results/logit_results/alpha_beta/
# but not the flipped_param/ subfolder used in the save names below.
mkdir -p ../results/logit_results/alpha_beta/flipped_param

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

            echo "Finished: ../results/logit_results/alpha_beta/${EXPERIMENT_NAME}.pkl"
        done
    done
done

echo
echo "=== All parameterized multi-attribute experiments completed ==="

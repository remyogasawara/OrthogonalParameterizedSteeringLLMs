#!/bin/bash
#SBATCH --job-name=run_alpha_beta_pairs
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
# Multi-attribute alpha/beta sweeps for the 7 trait pairs of Tables 2 and 3,
# with all four steering methods: alpha-iterative (AI), orthogonal
# alpha-iterative (OAI), parameterized (P) and orthogonal parameterized (OP).
#
# Paper results this feeds:
#   - Table 2, tab:avg-score-mean-sd-beta-alpha-fixed-full: the OAI, AI and
#     Delta_AI columns.
#   - Table 3, tab:alpha-range: the OAI, AI and Delta_AI columns. These
#     pickles also hold OP and P (P in the PAIRS order only), but
#     src/fixed_endpoint_range.py reports only OAI and AI, and no public script
#     prints Table 3's OP / P / Delta_P columns.
#   - Appendix "Additional Tables with Percent Steered": the OA-I/A-I columns of
#     tab:percent-steered-mean-sd-beta-alpha-fixed-full and of
#     tab:alpha-percent-scored-range.
#   The order-specific OP_alpha / P_alpha / OP_beta / P_beta columns of Table 2
#   and of the appendix sensitivity table come from
#   run_alpha_beta_pairs_flipped_param.sh.
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
#   INTERVAL_MAP_NAME=<name> sbatch run_alpha_beta_pairs.sh
#
# Run from the scripts/ directory:
#   sbatch run_alpha_beta_pairs.sh                  # all 7 pairs (14 runs)
#   bash   run_alpha_beta_pairs.sh                  # same, without Slurm
#   bash   run_alpha_beta_pairs.sh <first> <second> # one pair, <first> steered first
#
# Writes 2 pickles per pair (one per test behaviour), 14 in total:
#   results/logit_results/alpha_beta/<DATE>_<test>_<other>_multi_attribute_experiment.pkl
# The row "<test> / <other>" is the (alpha trait / beta trait) row of the tables.
#
# Sweep settings are the defaults of run_multi_attribute_experiment.py: alpha
# in [-2, 2] at step 0.25, beta in [-2, 2] at step 0.5, 200 held-out test
# questions per behaviour, batch size 4, seed 42.
#
# Turn the pickles into table numbers (commands run from src/):
#   python mean_sd.py ../results/logit_results/alpha_beta \
#       --pattern "*_multi_attribute_experiment.pkl" --metric avg_score \
#       --output-dir ../results/tables/avg_score_mean_sd
#   python fixed_endpoint_range.py \
#       ../results/logit_results/alpha_beta/*_multi_attribute_experiment.pkl \
#       --metric avg_score --correlations correlations.json \
#       --output-dir ../results/tables/avg_score_fixed_endpoint_range
# Repeat both with --metric percent_steered for the appendix tables.
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
# Override with: INTERVAL_MAP_NAME=<name> bash run_alpha_beta_pairs.sh
INTERVAL_MAP_NAME="${INTERVAL_MAP_NAME:-${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}")_large_interval_map}"

# Each entry is "first|second": TARGET_CLASSES=(first second), so `first` is
# steered first. Each pair is run twice, once per test behaviour, which gives
# both ordered rows (alpha trait / beta trait) of Tables 2 and 3.
# AI, OAI and OP do not depend on this order (up to floating-point noise);
# only P does. Both P orders are run by run_alpha_beta_pairs_flipped_param.sh.
# The order used for Table 3's single P column is known for
# coordinate-other-ais / power-seeking-inclination (coordinate-other-ais first,
# as listed). For the other pairs it is not recorded, so the order listed here
# is not guaranteed to match it.
PAIRS=(
    "coordinate-other-ais|power-seeking-inclination"
    "coordinate-other-ais|corrigible-neutral-HHH"
    "corrigible-neutral-HHH|power-seeking-inclination"
    "power-seeking-inclination|wealth-seeking-inclination"
    "coordinate-other-ais|wealth-seeking-inclination"
    "wealth-seeking-inclination|corrigible-neutral-HHH"
    "myopic-reward|wealth-seeking-inclination"
)

if [ "$#" -eq 2 ]; then
    PAIRS=("$1|$2")
elif [ "$#" -ne 0 ]; then
    echo "Usage: bash run_alpha_beta_pairs.sh [<first_behavior> <second_behavior>]" >&2
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

# -------------------------
# Run all pairs
# -------------------------
for PAIR in "${PAIRS[@]}"; do
    IFS="|" read -r TEST_BEHAVIOR1 TEST_BEHAVIOR2 <<< "$PAIR"
    TARGET_CLASSES=("$TEST_BEHAVIOR1" "$TEST_BEHAVIOR2")

    EXPERIMENT_NAME1="${DATE}_${TEST_BEHAVIOR1}_${TEST_BEHAVIOR2}_multi_attribute_experiment"
    EXPERIMENT_NAME2="${DATE}_${TEST_BEHAVIOR2}_${TEST_BEHAVIOR1}_multi_attribute_experiment"

    echo
    echo "============================================================"
    echo "Pair: ${TARGET_CLASSES[*]} (steered in this order)"
    echo "============================================================"

    echo "=== Multi-attribute experiment, test behavior 1: ${TEST_BEHAVIOR1} ==="
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

    echo "=== Multi-attribute experiment, test behavior 2: ${TEST_BEHAVIOR2} ==="
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

    echo "=== Done. Experiment 1: ../results/logit_results/alpha_beta/${EXPERIMENT_NAME1}.pkl ==="
    echo "=== Done. Experiment 2: ../results/logit_results/alpha_beta/${EXPERIMENT_NAME2}.pkl ==="
done

echo
echo "=== All multi-attribute pair experiments completed ==="

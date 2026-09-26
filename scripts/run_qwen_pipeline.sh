#!/bin/bash
#SBATCH --job-name=qwen_multi_steering_pipeline
#SBATCH --partition=<SLURM_PARTITION>
#SBATCH --account=<SLURM_ACCOUNT>

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --time=96:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Qwen3-8B multi-attribute steering pipeline (Qwen appendix)
# -----------------------------------------------------------------------------
# Paper result: appendix "Replicating Results on Qwen 3.8"
#   (\label{app:qwen_results}): Table tab:qwen-alpha-range (range of avg_score)
#   and Table tab:qwen-alpha-percent-range (range of percent_steered), both
#   "across alpha in [-1,1] with beta fixed at -1 and +1", for the six ordered
#   trait pairs coordination/power, power/coordination,
#   coordination/corrigibility, corrigibility/coordination, power/wealth and
#   wealth/power, with the OAI, AI, OP and P estimators.
#
# This is scripts/run_multi_steering_pipeline.sh with only these changes:
#   MODEL_PATH=Qwen/Qwen3-8B, MODEL_NAME=Qwen3-8B, LAYER=22 (the layer chosen
#   by scripts/layer_sweep_qwen.sh); Stage 4 loops over the three trait pairs
#   in PAIRS (4a and 4b per pair give the pair's two table rows); Stage 4 file names include
#   MODEL_NAME so they do not overwrite a Llama run from the same day; Stage 1
#   passes --skip-auth-check because Qwen3-8B is not gated. The alpha grid
#   (-2.0 to 2.0, step 0.25), the interval-fitting step (0.25), the default
#   gamma (0.1), the dataset, the six behaviors, the Stage 4 alpha/beta grids
#   and all seeds are identical to the Llama pipeline.
#
# How to run (from the scripts/ directory; one GPU):
#   Fill in <SLURM_PARTITION> and <SLURM_ACCOUNT> above, or pass
#   --partition=... --account=... to sbatch (command-line options override them).
#   cd scripts
#   mkdir -p logs          # Slurm opens logs/ before this script starts
#   sbatch run_qwen_pipeline.sh      # or: bash run_qwen_pipeline.sh
#   To run a subset of pairs, edit PAIRS below. Stages 1-3 are shared by all
#   pairs; to skip them on a rerun, comment them out.
#
# Writes (paths relative to the repository root; <DATE> is the start date):
#   activations/Qwen3-8B_<behaviors>.pkl                                   (Stage 1)
#   results/logit_results/Qwen3-8B_<behaviors>_{alpha-iterative,parameterized}_training_experiment_large_intervals.pkl  (Stage 2)
#   results/intervals/Qwen3-8B_<behaviors>_large_interval_map.pkl          (Stage 3)
#   results/logit_results/alpha_beta/<DATE>_Qwen3-8B_<alpha trait>_<beta trait>_multi_attribute_experiment.pkl
#       six files, one per table row                                       (Stage 4)
# The OAI and AI range columns (and Delta_AI) of both tables can be computed
# from the Stage 4 files with src/fixed_endpoint_range.py, for example:
#   python src/fixed_endpoint_range.py \
#     results/logit_results/alpha_beta/<DATE>_Qwen3-8B_*_multi_attribute_experiment.pkl \
#     --metric avg_score --output-dir results/qwen_fixed_endpoint_range_tables/avg_score
#   (repeat with --metric percent_steered). That script reports only the OAI and
#   AI estimators; the OP and P columns come from the same Stage 4 files.
# =============================================================================

# -------------------------
# Environment setup
# Run from the scripts/ directory: sbatch run_qwen_pipeline.sh  (or: bash run_qwen_pipeline.sh)
# -------------------------
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate steering
fi

set -euo pipefail  # stop the whole chain if any stage fails; later stages depend on earlier outputs

mkdir -p logs
# Optional: a file that exports HF_TOKEN; alternatively run `hf auth login` or export HF_TOKEN yourself.
# Qwen/Qwen3-8B is not gated, so no token is required.
if [ -f "$HOME/.hf_token" ]; then source "$HOME/.hf_token"; fi
# -------------------------
# Run config
# -------------------------
DATE="$(date +%-m-%-d-%Y)"

MODEL_PATH="Qwen/Qwen3-8B"
MODEL_NAME="Qwen3-8B"
LAYER=22   # zero-indexed; chosen by scripts/layer_sweep_qwen.sh + src/layer_evaluation.py

DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"

BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH myopic-reward survival-instinct power-seeking-inclination wealth-seeking-inclination)

# Trait pairs of the Qwen appendix tables, as "behavior 1|behavior 2".
# Both Stage 4 runs of a pair use TARGET_CLASSES=(behavior 1, behavior 2),
# exactly as in run_multi_steering_pipeline.sh. Stage 4a evaluates on
# behavior 1's test items and Stage 4b on behavior 2's. In each run,
# src/multi_attribute_steering.py (evaluate_multi_attribute) applies alpha to the
# evaluated behavior and beta to the other one, and saves the run with
# behavior=<evaluated behavior> and beta_behavior=<the other behavior>.
PAIRS=(
    "coordinate-other-ais|power-seeking-inclination"
    "coordinate-other-ais|corrigible-neutral-HHH"
    "power-seeking-inclination|wealth-seeking-inclination"
)

ACTIVATIONS_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}")"
ALPHA_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_alpha-iterative_training_experiment_large_intervals")"
PARAM_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_parameterized_training_experiment_large_intervals")"
INTERVAL_MAP_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_large_interval_map")"

# --skip-auth-check only skips the Hugging Face login check, which would fail
# without a login even though Qwen3-8B is public. It does not change any result.
echo "=== Stage 1: compute activations ==="
python -u ../experiments/new_get_activations.py "$MODEL_PATH" \
--behaviors "${BEHAVIORS[@]}" \
--dataset-subfolder "$DATASET_SUBFOLDER" \
--layer "$LAYER" \
--save-name "$ACTIVATIONS_NAME" \
--skip-auth-check

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

EXPERIMENT_NAMES=()
for PAIR in "${PAIRS[@]}"; do
IFS="|" read -r TEST_BEHAVIOR1 TEST_BEHAVIOR2 <<< "$PAIR"
TARGET_CLASSES=("$TEST_BEHAVIOR1" "$TEST_BEHAVIOR2")

EXPERIMENT_NAME1="${DATE}_${MODEL_NAME}_$(IFS=_; echo "${TEST_BEHAVIOR1}_${TEST_BEHAVIOR2}")_multi_attribute_experiment"
EXPERIMENT_NAME2="${DATE}_${MODEL_NAME}_$(IFS=_; echo "${TEST_BEHAVIOR2}_${TEST_BEHAVIOR1}")_multi_attribute_experiment"

echo "=== Stage 4a: multi-attribute experiment, alpha on ${TEST_BEHAVIOR1} (evaluated), beta on ${TEST_BEHAVIOR2} ==="
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

echo "=== Stage 4b: multi-attribute experiment, alpha on ${TEST_BEHAVIOR2} (evaluated), beta on ${TEST_BEHAVIOR1} ==="
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

EXPERIMENT_NAMES+=("$EXPERIMENT_NAME1" "$EXPERIMENT_NAME2")
done

for EXPERIMENT_NAME in "${EXPERIMENT_NAMES[@]}"; do
echo "=== Done. Experiment: results/logit_results/alpha_beta/${EXPERIMENT_NAME}.pkl ==="
done

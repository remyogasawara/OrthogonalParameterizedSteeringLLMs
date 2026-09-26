#!/bin/bash
#SBATCH --job-name=mat_steer_pipeline
#SBATCH --partition=<SLURM_PARTITION>
#SBATCH --account=<SLURM_ACCOUNT>

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --time=96:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# MAT-Steer comparison: Table tab:matsteer_compare ("Percent_steered across bias
# datasets and steering methods compared with MAT-Steer", subsection "Comparison
# to MAT-Steer"). Produces our rows of that table on BBQ, Toxigen and TruthfulQA;
# the MAT-Steer rows are quoted from the MAT-Steer paper.
#
# Model and layer used for the table (NOT the Llama 2 model used elsewhere):
#   MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct   LAYER=14 (zero-indexed)
# Datasets: datasets/mat_steer/{bbq,truthfulqa,toxigen}.jsonl, built by
#   experiments/prepare_mat_steer_datasets.py (Stage 0 below runs it if missing).
#   Each file is shuffled with seed 42; the last 200 items are the test set and
#   the rest are used to fit the vectors and intervals (src/dataset.py).
#
# How to run: from the scripts/ directory (all paths below are relative to it):
#   cd scripts && mkdir -p logs && sbatch run_mat_steer_pipeline.sh
#   (or without Slurm: cd scripts && bash run_mat_steer_pipeline.sh)
# Needs one GPU, a "steering" conda env (environment.yaml), and a Hugging Face
# token with access to the gated Llama 3.1 and ToxiGen repos (export HF_TOKEN,
# run `hf auth login`, or put `export HF_TOKEN=...` in $HOME/.hf_token).
#
# Writes (paths relative to the repo root):
#   datasets/mat_steer/*.jsonl                         Stage 0 (only if missing)
#   activations/${ACTIVATIONS_NAME}.pkl                Stage 1
#   results/logit_results/${ALPHA_TRAIN_NAME}.pkl      Stage 2a
#   results/logit_results/${PARAM_TRAIN_NAME}.pkl      Stage 2b
#   results/intervals/${INTERVAL_MAP_NAME}.pkl         Stage 3
#   results/logit_results/<DATE>_mat_multi_attribute_alpha_one.pkl
#       Stage 4: rows alpha-iterative, Parameterized, Orthogonal alpha-iterative,
#       Orthogonal Parameterized
#   results/logit_results/<DATE>_mat_no_steer_baseline.pkl
#       Stage 5a: row No-steering baseline
#   results/logit_results/<DATE>_mat_unclipped_multi_alpha_one.pkl
#       Stage 5b: row Unclipped alpha-iterative multi-steering ("alpha-iterative" rows)
#   results/logit_results/<DATE>_mat_unclipped_single_<behavior>_alpha_one.pkl
#       Stage 5c: row Unclipped alpha-iterative single-steering ("alpha-iterative" rows)
# Each driver run also prints a percent_steered (%) summary to the log.
# =============================================================================

# -------------------------
# Environment setup
# Run from the scripts/ directory (see above).
# -------------------------
if command -v conda >/dev/null 2>&1; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate steering; fi

set -euo pipefail

mkdir -p logs

python --version
which python

# Optional: a file that exports HF_TOKEN
if [ -f "$HOME/.hf_token" ]; then source "$HOME/.hf_token"; fi

# -------------------------
DATE="$(date +%-m-%-d-%Y)"
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3.1-8B-Instruct"
LAYER=14

DATASET_SUBFOLDER="mat_steer"
BEHAVIORS=(bbq truthfulqa toxigen)
TARGET_CLASSES=(bbq truthfulqa toxigen)

ACTIVATIONS_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}")"
ALPHA_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_alpha-iterative_training_experiment")"
PARAM_TRAIN_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_parameterized_training_experiment")"
INTERVAL_MAP_NAME="${MODEL_NAME}_$(IFS=_; echo "${BEHAVIORS[*]}_interval_map")"

echo "=== Stage 0: build datasets/${DATASET_SUBFOLDER} (skipped if already present) ==="
NEED_DATA=0
for B in "${BEHAVIORS[@]}"; do
  if [ ! -f "../datasets/${DATASET_SUBFOLDER}/${B}.jsonl" ]; then NEED_DATA=1; fi
done
if [ "$NEED_DATA" -eq 1 ]; then
  # Downloads TruthfulQA, ToxiGen (gated) and BBQ from the Hugging Face Hub.
  # If compute nodes have no internet, run this line once on a login node first.
  python -u ../experiments/prepare_mat_steer_datasets.py \
    --output-dir "../datasets/${DATASET_SUBFOLDER}"
fi

echo "=== Stage 1: compute activations ==="
python -u ../experiments/new_get_activations.py "$MODEL_PATH" \
  --behaviors "${BEHAVIORS[@]}" \
  --dataset-subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --save-name "$ACTIVATIONS_NAME"

echo "=== Stage 2a: alpha-iterative single-behavior sweep ==="
python -u ../experiments/train_single_behavior.py \
  --model_path "$MODEL_PATH" \
  --activations_name "$ACTIVATIONS_NAME" \
  --dataset_subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --steering_type alpha-iterative \
  --save_name "$ALPHA_TRAIN_NAME"

echo "=== Stage 2b: parameterized single-behavior sweep ==="
python -u ../experiments/train_single_behavior.py \
  --model_path "$MODEL_PATH" \
  --activations_name "$ACTIVATIONS_NAME" \
  --dataset_subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --steering_type parameterized \
  --save_name "$PARAM_TRAIN_NAME"

echo "=== Stage 3: fit intervals ==="
python -u ../experiments/compute_intervals.py \
  --training_experiments \
    "alpha-iterative:${ALPHA_TRAIN_NAME}" \
    "parameterized:${PARAM_TRAIN_NAME}" \
  --behaviors "${BEHAVIORS[@]}" \
  --layer "$LAYER" \
  --gamma 0.1 \
  --save_name "$INTERVAL_MAP_NAME"

echo "=== Stage 4: joint steering at normalized alpha=1 ==="
EXPERIMENT_NAME="${DATE}_mat_multi_attribute_alpha_one"
python -u ../experiments/run_multi_attribute_alpha_one.py \
  --model_path "$MODEL_PATH" \
  --activations_name "$ACTIVATIONS_NAME" \
  --dataset_subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --target_classes "${TARGET_CLASSES[@]}" \
  --test_behaviors "${BEHAVIORS[@]}" \
  --interval_map_name "$INTERVAL_MAP_NAME" \
  --alpha --parameterized \
  --save_name "$EXPERIMENT_NAME"

# Stages 5a-5c produce the remaining rows of the table with the same driver.

echo "=== Stage 5a: No-steering baseline ==="
python -u ../experiments/run_multi_attribute_alpha_one.py \
  --model_path "$MODEL_PATH" \
  --activations_name "$ACTIVATIONS_NAME" \
  --dataset_subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --target_classes "${TARGET_CLASSES[@]}" \
  --test_behaviors "${BEHAVIORS[@]}" \
  --include_no_steer \
  --save_name "${DATE}_mat_no_steer_baseline"

echo "=== Stage 5b: unclipped alpha-iterative multi-steering (literal alpha=1, no interval map) ==="
python -u ../experiments/run_multi_attribute_alpha_one.py \
  --model_path "$MODEL_PATH" \
  --activations_name "$ACTIVATIONS_NAME" \
  --dataset_subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --target_classes "${TARGET_CLASSES[@]}" \
  --test_behaviors "${BEHAVIORS[@]}" \
  --alpha \
  --save_name "${DATE}_mat_unclipped_multi_alpha_one"

echo "=== Stage 5c: unclipped alpha-iterative single-steering (one vector at a time, literal alpha=1) ==="
for B in "${BEHAVIORS[@]}"; do
  python -u ../experiments/run_multi_attribute_alpha_one.py \
    --model_path "$MODEL_PATH" \
    --activations_name "$ACTIVATIONS_NAME" \
    --dataset_subfolder "$DATASET_SUBFOLDER" \
    --layer "$LAYER" \
    --target_classes "$B" \
    --test_behaviors "$B" \
    --alpha \
    --save_name "${DATE}_mat_unclipped_single_${B}_alpha_one"
done

echo "=== Done: results in ../results/logit_results/${DATE}_mat_*.pkl ==="

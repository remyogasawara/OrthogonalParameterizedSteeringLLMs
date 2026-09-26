#!/bin/bash
#SBATCH --job-name=layer_sweep
#SBATCH --partition=<SLURM_PARTITION>
#SBATCH --account=<SLURM_ACCOUNT>

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --time=48:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Qwen3-8B layer sweep (how layer 22 was chosen)
# -----------------------------------------------------------------------------
# Paper result: the choice of steering layer for Qwen 3 8B ("using layer 22
#   after performing a sweep on the layers", section Preliminaries). Layer 22 is
#   the layer used for every Qwen number in the appendix "Replicating Results on
#   Qwen 3.8" (\label{app:qwen_results}); those tables are produced by
#   scripts/run_qwen_pipeline.sh.
#
# What it does, for each zero-indexed layer in LAYERS = 12, 14, ..., 28:
#   Stage 1  experiments/new_get_activations.py: activations and
#            difference-of-means steering vectors for the six xrisk behaviors.
#   Stage 2  experiments/test_single_behavior.py, once with alpha-iterative and
#            once with parameterized steering: one behavior steered at a time,
#            alpha from -2.0 to 2.0 in steps of 0.1, scored on the 200 held-out
#            test items per behavior (the script's defaults, left unchanged).
# After the loop:
#   Stage 3  src/layer_evaluation.py ranks the layers from the alpha-iterative
#            sweep: per behavior, avg_score(alpha=+1) - avg_score(0) and
#            avg_score(-1) - avg_score(0); layers are printed in order of the
#            mean directional effect over the six behaviors, i.e. the mean of
#            (delta(+1) - delta(-1)) / 2. Layer 22, the layer the paper
#            uses, is expected to rank first.
#
# How to run (from the scripts/ directory; one GPU; about 30 GPU-hours):
#   Fill in <SLURM_PARTITION> and <SLURM_ACCOUNT> above, or pass
#   --partition=... --account=... to sbatch (command-line options override them).
#   cd scripts
#   mkdir -p logs          # Slurm opens logs/ before this script starts
#   sbatch layer_sweep_qwen.sh      # or: bash layer_sweep_qwen.sh
#
# Writes (paths relative to the repository root; <DATE> is the start date):
#   activations/<DATE>_Qwen3-8B_<behaviors>_layer<L>.pkl
#   results/logit_results/<DATE>_Qwen3-8B_alpha-iterative_training_experiment_layer<L>.pkl
#   results/logit_results/<DATE>_Qwen3-8B_parameterized_training_experiment_layer<L>.pkl
#   results/layer_sweep/<DATE>_<behaviors>_layer_sweep_summary.csv  (index of the files above)
#   results/layer_sweep/Qwen_Qwen3-8B/avg_score_eval/  layer_sweep_avg_score_by_behavior.csv,
#       layer_sweep_avg_score_macro.csv, best_layers_by_behavior.csv,
#       layer_sweep_avg_score.png (per-behavior curves), layer_sweep_avg_score_macro.png
#       (each plot also as .svg),
#       layer_sweep_avg_score_metadata.json; the ranking is printed at the end of the log.
# =============================================================================

# -------------------------
# Environment setup
# Run from the scripts/ directory: sbatch layer_sweep_qwen.sh  (or: bash layer_sweep_qwen.sh)
# -------------------------
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate steering
fi

set -euo pipefail  # stop the whole chain if any stage fails; later stages depend on earlier outputs

mkdir -p logs
command -v python >/dev/null 2>&1 || { echo "ERROR: python not found; activate the 'steering' environment first" >&2; exit 1; }

# Optional: a file that exports HF_TOKEN; alternatively run `hf auth login` or export HF_TOKEN yourself.
# Qwen/Qwen3-8B is not gated, so no token is required.
if [ -f "$HOME/.hf_token" ]; then source "$HOME/.hf_token"; fi

# -------------------------
# Run config
# -------------------------
DATE="$(date +%-m-%-d-%Y)"
MODEL_PATH="Qwen/Qwen3-8B"
MODEL_NAME="Qwen3-8B"


DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"
BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH myopic-reward survival-instinct power-seeking-inclination wealth-seeking-inclination)


# Layers to sweep. Qwen3-8B has 36 layers; literature on this model tends to
# land steering effects deeper than proportional-to-Llama estimates would
# suggest (roughly layers 14-25), so we sweep a wider band around that.
LAYERS=(12 14 16 18 20 22 24 26 28)

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

    # --skip-auth-check only skips the Hugging Face login check, which would
    # fail without a login even though Qwen3-8B is public. It does not change
    # any result.
    echo "=== [layer ${LAYER}] Stage 1: compute activations ==="
    python -u ../experiments/new_get_activations.py "$MODEL_PATH" \
        --behaviors "${BEHAVIORS[@]}" \
        --dataset-subfolder "$DATASET_SUBFOLDER" \
        --layer "$LAYER" \
        --save-name "$ACTIVATIONS_NAME" \
        --skip-auth-check

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

# Rank the layers from the alpha-iterative sweep (CPU only, a few seconds).
# To also inspect the parameterized sweep, rerun with
#   --pattern "${DATE}_${MODEL_NAME}_parameterized_training_experiment_layer*.pkl"
# and a different --output-dir.
EVAL_DIR="${SWEEP_DIR}/Qwen_Qwen3-8B/avg_score_eval"
echo "=== Stage 3: rank layers (src/layer_evaluation.py) ==="
python -u ../src/layer_evaluation.py \
    --input ../results/logit_results \
    --pattern "${DATE}_${MODEL_NAME}_alpha-iterative_training_experiment_layer*.pkl" \
    --baseline-alpha 0 \
    --positive-alpha 1 \
    --negative-alpha -1 \
    --score-source stored \
    --output-dir "$EVAL_DIR" \
    --title "Per-layer Average Score Effect: ${MODEL_NAME}"

echo "=== Layer ranking written to ${EVAL_DIR}; the ranking by mean directional effect is printed above. ==="

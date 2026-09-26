# Orthogonal Parameterized Steering

> **Anonymous ICLR submission.** This repository contains the code used for the accompanying anonymous manuscript. Author names, affiliations, personal links, and citation metadata are intentionally omitted during peer review.

This repository studies interpretable control of multiple large-language-model behaviors through activation steering. It implements conventional additive steering, parameterized steering, and orthogonalized variants designed to reduce interference between steering directions.

The primary entry point is the end-to-end pipeline:

```text
scripts/run_multi_steering_pipeline.sh
```

The pipeline is configurable and can be run with a supported Hugging Face causal language model and any dataset that follows the repository's behavior-dataset format.

## Overview

The full pipeline:

1. extracts hidden activations for the requested behaviors;
2. constructs behavior steering directions;
3. evaluates each behavior over an alpha-iterative steering sweep;
4. evaluates each behavior over a parameterized steering sweep;
5. fits behavior-specific steerable intervals; and
6. evaluates two selected behaviors under simultaneous multi-attribute steering.

The final evaluation is run once on each selected behavior's test set so that the effect of jointly steering the pair can be measured in both directions.

## Steering methods

The repository includes the following methods:

- **Alpha-iterative steering:** adds a scaled steering direction to an activation.
- **Parameterized steering:** maps a normalized steering value to a behavior-specific location defined relative to the positive and negative activation distributions.
- **Orthogonal alpha-iterative steering:** applies additive steering after removing overlap between the selected steering directions.
- **Orthogonal parameterized steering:** combines behavior-specific parameterization with orthogonalized multi-attribute steering.

## Repository structure

```text
.
├── activations/                  # Generated activations and steering vectors
├── datasets/                     # Behavior datasets
├── experiments/
│   ├── new_get_activations.py
│   ├── train_single_behavior.py
│   ├── compute_intervals.py
│   └── run_multi_attribute_experiment.py
├── results/
│   ├── intervals/                # Fitted behavior-specific interval maps
│   └── logit_results/            # Single-behavior evaluations; multi-attribute runs in logit_results/alpha_beta/
├── scripts/
│   └── run_multi_steering_pipeline.sh
├── src/                          # Dataset, model, steering, and evaluation utilities
├── paper_figures/                # Rebuilds every figure in the paper from committed data (no GPU)
├── dataset_generation/           # Warmth dataset, generation prompts and validation (Appendix G)
└── environment.yaml              # Conda environment specification
```

Generated files are generally not required to be committed to the repository.

## Requirements

The default Slurm job requests:

- one CUDA-capable GPU;
- 16 GB of host memory;
- four CPU tasks; and
- up to 96 hours of wall-clock time.

Actual memory and runtime requirements depend on the model, number of behaviors, dataset size, intervention layer, and alpha-grid resolution.

The default example uses:

```text
meta-llama/Llama-2-7b-chat-hf
```

This model requires access through Hugging Face. A different compatible causal language model can be selected in the pipeline configuration.

## Installation

Clone the anonymous repository and enter its root directory:

```bash
git clone <ANONYMOUS_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

Create the Conda environment:

```bash
conda env create -f environment.yaml
conda activate steering
```

To update an existing environment:

```bash
conda env update -f environment.yaml --prune
conda activate steering
```

## Hugging Face authentication

Authenticate before loading a gated model:

```bash
hf auth login
```

Alternatively, expose a token through the environment:

```bash
export HF_TOKEN="<YOUR_HUGGING_FACE_TOKEN>"
```

The job script can also source a private token file at `~/.hf_token`. For example:

```bash
export HF_TOKEN="<YOUR_HUGGING_FACE_TOKEN>"
```

Never commit a Hugging Face token or another credential to the repository.

## Dataset configuration

Set `DATASET_SUBFOLDER` to a path relative to `datasets/`:

```bash
DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"
```

This corresponds to:

```text
datasets/tan_paper_datasets/mwe/xrisk/
```

To run on another dataset, replace it with the desired relative path:

```bash
DATASET_SUBFOLDER="my_dataset"
```

which resolves to:

```text
datasets/my_dataset/
```

A custom dataset must follow the same schema expected by the repository's dataset loader. The safest way to add a dataset is to copy one of the included dataset directories, preserve its file structure and fields, and replace the examples with data for the new behaviors.

Behavior identifiers in the script must exactly match the identifiers exposed by the selected dataset. For example:

```bash
BEHAVIORS=(
  coordinate-other-ais
  corrigible-neutral-HHH
  myopic-reward
  survival-instinct
  power-seeking-inclination
  wealth-seeking-inclination
)
```

`BEHAVIORS` controls which directions are extracted, swept, and included when fitting the interval map. It may contain two or more behaviors.

## Configure an experiment

Edit the run configuration near the top of:

```text
scripts/run_multi_steering_pipeline.sh
```

### Model and intervention layer

```bash
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME="Llama-2-7b-chat-hf"
LAYER=13
```

- `MODEL_PATH` is the Hugging Face model identifier or a local model path.
- `MODEL_NAME` is a filesystem-safe name used in generated filenames.
- `LAYER` is the transformer layer at which activations are extracted and steering is applied.

When changing models, select a valid layer index for that architecture. The layer used for one model should not be assumed to be optimal for another model.

### Dataset and behavior set

```bash
DATASET_SUBFOLDER="tan_paper_datasets/mwe/xrisk"

BEHAVIORS=(
  coordinate-other-ais
  corrigible-neutral-HHH
  myopic-reward
  survival-instinct
  power-seeking-inclination
  wealth-seeking-inclination
)
```

### Multi-attribute behavior pair

Choose the two behaviors to steer jointly:

```bash
TEST_BEHAVIOR1="coordinate-other-ais"
TEST_BEHAVIOR2="power-seeking-inclination"
TARGET_CLASSES=("$TEST_BEHAVIOR1" "$TEST_BEHAVIOR2")
```

Both selected behaviors must also appear in `BEHAVIORS`.

`TARGET_CLASSES` is required by the final multi-attribute commands. Ensure that this array is defined before the generated experiment names and before Stage 4. Without it, `set -u` causes the script to terminate when it reaches `--target_classes`.

### Alpha sweep

The default single-behavior sweeps use:

```bash
--alpha-min -2.0 \
--alpha-max 2.0 \
--alpha-step 0.25
```

The interval-fitting stage must use the same step size:

```bash
--step 0.25
```

When changing the sweep resolution, update all three commands consistently.

## Slurm configuration

Set a job name and fill in the cluster-specific partition and account:

```bash
#SBATCH --job-name=multi_steering_pipeline
#SBATCH --partition=<SLURM_PARTITION>
#SBATCH --account=<SLURM_ACCOUNT>
```

Initialize Conda using a path valid on the target system. A portable option is:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate steering
```

If `conda` is not initially available in batch jobs, replace this with the appropriate cluster module or absolute Conda initialization path.

Create the log directory before submitting the first job:

```bash
mkdir -p scripts/logs
```

## Run the complete pipeline

The script uses paths such as `../experiments/...`, so submit it from the `scripts/` directory:

```bash
cd scripts
sbatch run_multi_steering_pipeline.sh
```

For an interactive allocation or a non-Slurm machine with a configured GPU environment:

```bash
cd scripts
bash run_multi_steering_pipeline.sh
```

Slurm writes standard output and error logs to:

```text
scripts/logs/<JOB_NAME>_<JOB_ID>.out
scripts/logs/<JOB_NAME>_<JOB_ID>.err
```

The script uses:

```bash
set -euo pipefail
```

Therefore, it stops immediately if a stage fails or if a required variable is undefined. Later stages are not run with incomplete inputs.

## Pipeline stages

### Stage 1: extract activations

```bash
python -u ../experiments/new_get_activations.py "$MODEL_PATH" \
  --behaviors "${BEHAVIORS[@]}" \
  --dataset-subfolder "$DATASET_SUBFOLDER" \
  --layer "$LAYER" \
  --save-name "$ACTIVATIONS_NAME"
```

This stage loads the selected dataset, records hidden activations at the intervention layer, and computes the information required to construct behavior steering directions.

### Stage 2a: alpha-iterative single-behavior sweep

```bash
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
```

This stage evaluates conventional additive steering separately for every behavior over the configured alpha grid.

### Stage 2b: parameterized single-behavior sweep

```bash
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
```

This stage evaluates parameterized steering separately for every behavior over the same normalized alpha grid.

### Stage 3: fit steerable intervals

```bash
python -u ../experiments/compute_intervals.py \
  --training_experiments \
    "alpha-iterative:${ALPHA_TRAIN_NAME}" \
    "parameterized:${PARAM_TRAIN_NAME}" \
  --behaviors "${BEHAVIORS[@]}" \
  --layer "$LAYER" \
  --step 0.25 \
  --save_name "$INTERVAL_MAP_NAME"
```

This stage uses the single-behavior sweeps to estimate behavior-specific steering intervals and stores the mappings needed for normalized parameterized steering.

### Stage 4a: jointly steer the pair and evaluate behavior 1

```bash
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
```

### Stage 4b: jointly steer the pair and evaluate behavior 2

```bash
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
```

The selected pair is held fixed in both commands. Only the evaluation dataset changes, allowing the same joint steering intervention to be measured with respect to each target behavior.

## Example: use another dataset and behavior pair

Suppose a compatible dataset is stored at:

```text
datasets/custom_behaviors/
```

and exposes the behaviors `warmth`, `sycophancy`, and `formality`. Configure:

```bash
DATASET_SUBFOLDER="custom_behaviors"

BEHAVIORS=(
  warmth
  sycophancy
  formality
)

TEST_BEHAVIOR1="warmth"
TEST_BEHAVIOR2="sycophancy"
TARGET_CLASSES=("$TEST_BEHAVIOR1" "$TEST_BEHAVIOR2")
```

The pipeline will extract all three directions and fit intervals for all three, then jointly steer and evaluate the selected `warmth`/`sycophancy` pair.

## Generated names and outputs

The pipeline constructs deterministic names from the model and behavior list:

```bash
ACTIVATIONS_NAME="${MODEL_NAME}_<BEHAVIORS>"
ALPHA_TRAIN_NAME="${MODEL_NAME}_<BEHAVIORS>_alpha-iterative_training_experiment_large_intervals"
PARAM_TRAIN_NAME="${MODEL_NAME}_<BEHAVIORS>_parameterized_training_experiment_large_intervals"
INTERVAL_MAP_NAME="${MODEL_NAME}_<BEHAVIORS>_large_interval_map"
```

The two final experiment names include the execution date and selected behavior order:

```text
<DATE>_<BEHAVIOR_1>_<BEHAVIOR_2>_multi_attribute_experiment
<DATE>_<BEHAVIOR_2>_<BEHAVIOR_1>_multi_attribute_experiment
```

The script reports final experiment files under:

```text
results/logit_results/alpha_beta/
```

Intermediate outputs are written to the repository's activation, interval, and result directories using the generated names above.

## Inspect final results

The experiment files are Python pickles. A typical result can be inspected from the repository root, inside the `steering` environment (loading a result imports the repository's `src` package), with:

```python
import pickle
from pathlib import Path

result_dir = Path("results/logit_results/alpha_beta")
result_paths = sorted(result_dir.glob("*_multi_attribute_experiment.pkl"))

if not result_paths:
    raise FileNotFoundError(
        "No multi-attribute experiment files were found in "
        f"{result_dir.resolve()}"
    )

for result_path in result_paths:
    with result_path.open("rb") as handle:
        experiment = pickle.load(handle)

    print(f"\nLoaded: {result_path}")

    if hasattr(experiment, "to_dataframe"):
        # include_eval_details=False drops the per-question lists and keeps one row per setting
        print(experiment.to_dataframe(include_eval_details=False).to_string(index=False))
    else:
        print(experiment)
```

Pickle files should only be loaded from trusted sources.

## Reproducing the paper's results

All commands are run from `scripts/` unless stated otherwise. Every `.sh` script has a header that says which result it produces, what it writes, and how to aggregate it. Fill in `<SLURM_PARTITION>` and `<SLURM_ACCOUNT>` (or pass `--partition`/`--account` to `sbatch`), or run the script with `bash` on a GPU machine.

| Paper result | Command | Output |
|---|---|---|
| All figures | `cd paper_figures && python make_figs.py` (no GPU; see `paper_figures/README.md`) | `paper_figures/out/*.pdf` |
| Sycophancy x warmth results (Table 1) | `sbatch run_sycophancy_warmth_pipeline.sh` | `results/logit_results/alpha_beta/<DATE>_{sycophancy_warmth,warmth_sycophancy}_multi_attribute_experiment.pkl` |
| Multi-steering sensitivity and range (Tables 2 and 3) and the `percent_steered` versions in the appendix | Stages 1-3 of `run_multi_steering_pipeline.sh`, then `sbatch run_alpha_beta_pairs.sh` and `sbatch run_alpha_beta_pairs_flipped_param.sh`; aggregate with `src/mean_sd.py`, `src/parameterized_mean_sd.py` and `src/fixed_endpoint_range.py` (exact commands in the two script headers) | `results/logit_results/alpha_beta/` (14 runs) and `.../alpha_beta/flipped_param/` (28 runs); tables in `results/tables/` |
| Comparison to MAT-Steer (Table 4; Llama 3.1 8B Instruct, layer 14) | `python experiments/prepare_mat_steer_datasets.py --output-dir datasets/mat_steer` from the repository root, then `sbatch run_mat_steer_pipeline.sh` | `results/logit_results/<DATE>_mat_*.pkl` |
| Qwen 3 8B appendix: layer choice and range tables | `sbatch layer_sweep_qwen.sh` (ranks layers with `src/layer_evaluation.py`), then `sbatch run_qwen_pipeline.sh` | `results/layer_sweep/Qwen_Qwen3-8B/`; `results/logit_results/alpha_beta/<DATE>_Qwen3-8B_*` |
| LLM-as-a-judge appendix | `python experiments/run_openended_alpha_beta_pairs.py ...` (arguments in its header), then `export OPENAI_API_KEY=...` and `python experiments/score_openended_with_judge.py` | `results/openended/`; `results/llm_judge/*.csv` |
| Warmth dataset creation and validation table (appendix) | `cd dataset_generation && python run_experiments.py --data data/warmth_families.jsonl` (no GPU; see `dataset_generation/README.md`) | `dataset_generation/results/` |

Notes on matching the reported numbers:

- **Interval grid.** The paper's interval maps were fitted on the single-behaviour sweep at the Python defaults (`train_single_behavior.py --alpha-step 0.1`, 41 values in [-2, 2]; `compute_intervals.py --step 0.1 --gamma 0.1`) and saved as `<MODEL>_<BEHAVIORS>_interval_map`. The pipeline script above uses step 0.25 and saves `..._large_interval_map`. To reproduce the tables exactly, run Stages 2-3 with step 0.1 and pass the map name to the pair scripts with `INTERVAL_MAP_NAME=<name>`.
- **Traced figure data.** The before/after-clipping curves and the alpha-iterative vs parameterized curves are replotted from data traced out of the original figure files, because their experiment outputs were not archived (see `paper_figures/README.md`).
- **Columns without a script.** `src/fixed_endpoint_range.py` reports the OAI and AI columns of the range tables. The OP and P columns can be computed from the same result files, but no script prints them yet.
- **Unrecorded settings.** The MAT-Steer baseline rows, the Qwen multi-attribute runs and the open-ended judge runs were not recorded as job scripts. The scripts above reconstruct them from the code and the paper; each header lists what was assumed.
- **Sampling.** Open-ended generation is sampled (temperature 0.1) and the judge model is not deterministic, so a rerun of the LLM-as-a-judge appendix will be close to, but not identical with, the reported values.

## Run or resume individual stages

Each pipeline stage can be run independently using the commands above. To resume a partially completed experiment:

1. verify that all expected outputs from the previous stages exist;
2. comment out only the completed stages; and
3. retain the exact same generated names, behavior order, dataset path, model, and layer.

Stage 3 requires the outputs of both Stage 2 sweeps. Stage 4 requires the activation output and interval map.

## Reproducibility notes

- Keep the model revision, software environment, dataset contents, behavior order, layer, and alpha grid fixed when reproducing reported numbers.
- The order of `BEHAVIORS` is incorporated into generated filenames; changing it creates a different name even when the set of behaviors is unchanged.
- Exact numerical results can vary across GPU architectures, CUDA versions, PyTorch versions, and model revisions.
- The date prefix can cause same-day runs with identical behavior pairs to overwrite one another. Add a descriptive suffix when retaining multiple runs.
- Large models or larger datasets may require more GPU memory, host memory, or wall-clock time than the default Slurm request.


## Anonymous-review notice

Citation information and author contact details will be added after the anonymous review period. During review, questions should be submitted through the conference review system rather than through channels that could reveal author identity.

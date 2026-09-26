"""Evaluate joint multi-attribute steering once at alpha=1.

Paper result: Table tab:matsteer_compare ("Percent_steered across bias datasets and
steering methods compared with MAT-Steer", subsection "Comparison to MAT-Steer"),
i.e. the BBQ / Toxigen / TruthfulQA columns of our rows. The MAT-Steer rows of that
table are quoted from the MAT-Steer paper and are not produced here.

Setup used for the table: model meta-llama/Llama-3.1-8B-Instruct, layer 14 (zero-indexed),
datasets/mat_steer/{bbq,truthfulqa,toxigen}.jsonl (built by
experiments/prepare_mat_steer_datasets.py), 200 held-out test items per dataset
(DataSet shuffle seed 42), difference-of-means vectors at the answer token, logit scoring
over the entire answer sequence, metric percent_steered.

All vectors named by --target_classes are applied simultaneously. Each dataset named by
--test_behaviors is evaluated exactly once per selected steering method.

Which table row comes from which call (scripts/run_mat_steer_pipeline.sh runs all of them):
  * alpha-iterative, Parameterized, Orthogonal alpha-iterative, Orthogonal Parameterized:
      --target_classes bbq truthfulqa toxigen --interval_map_name <map> --alpha --parameterized
      (normalized alpha=1 is mapped through each behavior's fitted interval)
  * Unclipped alpha-iterative multi-steering:
      same target classes, --alpha, no --interval_map_name (literal alpha=1);
      read the "alpha-iterative" rows
  * Unclipped alpha-iterative single-steering:
      --target_classes B --test_behaviors B --alpha, no --interval_map_name, once per B
  * No-steering baseline: --include_no_steer (adds an unsteered "no_steer" row per test set)

How to run (from scripts/; needs one GPU and access to the gated Llama 3.1 weights):
    python -u ../experiments/run_multi_attribute_alpha_one.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --activations_name Llama-3.1-8B-Instruct_bbq_truthfulqa_toxigen \
        --dataset_subfolder mat_steer \
        --layer 14 \
        --target_classes bbq truthfulqa toxigen \
        --test_behaviors bbq truthfulqa toxigen \
        --interval_map_name Llama-3.1-8B-Instruct_bbq_truthfulqa_toxigen_interval_map \
        --alpha --parameterized \
        --save_name mat_steer_alpha_one

Inputs: activations/<activations_name>.pkl (experiments/new_get_activations.py) and, when
--interval_map_name is given, results/intervals/<interval_map_name>.pkl
(experiments/compute_intervals.py).
Writes: results/logit_results/<save_name>.pkl, an ExperimentOutput pickle. In
.to_dataframe(), "estimator" is the steering method, "behavior" is the test set,
and percent_steered * 100 is the table entry.
"""

import argparse
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import DataSet
from src.experiment_output import ExperimentOutput
from src.multi_attribute_steering import (
    get_steering_types_set,
    # Multi-vector evaluation helper. It was called
    # grab_results_and_evaluate_steering_multi in an earlier version of the code.
    grab_results_and_evaluate_steering,
)
from src.steerable_model import SteerableModel
from src.utils import set_global_seed


ALPHA = 1.0
TOKEN_POSITION = "answer_token"
ESTIMATOR = "sample_diff_of_means"
NO_STEER = "no_steer"


def mapped_alphas(
    target_classes: List[str],
    interval_mapping: Optional[Dict],
) -> Dict[str, float]:
    """Map normalized alpha=1 through fitted intervals, or use literal 1.

    Returns {behavior: alpha}: the steering hook in src/multi_attribute_steering.py
    looks alphas up by behavior name.
    """
    if interval_mapping is None:
        return {behavior: ALPHA for behavior in target_classes}

    values = {}
    for behavior in target_classes:
        behavior_map = interval_mapping.get(behavior)
        if behavior_map is None:
            raise KeyError(f"Interval map has no entry for behavior {behavior!r}")

        # Pickle maps may use either 1 or 1.0; they compare equally in normal dicts.
        if ALPHA not in behavior_map:
            available = sorted(behavior_map)
            raise KeyError(
                f"Interval map for {behavior!r} has no alpha={ALPHA}; "
                f"available keys include {available[:10]}"
            )
        values[behavior] = float(behavior_map[ALPHA])
    return values


def evaluate_target_classes_at_alpha_one(
    model,
    steering_vecs,
    mean_as,
    mean_bs,
    steering_types,
    test_data_dict,
    layer: int,
    target_classes: List[str],
    test_behaviors: List[str],
    batch_size: int,
    behavior_alpha_mapping=None,
    behavior_alpha_mapping_parameterized=None,
) -> ExperimentOutput:
    """Apply all target classes and test each requested behavior exactly once."""
    experiment = ExperimentOutput()

    missing_vectors = [x for x in target_classes if x not in steering_vecs]
    missing_means = [x for x in target_classes if x not in mean_as or x not in mean_bs]
    missing_tests = [x for x in test_behaviors if x not in test_data_dict]
    if missing_vectors:
        raise KeyError(f"Missing steering vectors for: {missing_vectors}")
    if missing_means:
        raise KeyError(f"Missing positive/negative means for: {missing_means}")
    if missing_tests:
        raise KeyError(f"Missing test datasets for: {missing_tests}")

    # Validate the shared activation location once. All jointly steered vectors must have it.
    for behavior in target_classes:
        try:
            steering_vecs[behavior][layer][TOKEN_POSITION][ESTIMATOR]
        except KeyError as exc:
            raise KeyError(
                f"{behavior!r} does not contain vector "
                f"[{layer!r}][{TOKEN_POSITION!r}][{ESTIMATOR!r}]"
            ) from exc

    raw_alpha_alphas = mapped_alphas(target_classes, behavior_alpha_mapping)
    parameterized_alphas = mapped_alphas(
        target_classes, behavior_alpha_mapping_parameterized
    )

    for steering_type in steering_types:
        if steering_type == NO_STEER:
            # No hook is installed, so record alpha 0 rather than the steered values.
            applied_alphas = {behavior: 0.0 for behavior in target_classes}
            row_alpha = 0.0
        else:
            applied_alphas = (
                parameterized_alphas
                if "parameterized" in steering_type
                else raw_alpha_alphas
            )
            row_alpha = ALPHA
        applied_alpha_values = tuple(applied_alphas[b] for b in target_classes)

        if steering_type == NO_STEER:
            print(f"Steering method={steering_type}; no steering applied")
        else:
            print(
                f"Steering method={steering_type}; target_classes={target_classes}; "
                f"normalized_alpha={ALPHA}; applied_alphas={list(applied_alpha_values)}"
            )

        # This argument is retained because the repository helper uses it to determine
        # whether steering is active. The actual multi-vector hook uses all_steering_dir,
        # behavior_names, and alphas. For the no_steer baseline no hook is installed.
        reference_steering_dir = (
            None
            if steering_type == NO_STEER
            else steering_vecs[target_classes[0]][layer][TOKEN_POSITION]
        )

        for test_behavior in test_behaviors:
            print(f"Evaluating joint intervention on {test_behavior} test data")
            result_and_eval = grab_results_and_evaluate_steering(
                model=model,
                test_data=test_data_dict[test_behavior],
                all_steering_dir=steering_vecs,
                steering_dir=reference_steering_dir,
                meanAs=mean_as,
                meanBs=mean_bs,
                steering_type=steering_type,
                intervention_layers=[layer],
                alphas=applied_alphas,
                behavior_names=target_classes,
                open_ended=False,
                gpt_client=None,
                logits_only=True,
                logit_aggregation_method="entire_sequence",
                generation_batch_size=batch_size,
                generation_max_tokens=64,
                token_pos=TOKEN_POSITION,
                estimator_name=ESTIMATOR,
            )

            # Use the tested behavior as the primary row label. The full intervention is
            # recorded explicitly in additional_kwargs.
            display_vector = steering_vecs[test_behavior][layer][TOKEN_POSITION][ESTIMATOR]
            experiment.add_result(
                behavior=test_behavior,
                layer=layer,
                token_pos=TOKEN_POSITION,
                estimator=steering_type,
                eval_dict=result_and_eval["eval"],
                alpha=row_alpha,
                steering_vec=display_vector,
                raw_results=result_and_eval["results"],
                additional_kwargs={
                    "test_behavior": test_behavior,
                    "target_classes": tuple(target_classes),
                    "applied_alphas": applied_alpha_values,
                },
            )

            evaluation = result_and_eval["eval"]
            print(
                f"{test_behavior}: percent_steered={evaluation['percent_steered']}; "
                f"avg_score={evaluation['avg_score']}"
            )

    return experiment


def print_summary(experiment: ExperimentOutput, test_behaviors: List[str]) -> None:
    """Print percent_steered * 100 per method and test set, as reported in the table."""
    table = {}
    for output in experiment.outputs:
        table.setdefault(output["estimator"], {})[output["behavior"]] = (
            100.0 * output["percent_steered"]
        )
    print("percent_steered (%)")
    print("method".ljust(30) + "".join(b.rjust(12) for b in test_behaviors))
    for method, row in table.items():
        cells = "".join(
            (f"{row[b]:.1f}" if b in row else "-").rjust(12) for b in test_behaviors
        )
        print(method.ljust(30) + cells)


def main(args):
    activations_path = os.path.join(
        PROJECT_ROOT, "activations", f"{args.activations_name}.pkl"
    )
    with open(activations_path, "rb") as handle:
        activations = pickle.load(handle)

    steering_vecs = activations.steering_vectors
    mean_as = activations.pos_act_means
    mean_bs = activations.neg_act_means

    dataset = DataSet(
        subfolders=[args.dataset_subfolder],
        test_size=args.test_size,
    )
    model = SteerableModel(model_name=args.model_path)
    print("Successfully loaded model")

    steering_types = get_steering_types_set(
        alpha=args.use_alpha,
        parameterized=args.use_parameterized,
    )
    if args.include_no_steer:
        steering_types = [NO_STEER] + steering_types

    alpha_mapping = None
    parameterized_mapping = None
    if args.interval_map_name:
        interval_path = os.path.join(
            PROJECT_ROOT,
            "results",
            "intervals",
            f"{args.interval_map_name}.pkl",
        )
        with open(interval_path, "rb") as handle:
            mappings = pickle.load(handle)
        alpha_mapping = mappings.get("alpha-iterative")
        parameterized_mapping = mappings.get("parameterized")

    print(
        f"Running alpha={ALPHA} joint experiment: "
        f"target_classes={args.target_classes}, "
        f"test_behaviors={args.test_behaviors}, "
        f"steering_types={steering_types}"
    )
    start = time.time()
    experiment = evaluate_target_classes_at_alpha_one(
        model=model,
        steering_vecs=steering_vecs,
        mean_as=mean_as,
        mean_bs=mean_bs,
        steering_types=steering_types,
        test_data_dict=dataset.test_data,
        layer=args.layer,
        target_classes=args.target_classes,
        test_behaviors=args.test_behaviors,
        batch_size=args.batch_size,
        behavior_alpha_mapping=alpha_mapping,
        behavior_alpha_mapping_parameterized=parameterized_mapping,
    )
    print(f"Evaluation completed in {time.time() - start:.2f}s")
    print_summary(experiment, args.test_behaviors)

    output_dir = os.path.join(PROJECT_ROOT, "results", "logit_results")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{args.save_name}.pkl")
    with open(save_path, "wb") as handle:
        pickle.dump(experiment, handle)
    print(f"Saved experiment to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply all target classes and evaluate once at alpha=1"
    )
    parser.add_argument(
        "--model_path",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--activations_name", required=True)
    parser.add_argument("--dataset_subfolder", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--target_classes", nargs="+", required=True)
    parser.add_argument("--test_behaviors", nargs="+", required=True)
    parser.add_argument(
        "--interval_map_name",
        default=None,
        help=(
            "Optional interval-map pickle. When supplied, normalized alpha=1 is mapped "
            "to each behavior's fitted value. Omit it to apply literal alpha=1."
        ),
    )
    parser.add_argument("--alpha", dest="use_alpha", action="store_true")
    parser.add_argument(
        "--parameterized", dest="use_parameterized", action="store_true"
    )
    parser.add_argument(
        "--include_no_steer",
        action="store_true",
        help="Also evaluate each test set with no steering (the No-steering baseline row).",
    )
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=200)
    args = parser.parse_args()
    if not (args.use_alpha or args.use_parameterized or args.include_no_steer):
        parser.error("Pass at least one of --alpha, --parameterized or --include_no_steer")
    return args


if __name__ == "__main__":
    set_global_seed(42)
    main(parse_args())

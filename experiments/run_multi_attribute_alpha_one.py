"""Evaluate joint multi-attribute steering once at alpha=1.

All vectors named by --target_classes are applied simultaneously. Each dataset named by
--test_behaviors is evaluated exactly once per selected steering method.

Example (run from scripts/):
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
    grab_results_and_evaluate_steering_multi,
)
from src.steerable_model import SteerableModel
from src.utils import set_global_seed


ALPHA = 1.0
TOKEN_POSITION = "answer_token"
ESTIMATOR = "sample_diff_of_means"


def mapped_alphas(
    target_classes: List[str],
    interval_mapping: Optional[Dict],
) -> List[float]:
    """Map normalized alpha=1 through fitted intervals, or use literal 1."""
    if interval_mapping is None:
        return [ALPHA] * len(target_classes)

    values = []
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
        values.append(float(behavior_map[ALPHA]))
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
        applied_alphas = (
            parameterized_alphas
            if "parameterized" in steering_type
            else raw_alpha_alphas
        )

        print(
            f"Steering method={steering_type}; target_classes={target_classes}; "
            f"normalized_alpha={ALPHA}; applied_alphas={applied_alphas}"
        )

        # This argument is retained because the repository helper uses it to determine
        # whether steering is active. The actual multi-vector hook uses all_steering_dir,
        # behavior_names, and alphas.
        reference_steering_dir = steering_vecs[target_classes[0]][layer][TOKEN_POSITION]

        for test_behavior in test_behaviors:
            print(f"Evaluating joint intervention on {test_behavior} test data")
            result_and_eval = grab_results_and_evaluate_steering_multi(
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
                alpha=ALPHA,
                steering_vec=display_vector,
                raw_results=result_and_eval["results"],
                additional_kwargs={
                    "test_behavior": test_behavior,
                    "target_classes": tuple(target_classes),
                    "applied_alphas": tuple(applied_alphas),
                },
            )

            evaluation = result_and_eval["eval"]
            print(
                f"{test_behavior}: percent_steered={evaluation['percent_steered']}; "
                f"avg_score={evaluation['avg_score']}"
            )

    return experiment


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
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=200)
    args = parser.parse_args()
    if not (args.use_alpha or args.use_parameterized):
        parser.error("Pass at least one of --alpha or --parameterized")
    return args


if __name__ == "__main__":
    set_global_seed(42)
    main(parse_args())

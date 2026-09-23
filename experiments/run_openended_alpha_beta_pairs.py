"""Generate fixed alpha/beta steering outputs without running evaluations."""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import DataSet
from src.multi_attribute_steering import get_steering_results
from src.steerable_model import SteerableModel
from src.utils import set_global_seed

ALPHA_BETA_VALUES = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

TRAIT_PAIRS = [
    ("coordinate-other-ais", "power-seeking-inclination"),
    ("coordinate-other-ais", "corrigible-neutral-HHH"),
    ("power-seeking-inclination", "wealth-seeking-inclination"),
]

STEERING_TYPES = [
    "alpha-iterative",
    "orthogonal-alpha-iterative",
    "orthogonal-parameterized",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses for fixed alpha/beta steering settings"
    )
    parser.add_argument("--model_path", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--activations_name", required=True)
    parser.add_argument("--dataset_subfolder", required=True)
    parser.add_argument("--interval_map_name", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--generation-max-tokens", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--date", default=datetime.now().strftime("%-m-%-d-%Y"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "results", "openended"),
    )
    return parser.parse_args()


def mapped_value(mapping, behavior: str, requested_value: float) -> float:
    behavior_map = mapping[behavior]
    lookup_value = requested_value
    if lookup_value not in behavior_map:
        lookup_value = round(float(requested_value), 1)
    return behavior_map[lookup_value]


def save_results(results: dict, output_path: str) -> None:
    """Atomically checkpoint completed generations."""
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "wb") as f:
        pickle.dump(results, f)
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    set_global_seed(42)

    activations_path = os.path.join(
        PROJECT_ROOT, "activations", f"{args.activations_name}.pkl"
    )
    with open(activations_path, "rb") as f:
        activations = pickle.load(f)

    interval_map_path = os.path.join(
        PROJECT_ROOT, "results", "intervals", f"{args.interval_map_name}.pkl"
    )
    with open(interval_map_path, "rb") as f:
        interval_maps = pickle.load(f)

    alpha_mapping = interval_maps.get("alpha-iterative")
    parameterized_mapping = interval_maps.get("parameterized")
    if alpha_mapping is None:
        raise KeyError(f"Missing 'alpha-iterative' mapping in {interval_map_path}")
    if parameterized_mapping is None:
        raise KeyError(f"Missing 'parameterized' mapping in {interval_map_path}")

    dataset = DataSet(
        subfolders=[args.dataset_subfolder],
        test_size=args.test_size,
    )
    model = SteerableModel(model_name=args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Each undirected pair is run twice, switching which trait is alpha/beta.
    orientations = [
        orientation
        for trait_1, trait_2 in TRAIT_PAIRS
        for orientation in ((trait_1, trait_2), (trait_2, trait_1))
    ]

    for alpha_trait, beta_trait in orientations:
        target_classes = [alpha_trait, beta_trait]
        test_behavior = alpha_trait
        test_data = dataset.test_data.get(test_behavior)
        if test_data is None:
            raise ValueError(f"No test data found for {test_behavior}")

        filename = (
            f"{args.date}_alpha-{alpha_trait}_beta-{beta_trait}_generations.pkl"
        )
        output_path = os.path.join(args.output_dir, filename)

        # Resume this orientation when an existing checkpoint is present.
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                generation_results = pickle.load(f)
            print(f"Resuming {output_path}: {len(generation_results)} runs complete")
        else:
            generation_results = {}

        behavior_vecs = activations.steering_vectors[test_behavior]

        for requested_alpha, requested_beta in ALPHA_BETA_VALUES:
            requested_values = {
                alpha_trait: requested_alpha,
                beta_trait: requested_beta,
            }
            alpha_iterative_values = {
                behavior: mapped_value(alpha_mapping, behavior, value)
                for behavior, value in requested_values.items()
            }
            parameterized_values = {
                behavior: mapped_value(parameterized_mapping, behavior, value)
                for behavior, value in requested_values.items()
            }

            for layer, token_map in behavior_vecs.items():
                if layer != args.layer:
                    continue

                for token_pos, steering_dir in token_map.items():
                    if token_pos != "answer_token":
                        continue

                    for steering_type in STEERING_TYPES:
                        result_key = (
                            requested_alpha,
                            requested_beta,
                            steering_type,
                            layer,
                            token_pos,
                        )
                        if result_key in generation_results:
                            print(f"Skipping completed run: {result_key}")
                            continue

                        steering_alphas = (
                            parameterized_values.copy()
                            if "parameterized" in steering_type
                            else alpha_iterative_values.copy()
                        )

                        print(
                            f"Generating alpha_trait={alpha_trait}, "
                            f"beta_trait={beta_trait}, alpha={requested_alpha}, "
                            f"beta={requested_beta}, method={steering_type}"
                        )
                        start_time = time.time()

                        results = get_steering_results(
                            model=model,
                            test_data=test_data,
                            all_steering_dir=activations.steering_vectors,
                            steering_dir=steering_dir,
                            meanAs=activations.pos_act_means,
                            meanBs=activations.neg_act_means,
                            steering_type=steering_type,
                            behavior_names=target_classes,
                            intervention_layers=[args.layer],
                            alphas=steering_alphas,
                            open_ended=True,
                            logits_only=False,
                            generation_batch_size=args.batch_size,
                            generation_max_tokens=args.generation_max_tokens,
                            token_pos=token_pos,
                            estimator_name="sample_diff_of_means",
                        )

                        generation_results[result_key] = {
                            "results": results,
                            "alpha_trait": alpha_trait,
                            "beta_trait": beta_trait,
                            "test_behavior": test_behavior,
                            "requested_alpha": requested_alpha,
                            "requested_beta": requested_beta,
                            "mapped_alphas": steering_alphas,
                            "steering_type": steering_type,
                            "layer": layer,
                            "token_pos": token_pos,
                        }
                        save_results(generation_results, output_path)
                        print(
                            f"Checkpointed {output_path} "
                            f"({time.time() - start_time:.2f} seconds)"
                        )

        print(f"Completed {alpha_trait} -> {beta_trait}: {output_path}")


if __name__ == "__main__":
    main()

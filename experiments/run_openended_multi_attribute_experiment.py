"""Run an open-ended multi-attribute generation experiment.

This runner intentionally enables only alpha-iterative steering.  The first
class passed via --test_behavior is evaluated while the other target class is
used as the beta behavior.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import DataSet
from src.multi_attribute_steering import evaluate_multi_attribute, get_steering_types_set
from src.steerable_model import SteerableModel
from src.utils import set_global_seed


set_global_seed(42)


def main(args: argparse.Namespace) -> None:
    if args.test_behavior not in args.target_classes:
        raise ValueError("--test_behavior must be one of --target_classes")
    if len(args.target_classes) != 2:
        raise ValueError("This alpha/beta runner requires exactly two target classes")

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

    alpha_intervals = interval_maps.get("alpha-iterative")
    if alpha_intervals is None:
        raise KeyError(f"No 'alpha-iterative' mapping found in {interval_map_path}")

    dataset = DataSet(
        subfolders=[args.dataset_subfolder],
        test_size=args.test_size,
    )
    model = SteerableModel(model_name=args.model_path)

    # Explicitly exclude parameterized steering.
    steering_types = get_steering_types_set(alpha=True, parameterized=False)
    alphas = np.arange(
        args.alpha_min,
        args.alpha_max + args.alpha_step / 2,
        args.alpha_step,
    )
    betas = np.arange(
        args.beta_min,
        args.beta_max + args.beta_step / 2,
        args.beta_step,
    )

    beta_behavior = next(
        behavior for behavior in args.target_classes if behavior != args.test_behavior
    )
    print(
        "Starting open-ended alpha-iterative evaluation: "
        f"test_behavior={args.test_behavior}, beta_behavior={beta_behavior}"
    )
    start_time = time.time()
    experiment = evaluate_multi_attribute(
        model,
        activations.steering_vectors,
        activations.pos_act_means,
        activations.neg_act_means,
        steering_types,
        dataset.test_data,
        intervention_layers=[args.layer],
        alpha_values=alphas,
        betas=betas,
        target_classes=args.target_classes,
        save_dir=None,
        logits_only=False,
        open_ended=True, 
        generation_batch_size=args.batch_size,
        logit_aggregation_method="entire_sequence",
        use_pickle=True,
        test_behaviors=[args.test_behavior],
        behavior_alpha_mapping=alpha_intervals,
        behavior_alpha_mapping_parameterized=None,
    )
    print(f"Generation completed in {time.time() - start_time:.2f} seconds")

    output_dir = os.path.join(PROJECT_ROOT, "results", "openended")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.save_name}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(experiment, f)
    print(f"Saved open-ended generations to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run alpha-iterative open-ended multi-attribute generations"
    )
    parser.add_argument("--model_path", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--activations_name", required=True)
    parser.add_argument("--dataset_subfolder", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--target_classes", nargs=2, required=True)
    parser.add_argument("--test_behavior", required=True)
    parser.add_argument("--interval_map_name", required=True)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.25)
    parser.add_argument("--beta-min", type=float, default=-2.0)
    parser.add_argument("--beta-max", type=float, default=2.0)
    parser.add_argument("--beta-step", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

import sys, os
import argparse

# SET TO RUN
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.steerable_model import SteerableModel
from src.activations import Activations
from src.dataset import DataSet
from src.utils import set_global_seed, outlier_pruning_stats, format_steering_vecs_for_eval
from src.corruption import corrupt_with_orthogonal_outliers, grid_search_steering_corruption, label_corruption
from src.corruption_utils import get_percent_steered_to_param_mapping, filter_mapping
from src.corrupted_activations import CorruptedActivations
from src.eval_utils import evaluate_across_behaviors_and_vecs
from src.experiment_output import ExperimentOutput
from src.corrupted_activations import CorruptedActivations
from src.corruption import corrupt_with_shared_random, get_acts_excluding_behavior
from estimators.steering_only_estimators import sample_diff_of_means
from src.eval_utils import evaluate_across_behaviors_and_vecs

import torch
import numpy as np
import time
import pickle
from huggingface_hub import login

set_global_seed(42)

# VARIABLES
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# layer = 12
# activations_name = "llama2_7b_chat_layer12"
# batch_size = 16
# behaviors = [
#             "uncorrigible-neutral-HHH",
#             "self-awareness-text-model",
#             "power-seeking-inclination",
#             "wealth-seeking-inclination",
#             "survival-instinct",
#             "coordinate-other-ais",
#         ]
# alpha_values = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
# save_name = f"baseline_steerability/{activations_name}"

def main(
    model_path: str,
    layer: int,
    activations_name: str, # ASSUMED TO NOT CONTAIN .pkl
    batch_size: int = 16,
    alphas: list[float] | None = None,
    save_name: str | None = None,
    test_size: int = 200,
):
    if alphas is None:
        alphas = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

    if save_name is None:
        save_name = f"baseline_steerability/{activations_name}.pkl"

    activations_base_path = f"{PROJECT_ROOT}/activations"
    activations_path = f"{activations_base_path}/{activations_name}.pkl"

    with open(activations_path, "rb") as f:
        activations_obj = pickle.load(f)

    # Assume sample steering vectors already present
    # Aligned with how activations are stored from get_activations.py
    steering_vecs = activations_obj.steering_vectors

    dataset = DataSet(subfolders=["tan_paper_datasets/mwe/xrisk"], test_size=test_size)

    model = SteerableModel(model_name=model_path)
    print("Successfully loaded model")

    print("Starting evaluation across behaviors and steering vecs")
    start_time = time.time()
    experiment = evaluate_across_behaviors_and_vecs(
        model,
        steering_vecs,
        dataset.test_data,
        intervention_layers = [layer], # just use layer calculated at
        alpha_values=alphas,
        save_dir = None,
        logits_only=True,
        generation_batch_size=batch_size,
        logit_aggregation_method="entire_sequence",
        use_pickle=True
    )
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time} seconds")

    save_path_base = f"{PROJECT_ROOT}/experiment_results"

    with open(f"{save_path_base}/{save_name}", "wb") as f:
        pickle.dump(experiment, f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate steerability as a function of alpha"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="HuggingFace model path or local checkpoint"
    )

    parser.add_argument(
        "--activations_name",
        type=str,
        help="Name of the activations file"
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to evaluate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="List of alpha values to sweep over (space-separated)"
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional output name"
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Test set size (default: 200)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        layer=args.layer,
        activations_name=args.activations_name,
        batch_size=args.batch_size,
        alphas=args.alphas,
        save_name=args.save_name,
        test_size=args.test_size
    )
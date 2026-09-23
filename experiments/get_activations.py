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
from src.utils import large_dataset_generator
import torch
import numpy as np
import time
import pickle
from huggingface_hub import login

set_global_seed(42)

def main(
    model_path: str,
    batch_size: int = 16,
    behaviors: list | None = None,
    layer: int = 12, # Zero Indexed
    save_name: str | None = None,
    test_size: int = 200,
    use_large_dataset: bool = False, # specific to some datasets for a particular experiment, not generael
):
    if not use_large_dataset:
        dataset = DataSet(
            subfolders=["tan_paper_datasets/mwe/xrisk"],
            test_size=test_size
        )
    else:
        # this grabs the large datasets and reformats them to be compatible with the same functionality
        # is specific to three behaviors I manually chose
        dataset = large_dataset_generator(test_size=test_size)

    if behaviors is None:
        if not use_large_dataset:
            behaviors = [
                "uncorrigible-neutral-HHH",
                "self-awareness-text-model",
                "power-seeking-inclination",
                "wealth-seeking-inclination",
                "survival-instinct",
                "coordinate-other-ais",
            ]
        else:
            # three large dataset behaviors
            behaviors = [
                "uncorrigible-neutral-HHH",
                "power-seeking-inclination",
                "wealth-seeking-inclination",
            ]

    if save_name is None:
        save_name = f"{model_path.split('/')[-1]}_layer{layer+1}"
        if behaviors is not None:
            behaviors_str = "_".join(behaviors)
            save_name += f"_{behaviors_str}"
            
    model = SteerableModel(model_name=model_path)
    print("Successfully loaded model")

    # Answers will be of form (A) and (B) -> -2 token gets actual answer token
    behavior_token_mapping = {behavior: -2 for behavior in behaviors}

    start_time = time.time()
    activations = model.get_binary_activations_on_dataset(
        dataset.train_data,
        layers=[layer],
        token_positions=["answer_token"],
        batch_size=batch_size,
        behaviors=behaviors,
        behavior_token_mapping=behavior_token_mapping,
    )
    end_time = time.time()
    #activations.save(name=f"{save_name}_activations_backup", activations_only=True)
    print(f"Successfully got activations in {end_time - start_time:.2f} seconds")

    estimators = {
        "sample_diff_of_means": sample_diff_of_means
    }

    # Steering vectors stored in activations object
    steering_vecs = activations.get_steering_vecs(estimators, behaviors)

    activations.save(name=save_name)

    print("Successfully saved activations and steering vectors")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save steering vectors from a steerable model."
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="HuggingFace model path or local checkpoint"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for activation extraction (default: 16)"
    )

    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=None,
        help="List of behaviors to evaluate (space-separated). Defaults to default 6 behaviors set."
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Layer index to extract activations from (default: 12)"
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional name for saved activations, defaults to model name and layer index"
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Test set size (default: 200)"
    )

    parser.add_argument(
        "--skip",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--use-large-dataset",
        action="store_true",
        help="Whether to use the large datasets specific to certain experiments"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.skip:
        print("Skipping execution as per --skip flag.")
    else:
        main(
            model_path=args.model_path,
            batch_size=args.batch_size,
            behaviors=args.behaviors,
            layer=args.layer,
            save_name=args.save_name,
            test_size=args.test_size,
            use_large_dataset=args.use_large_dataset,
        )

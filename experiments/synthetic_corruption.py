# Run Synthetic Corruption Experiments

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
from src.corruption import angle_outlier_corruption, corrupt_with_orthogonal_outliers, grid_search_steering_corruption, label_corruption
from src.corruption_utils import get_percent_steered_to_param_mapping, filter_mapping
from src.corrupted_activations import CorruptedActivations
from src.eval_utils import evaluate_across_behaviors_and_vecs
from src.experiment_output import ExperimentOutput
from src.corrupted_activations import CorruptedActivations
from src.corruption import corrupt_with_shared_random, get_acts_excluding_behavior
from estimators.steering_only_estimators import sample_diff_of_means
from src.eval_utils import evaluate_across_behaviors_and_vecs, grab_generations_across_behaviors_and_vecs, tinymmlu_eval

import torch
import numpy as np
import time
import pickle
from huggingface_hub import login
from datasets import load_dataset

from estimators.steering_only_estimators import sample_diff_of_means
from estimators.steering_estimator_wrappers import diff_of_means, mean_of_diffs
from estimators.que import que_mean
from estimators.simple_estimators import median_of_means
from estimators.lee_valiant import lee_valiant_simple
from estimators.lrv import lrv
from estimators.simple_estimators import coord_trimmed_mean

que_diff = lambda pos, neg, tau: diff_of_means(pos, neg, tau=tau, mean_fun=que_mean, return_outlier_indices=True)
que_match = lambda pos, neg, tau: mean_of_diffs(pos, neg, tau=tau, mean_fun=que_mean, mismatch=False, return_outlier_indices=True)
med_mean_diff = lambda pos, neg: diff_of_means(pos, neg, mean_fun=median_of_means)
med_mean_match = lambda pos, neg: mean_of_diffs(pos, neg, mean_fun=median_of_means, mismatch=False)
lee_valiant_diff = lambda pos, neg, tau: diff_of_means(pos, neg, tau=tau, mean_fun=lee_valiant_simple, return_outlier_indices=True)
lee_valiant_match = lambda pos, neg, tau: mean_of_diffs(pos, neg, tau=tau,mean_fun=lee_valiant_simple, mismatch=False, return_outlier_indices=True)
lrv_diff = lambda pos, neg: diff_of_means(pos, neg, mean_fun=lrv)
lrv_match = lambda pos, neg: mean_of_diffs(pos, neg, mean_fun=lrv, mismatch=False)
coord_prune_diff = lambda pos, neg, tau: diff_of_means(pos, neg, tau=tau, mean_fun=coord_trimmed_mean)
coord_prune_match = lambda pos, neg, tau: mean_of_diffs(pos, neg, tau=tau, mean_fun=coord_trimmed_mean)

estimator_mapping = {
    "que_diff": que_diff,
    "que_match": que_match,
    "med_mean_diff": med_mean_diff,
    "med_mean_match": med_mean_match,
    "lee_valiant_diff": lee_valiant_diff,
    "lee_valiant_match": lee_valiant_match,
    "lrv_diff": lrv_diff,
    "lrv_match": lrv_match,
    "coord_prune_diff": coord_prune_diff,
    "coord_prune_match": coord_prune_match,
}

set_global_seed(42)

llama_alpha_values = {
    "uncorrigible-neutral-HHH": 0.75,
    "myopic-reward": 0.75,
    #"self-awareness-text-model": 0.75,
    "power-seeking-inclination": 1,  
    "wealth-seeking-inclination": 1,
    "survival-instinct": 0.75,
    "coordinate-other-ais": 1
}

olmo_alpha_values = {
    "uncorrigible-neutral-HHH": 1,
    "myopic-reward": 0.75,
    #"self-awareness-text-model": 0.5,
    "power-seeking-inclination": 0.75,  
    "wealth-seeking-inclination": 1,
    "survival-instinct": 0.75,
    "coordinate-other-ais": 0.75
}

mistral_alpha_values = {
    "uncorrigible-neutral-HHH": 1,
    "myopic-reward": 0.5,
    #"self-awareness-text-model": 0.25,
    "power-seeking-inclination": 1,  
    "wealth-seeking-inclination": 1,
    "survival-instinct": 0.75,
    "coordinate-other-ais": 0.5
}


def main(
    model_path: str,
    activations_name: str, # ASSUMED TO NOT CONTAIN .pkl
    alpha: float, # possibly different value for different models
    layer: int, # possibly different value for different models, zero indexed
    batch_size: int = 16,
    estimator_names: list | None = None,
    thetas: list | None = None,
    behaviors: list | None = None,
    test_size: int = 200,
    grab_generations: bool = False,
    tiny_mmlu: bool = False,
    save_steering_vecs: bool = True,
    runs: int = 3,
    save_name_postfix: str = "", # in case we want to run multiple variants
):
    if grab_generations:
        dataset = DataSet(subfolders=["open_ended"], test_size=test_size, format_type="open_ended")
    elif tiny_mmlu:
        dataset = load_dataset("tinyBenchmarks/tinyMMLU")["test"]
    else:
        dataset = DataSet(subfolders=["tan_paper_datasets/mwe/xrisk"], test_size=test_size)
    
    if behaviors is None:
        behaviors = ["uncorrigible-neutral-HHH",
                    "myopic-reward",
                    #"self-awareness-text-model", 
                    "power-seeking-inclination", 
                    "wealth-seeking-inclination", 
                    "survival-instinct", 
                    "coordinate-other-ais"
                    ]
    if thetas is None:
        thetas = [0, 15, 30, 45, 60, 75]

    if estimator_names is None:
        estimators = {
            "sample_diff_of_means": sample_diff_of_means,
            # "lee_valiant_diff": lee_valiant_diff
        }
    else:
        estimators = {name: estimator_mapping[name] for name in estimator_names}

    activations_base_path = f"{PROJECT_ROOT}/activations"
    activations_path = f"{activations_base_path}/{activations_name}.pkl"

    with open(activations_path, "rb") as f:
        activations_obj = pickle.load(f)

    corrupted_activations = CorruptedActivations(activations_obj)

    model = SteerableModel(model_name=model_path)
    print("Successfully loaded model")

    seed_run_kwargs = {i: {"seed": i} for i in range(runs)}
    # Corrupt data over 3 runs for each theta (runs correspond to random seeds)
    # theta expected in degrees
    eta = 0.3 #  just choosing 0.3, doesn't really matter too much,
    for theta in thetas:
        corrupted_activations.apply_corruption(
            corruption_fun = angle_outlier_corruption,
            eta=eta,
            theta=theta, # kwarg for angle_outlier_corruption, determines exactly where to cluster outliers
            corruption_name=f"orthogonal_outliers_theta_{theta}",
            behaviors=behaviors, # kwargs do not vary across experiments here
            token_positions = ["answer_token"],
            multiple_runs_kwargs = seed_run_kwargs # uses different seeds for different runs
        )
    print("Completed corruption with orthogonal outliers")

    intervention_layers = [layer]
    alpha_values = [alpha]

    experiment = ExperimentOutput(generation_mode=grab_generations, benchmark_mode=tiny_mmlu)
    first_run = True

    steering_vecs_full = {}
    outliers_detected_data_full = {} 


    # Override alpha with behavior specific
    if "llama" in model_path.lower():
        alpha_mapping = llama_alpha_values
    elif "olmo" in model_path.lower():
        alpha_mapping = olmo_alpha_values
    elif "mistral" in model_path.lower():
        alpha_mapping = mistral_alpha_values
    else:
        alpha_mapping = None

    if grab_generations:
        printing_x = "generation"
    else:
        printing_x = "evaluation"

    print("Beginning evaluation across corrupted activations.")
    time_start_total = time.time()
    for corruption_name, corrupted_version in corrupted_activations.corrupted_versions.items():
        print(f"Evaluating corruption version: {corruption_name}")
        theta = corruption_name.split("_")[-1]
        steering_vecs_full[theta] = {}
        outliers_detected_data_full[theta] = {}

        time_start = time.time()

        for run in corrupted_version.keys():
            if run == "corruption_details":
                continue
            if first_run:
                include_no_steer = True
                first_run = False
            else:
                include_no_steer = False

            steering_vecs_full[theta][run] = {}
            outliers_detected_data_full[theta][run] = {}

            corrupted_activations = corrupted_version[run] # stores runs
            # steering_vecs = corrupted_activations.get_steering_vecs(estimators, behaviors)
            # UPDATED 1/9/26
            # Now has option to pass in eta as expected corruption
            # Includes inlier steering vec
            # Also returns outlier detected info for methods that do outlier detection (lee valiant)
            steering_vecs, outliers_detected_data = corrupted_activations.get_steering_vecs(estimators, 
                                                                                            behaviors, 
                                                                                            eta=float(eta), # 0.3 fixed here, not too important
                                                                                            include_sample_uncorrupted=True,
                                                                                            return_outlier_info=True,
                                                                                            )

            steering_vecs_full[theta][run] = steering_vecs
            outliers_detected_data_full[theta][run] = outliers_detected_data

            if grab_generations:
                grab_generations_across_behaviors_and_vecs(
                    model,
                    steering_vecs,
                    dataset.test_data,
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values, # overriden by behavior specific alpha values
                    generation_batch_size=batch_size,
                    include_no_steer = include_no_steer,
                    experiment_output = experiment, # accumulate results in this single object
                    # below two are used in formatting results table
                    # they are not passed as parameters into any evaluations
                    varying_variable = "theta", 
                    # CHANGE HOW THIS IS HANDLED
                    varying_variable_value = theta,
                    run = run,
                    behavior_alpha_mapping = alpha_mapping,
                )
            elif tiny_mmlu:
                tinymmlu_eval(
                    model,
                    steering_vecs,
                    dataset, # dataset here of different format
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values, # overriden by behavior specific alpha values
                    generation_batch_size=batch_size,
                    include_no_steer = include_no_steer,
                    experiment_output = experiment, # accumulate results in this single object
                    # below two are used in formatting results table
                    # they are not passed as parameters into any evaluations
                    varying_variable = "theta", 
                    # CHANGE HOW THIS IS HANDLED
                    varying_variable_value = theta,
                    run = run,
                    behavior_alpha_mapping = alpha_mapping,
                )
            else:
                evaluate_across_behaviors_and_vecs(
                    model,
                    steering_vecs,
                    dataset.test_data,
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values,
                    generation_batch_size=batch_size,
                    include_no_steer = include_no_steer,
                    experiment_output = experiment, # accumulate results in this single object
                    # below two are used in formatting results table
                    # they are not passed as parameters into any evaluations
                    varying_variable = "theta", 
                    # CHANGE HOW THIS IS HANDLED
                    varying_variable_value = theta,
                    run = run,
                    behavior_alpha_mapping = alpha_mapping,
                )
            time_end = time.time()
            print(f"Completed {printing_x} for {corruption_name}, Run {run}, in {time_end - time_start} seconds")
    time_end_total = time.time()
    print(f"Completed all {printing_x}s in {time_end_total - time_start_total} seconds")

    if grab_generations:
        save_path = f"{PROJECT_ROOT}/experiment_results/synthetic_corruption/{activations_name}_{save_name_postfix}_generations.pkl"
    elif tiny_mmlu:
        save_path = f"{PROJECT_ROOT}/experiment_results/synthetic_corruption/{activations_name}_{save_name_postfix}_tinymmlu_eval.pkl"
    else:
        save_path = f"{PROJECT_ROOT}/experiment_results/synthetic_corruption/{activations_name}_{save_name_postfix}.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(experiment, f)
    
    print("Saved experiment results")

    if save_steering_vecs:
        steering_vec_info = {
            "steering_vecs": steering_vecs_full,
            "outliers_detected_data": outliers_detected_data_full
        }

        with open(f"{PROJECT_ROOT}/experiment_results/synthetic_corruption/{activations_name}_{save_name_postfix}_steering_vecs_and_outlier_detected_data.pkl", "wb") as f:
            pickle.dump(steering_vec_info, f)

        print("Saved steering vector and outlier detected data info")

    if not grab_generations and not tiny_mmlu:
        try:
            experiment.plot_performance_grid(xlabel="Angle To Inlier Diff-In-Means (degrees)", 
                                            metric="percent_steered",
                                            save_title = f"{PROJECT_ROOT}/saved_plots/synthetic_corruption/{activations_name}_{save_name_postfix}_percent-steered"
                                            )
            experiment.plot_performance_grid(xlabel="Angle To Inlier Diff-In-Means (degrees)", 
                                            metric="avg_score",
                                            save_title = f"{PROJECT_ROOT}/saved_plots/synthetic_corruption/{activations_name}_{save_name_postfix}_avg-score"
                                            )
            print("Saved performance grid plot")
        except Exception as e:
            print(f"Could not save performance grid plot due to error: {e}")
    elif tiny_mmlu:
        try:
            experiment.plot_performance_grid(xlabel="Corruption Percentage", 
                                            metric="score",
                                            save_title = f"{PROJECT_ROOT}/saved_plots/synthetic_corruption/{activations_name}_mmlu-accuracy_{save_name_postfix}"
                                            )
            print("Saved MMLU accuracy grid plot")
        except Exception as e:
            print(f"Could not save MMLU accuracy grid plot due to error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate mislabel corruption steerability experiments"
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
        "--alpha",
        type=float,
        required=True,
        help="Alpha value for steering evaluation"
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
        "--estimator-names",
        type=str,
        nargs="+",
        default=None,
        help="List of estimator names to use (space-separated). Defaults to sample_diff_of_means and lee_valiant_diff."
    )

    parser.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=None,
        help="List of theta values in degrees for synthetic corruption (space-separated). Defaults to [0, 15, 30, 45, 60, 75]."
    )

    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=None,
        help="List of behaviors to evaluate (space-separated). Defaults to default 6 behaviors set."
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Test set size (default: 200)"
    )

    parser.add_argument(
        "--grab-generations",
        action="store_true",
        help="Flag to indicate whether to grab generations instead of evaluations"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs (default: 3)"
    )

    parser.add_argument(
        "--save-steering-vecs",
        action="store_true",
        help="Flag to indicate whether to save steering vectors and outlier detected data"
    )

    parser.add_argument(
        "--save-name-postfix",
        type=str,
        default="",
        help="Postfix to add to saved experiment result filenames"
    )

    parser.add_argument(
        "--tiny-mmlu",
        action="store_true",
        help="Flag to indicate whether to use tiny mmlu benchmarking instead of behavior evaluation"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        activations_name=args.activations_name,
        alpha=args.alpha,
        layer=args.layer,
        batch_size=args.batch_size,
        estimator_names=args.estimator_names,
        thetas=args.thetas,
        behaviors=args.behaviors,
        test_size=args.test_size,
        grab_generations=args.grab_generations,
        runs=args.runs,
        save_steering_vecs=args.save_steering_vecs,
        save_name_postfix=args.save_name_postfix,
        tiny_mmlu=args.tiny_mmlu
    )


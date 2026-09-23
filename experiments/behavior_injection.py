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
from src.corruption import corrupt_with_other_behavior
from src.corruption_utils import get_percent_steered_to_param_mapping, filter_mapping
from src.corrupted_activations import CorruptedActivations
from src.eval_utils import evaluate_across_behaviors_and_vecs
from src.experiment_output import ExperimentOutput
from src.corruption import corrupt_with_shared_random, get_acts_excluding_behavior
from estimators.steering_only_estimators import sample_diff_of_means
from src.eval_utils import evaluate_across_behaviors_and_vecs, grab_generations_across_behaviors_and_vecs, tinymmlu_eval

import torch
import numpy as np
import time
import pickle
from huggingface_hub import login
from datasets import load_dataset
from src.utils import large_dataset_generator

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
que_diff_force_prune = lambda pos, neg, tau: diff_of_means(pos, neg, tau = 0.5*tau, mean_fun=que_mean, return_outlier_indices=True, always_prune=True)

estimator_mapping = {
    "que_diff": que_diff,
    "que_match": que_match,
    "que_diff_force_prune": que_diff_force_prune,
    "med_mean_diff": med_mean_diff,
    "med_mean_match": med_mean_match,
    "lee_valiant_diff": lee_valiant_diff,
    "lee_valiant_match": lee_valiant_match,
    "lrv_diff": lrv_diff,
    "lrv_match": lrv_match,
    "coord_prune_diff": coord_prune_diff,
    "coord_prune_match": coord_prune_match,
    "sample_diff_of_means": sample_diff_of_means,
}

# Corrupt each behavior with the other 5
corruption_mapping = {
    "uncorrigible-neutral-HHH": ["myopic-reward", "power-seeking-inclination", "wealth-seeking-inclination", "survival-instinct", "coordinate-other-ais"],
    "myopic-reward": ["uncorrigible-neutral-HHH", "power-seeking-inclination", "wealth-seeking-inclination", "survival-instinct", "coordinate-other-ais"],
    #"self-awareness-text-model": ["uncorrigible-neutral-HHH", "power-seeking-inclination", "wealth-seeking-inclination", "survival-instinct", "coordinate-other-ais"],
    "power-seeking-inclination": ["uncorrigible-neutral-HHH", "myopic-reward", "wealth-seeking-inclination", "survival-instinct", "coordinate-other-ais"],
    "wealth-seeking-inclination": ["uncorrigible-neutral-HHH", "myopic-reward", "power-seeking-inclination", "survival-instinct", "coordinate-other-ais"],
    "survival-instinct": ["uncorrigible-neutral-HHH", "myopic-reward", "power-seeking-inclination", "wealth-seeking-inclination", "coordinate-other-ais"],
    "coordinate-other-ais": ["uncorrigible-neutral-HHH", "myopic-reward", "power-seeking-inclination", "wealth-seeking-inclination", "survival-instinct"],
}


set_global_seed(42)

with open("experiment_results/behavior_activations/Llama-3.2-3B-Instruct_layer13_behavior-cos-sim.pkl", "rb") as f:      
    llama3_sim_dict = pickle.load(f)

with open("experiment_results/behavior_activations/Mistral-7B-Instruct-v0.3_layer13_behavior-cos-sim.pkl", "rb") as f:      
    mistral_sim_dict = pickle.load(f)

with open("experiment_results/behavior_activations/OLMo-2-1124-7B-Instruct_layer13_behavior-cos-sim.pkl", "rb") as f:      
    olmo_sim_dict = pickle.load(f)

with open("experiment_results/behavior_activations/Llama-3.2-3B-Instruct_layer13_separability-stats.pkl", "rb") as f:      
    llama3_separability_stats = pickle.load(f)

with open("experiment_results/behavior_activations/Mistral-7B-Instruct-v0.3_layer13_separability-stats.pkl", "rb") as f:      
    mistral_separability_stats = pickle.load(f)

with open("experiment_results/behavior_activations/OLMo-2-1124-7B-Instruct_layer13_separability-stats.pkl", "rb") as f:     
    olmo_separability_stats = pickle.load(f)

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

# ADD ALL ESTIMATORS OPTION ! -> then put in robust mean failure + large datasets (or just put list)
def main(
    model_path: str,
    activations_name: str, # ASSUMED TO NOT CONTAIN .pkl
    alpha: float, # possibly different value for different models
    layer: int, # possibly different value for different models, zero indexed
    batch_size: int = 16,
    estimator_names: list | None = None,
    etas: list | None = None,
    behaviors: list | None = None,
    test_size: int = 200,
    save_name_postfix: str = "", # in case we want to run multiple variants
    grab_generations: bool = False,
    tiny_mmlu: bool = False,
    save_steering_vecs: bool = True,
    runs: int = 3,
    inlier_outlier_behavior_mapping = None, # If supplied, overrides corruption_mapping and evaluates only those specified
    use_large_dataset: bool = False, # specific to some datasets for a particular experiment, not generael  
    supply_steering_path: str | None = None, # added for precomputed steering vecs use -> needed for robust mean estimation suite, where steering vec calculation is the bottleneck
    grab_steering_vecs_only : bool = False, # if true, only grab steering vecs and save them, do not do evaluations -> needed for robust mean estimation suite

):
    if grab_generations:
        dataset = DataSet(subfolders=["open_ended"], test_size=test_size, format_type="open_ended")
    elif tiny_mmlu:
        dataset = load_dataset("tinyBenchmarks/tinyMMLU")["test"]
    elif use_large_dataset:
        # if doing this we need different activations
        # so specify the activations name!!
        dataset = large_dataset_generator(test_size=test_size)
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
    if etas is None:
        # including 0 to see effect of one steering vector on another behavior
        # not the most efficient but whatever
        etas = [0, 0.1, 0.2, 0.3, 0.4]

    if estimator_names is None:
        estimators = {
            "sample_diff_of_means": sample_diff_of_means,
            "lee_valiant_diff": lee_valiant_diff
        }
    else:
        estimators = {name: estimator_mapping[name] for name in estimator_names}

    activations_base_path = f"{PROJECT_ROOT}/activations"
    activations_path = f"{activations_base_path}/{activations_name}.pkl"

    with open(activations_path, "rb") as f:
        activations_obj = pickle.load(f)

    corrupted_activations = CorruptedActivations(activations_obj)

    if not grab_steering_vecs_only:
        model = SteerableModel(model_name=model_path)
        print("Successfully loaded model")
    else:
        model = None
        print("Skipping model load for grab steering vectors only run.")
    
    if supply_steering_path is not None:
        with open(f"{PROJECT_ROOT}/{supply_steering_path}", "rb") as f:
            supply_steering_vecs = pickle.load(f)
        try:
            supply_steering_vecs = supply_steering_vecs["steering_vecs"]
        except:
            pass
        print("Loaded precomputed steering vectors from supplied path.")
    else:
        supply_steering_vecs = None
        
    # Precomputed quantities for plotting
    if use_large_dataset:
        # we don't have or need this information
        cos_sim_dict = None
        separability_stats = None
        alpha_mapping = None
    if "llama" and "3.2" in model_path.lower():
        cos_sim_dict = llama3_sim_dict
        separability_stats = llama3_separability_stats
        alpha_mapping = llama_alpha_values
    elif "mistral" in model_path.lower():
        cos_sim_dict = mistral_sim_dict
        separability_stats = mistral_separability_stats
        alpha_mapping = mistral_alpha_values
    elif "olmo" in model_path.lower():
        cos_sim_dict = olmo_sim_dict
        separability_stats = olmo_separability_stats
        alpha_mapping = olmo_alpha_values
    else:
        cos_sim_dict = None
        separability_stats = None
        alpha_mapping = None

    if grab_generations:
        printing_x = "generation"
    else:
        printing_x = "evaluation"

    seed_run_kwargs = {i: {"seed": i} for i in range(runs)}
    # Corrupt data over 3 runs for each eta (runs correspond to random seeds)

    # CAN PROVIDE THIS FOR CUSTOM CORRUPTION MAPPING
    if inlier_outlier_behavior_mapping is not None:
        corruption_mapping = {}
        for inlier_behavior, outlier_behavior in inlier_outlier_behavior_mapping.items():
            if inlier_behavior in corruption_mapping:
                corruption_mapping[inlier_behavior].append(outlier_behavior)
            else:
                corruption_mapping[inlier_behavior] = [outlier_behavior]

    for behavior, corruption_behaviors in corruption_mapping.items():
        if behavior not in behaviors:
            # allow specifying subset of inlier behaviors to evaluate
            # useful as we may want to split this up across runs
            continue
        for corruption_behavior in corruption_behaviors:
            print(f"Corrupting behavior {behavior} with behavior {corruption_behavior}")
            corrupt_acts = activations_obj.data[corruption_behavior][layer]["answer_token"]
            corrupt_pos_acts = corrupt_acts["pos"]
            corrupt_neg_acts = corrupt_acts["neg"]
            for eta in etas:
                corrupted_activations.apply_corruption(
                        corruption_fun = corrupt_with_other_behavior,
                        eta=eta,
                        corruption_name=f"{behavior}_corruptedby_{corruption_behavior}_{eta}",
                        behaviors=[behavior],
                        other_pos_acts = corrupt_pos_acts,
                        other_neg_acts = corrupt_neg_acts,
                        token_positions = ["answer_token"],
                        multiple_runs_kwargs = seed_run_kwargs,
                        corruption_details_kwargs={
                            "inlier_behavior": behavior,
                            "outlier_behavior": corruption_behavior 
                        } # can be accessed as ["corruption_details"][kwarg]
                    )
    print("Completed corruption with behavior injection")
    print()

    intervention_layers = [layer]
    alpha_values = [alpha]

    experiment = ExperimentOutput(generation_mode=grab_generations, benchmark_mode=tiny_mmlu)
    first_run = True
    #include_no_steer = True # False # GRAB FROM MISLABEL

    steering_vecs_full = {}
    outliers_detected_data_full = {} 

    print("Beginning evaluation across corrupted activations.")
    time_start_total = time.time()
    for corruption_name, corrupted_version in corrupted_activations.corrupted_versions.items():
        print(f"Evaluating corruption version: {corruption_name}")
        inlier_behavior = corrupted_version["corruption_details"]["inlier_behavior"]
        outlier_behavior = corrupted_version["corruption_details"]["outlier_behavior"]
        test_behaviors = [inlier_behavior, outlier_behavior]
        eta = corruption_name.split("_")[-1]

        # doing this here to be safe, instead of using alpha mapping in the eval
        # that should still work, but weird stuff on how activations is passed here, so just to be safe
        if alpha_mapping is not None:
            alpha_values = [alpha_mapping.get(inlier_behavior, alpha)]

        # IMPORTANT TO USE CORRUPTION_NAME NOT ETA HERE
        # Unlike other corruption schemes
        steering_vecs_full[corruption_name] = {}
        outliers_detected_data_full[corruption_name] = {}

        time_start = time.time()
        # INCLUDE NO STEER DOING WEIRD STUFF!
        for run in corrupted_version.keys():
            if run == "corruption_details":
                continue
            if first_run:
                include_no_steer = True
                first_run = False
            else:
                include_no_steer = False

            steering_vecs_full[corruption_name][run] = {}
            outliers_detected_data_full[corruption_name][run] = {}

            # Before 1/23 I overrode corrupted_activations here
            # still worked though, I guess how python internally handles things
            local_corrupted_activations = corrupted_version[run] # stores runs
            # this is an activations object with only the corrupted behavior, hencee why the confusing get steering vecs still worked

            # steering_vecs = corrupted_activations.get_steering_vecs(estimators, behaviors)
            # UPDATED 1/9/26
            # Now has option to pass in eta as expected corruption
            # Includes inlier steering vec
            # Also returns outlier detected info for methods that do outlier detection (lee valiant)
            # behaviors here should just be inlier behaviors
            if supply_steering_vecs is None:
                steering_vecs, outliers_detected_data = local_corrupted_activations.get_steering_vecs(estimators, 
                                                                                                behaviors, 
                                                                                                eta=float(eta), 
                                                                                                include_sample_uncorrupted=True,
                                                                                                return_outlier_info=True,
                                                                                                )
            else:
                steering_vecs = supply_steering_vecs[corruption_name][run]
                outliers_detected_data = None

            steering_vecs_full[corruption_name][run] = steering_vecs
            outliers_detected_data_full[corruption_name][run] = outliers_detected_data
            
            # Unlike mislabel and random injection, we need to pass in test behaviors here
            if grab_generations:
                grab_generations_across_behaviors_and_vecs(
                    model,
                    steering_vecs,
                    dataset.test_data,
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values, # overriding for every inlier behavior
                    generation_batch_size=batch_size,
                    include_no_steer = include_no_steer,
                    experiment_output = experiment, # accumulate results in this single object
                    # below two are used in formatting results table
                    # they are not passed as parameters into any evaluations
                    varying_variable = "eta", 
                    # CHANGE HOW THIS IS HANDLED
                    varying_variable_value = eta,
                    run = run,
                    test_behaviors = test_behaviors,
                    additional_info_kwargs={"inlier_behavior": inlier_behavior, "outlier_behavior": outlier_behavior},
                    # behavior_alpha_mapping = alpha_mapping # this should work, but just manipulating alpha_values directly above to be safe
                )
            elif tiny_mmlu:
                tinymmlu_eval(
                    model,
                    steering_vecs,
                    dataset,
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values, # overriding for every inlier behavior
                    generation_batch_size=batch_size,
                    include_no_steer = include_no_steer,
                    experiment_output = experiment, # accumulate results in this single object
                    # below two are used in formatting results table
                    # they are not passed as parameters into any evaluations
                    varying_variable = "eta", 
                    # CHANGE HOW THIS IS HANDLED
                    varying_variable_value = eta,
                    run = run,
                    additional_info_kwargs={"inlier_behavior": inlier_behavior, "outlier_behavior": outlier_behavior},
                    # NO TEST_BEHAVIORS INPUT HERE
                    # behavior_alpha_mapping = alpha_mapping # this should work, but just manipulating alpha_values directly above to be safe
                )
            elif not grab_steering_vecs_only:
                evaluate_across_behaviors_and_vecs(
                    model,
                    steering_vecs,
                    dataset.test_data,
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values, # overriding for every inlier behavior
                    generation_batch_size=batch_size,
                    include_no_steer = include_no_steer,
                    experiment_output = experiment, # accumulate results in this single object
                    # below two are used in formatting results table
                    # they are not passed as parameters into any evaluations
                    varying_variable = "eta", 
                    # CHANGE HOW THIS IS HANDLED
                    varying_variable_value = eta,
                    run = run,
                    test_behaviors = test_behaviors,
                    additional_info_kwargs={"inlier_behavior": inlier_behavior, "outlier_behavior": outlier_behavior},
                    # behavior_alpha_mapping = alpha_mapping # this should work, but just manipulating alpha_values directly above to be safe
                )
            time_end = time.time()
            print(f"Completed {printing_x} for {corruption_name}, Run {run}, in {time_end - time_start} seconds")
    time_end_total = time.time()
    print(f"Completed all {printing_x}s in {time_end_total - time_start_total} seconds")

    if grab_generations:
        save_name = f"{PROJECT_ROOT}/experiment_results/behavior_injection/{activations_name}_{save_name_postfix}_generations.pkl"
    elif tiny_mmlu:
        save_name = f"{PROJECT_ROOT}/experiment_results/behavior_injection/{activations_name}_{save_name_postfix}_tinymmlu_eval.pkl"
    else:
        save_name = f"{PROJECT_ROOT}/experiment_results/behavior_injection/{activations_name}_{save_name_postfix}.pkl"
    
    if not grab_steering_vecs_only:
        with open(save_name, "wb") as f:
            pickle.dump(experiment, f)
    
        print("Saved experiment results")

    # if save_steering_vecs:
    steering_vec_info = {
        "steering_vecs": steering_vecs_full,
        "outliers_detected_data": outliers_detected_data_full
    }

    with open(f"{PROJECT_ROOT}/experiment_results/behavior_injection/{activations_name}_{save_name_postfix}_steering_vecs_and_outlier_detected_data.pkl", "wb") as f:
        pickle.dump(steering_vec_info, f)

    print("Saved steering vector and outlier detected data info")

    if not grab_generations and not tiny_mmlu and not grab_steering_vecs_only:
        try:
            os.makedirs(f"{PROJECT_ROOT}/saved_plots/behavior_injection/{activations_name}", exist_ok=True)
            for behavior in behaviors:
                experiment.plot_corruption_grid(
                    behavior=behavior,
                    xlabel="Corruption Percentage",
                    metric="avg_score",
                    save_title=f"{PROJECT_ROOT}/saved_plots/behavior_injection/{activations_name}/corr_{behavior}_avg-score.png",
                    cos_sim_dict=cos_sim_dict,
                    separability_stats=separability_stats
                )
                experiment.plot_corruption_grid(
                    behavior=behavior,
                    xlabel="Corruption Percentage",
                    metric="percent_steered",
                    save_title=f"{PROJECT_ROOT}/saved_plots/behavior_injection/{activations_name}/corr_{behavior}_percent-steered.png",
                    cos_sim_dict=cos_sim_dict,
                    separability_stats=separability_stats
                )
            print("Saved plots")
        except Exception as e:
            print(f"Error saving plots: {e}")
    elif tiny_mmlu:
        try:
            experiment.plot_performance_grid(xlabel="Corruption Percentage", 
                                            metric="score",
                                            save_title = f"{PROJECT_ROOT}/saved_plots/behavior_injection/{activations_name}_mmlu-accuracy_{save_name_postfix}"
                                            )
            print("Saved TinyMMLU accuracy plots")
        except Exception as e:
            print(f"Error saving TinyMMLU accuracy plots: {e}")


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
        "--etas",
        type=float,
        nargs="+",
        default=None,
        help="List of eta values for mislabel corruption (space-separated). Defaults to [0, 0.1, 0.2, 0.3, 0.4]."
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
        "--save-name-postfix",
        type=str,
        default="",
        help="Postfix to add to saved experiment result filenames"
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

    # indices of inlier and outlier behaviors for custom mapping
    # e.g. --inlier-behaviors behavior1 behavior2 --outlier-behaviors behavior3 behavior4
    # yields testing of behavior1 corrupted by behavior3 and behavior2 corrupted by behavior4
    parser.add_argument(
        "--inlier-behaviors",
        nargs="+",
        default=None,
        help="List of inlier behaviors for custom mapping (space-separated). Must be same length as outlier-behaviors."
    )

    parser.add_argument(
        "--outlier-behaviors",
        nargs="+",
        default=None,
        help="List of outlier behaviors for custom mapping (space-separated). Must be same length as inlier-behaviors."
    )

    parser.add_argument(
        "--tiny-mmlu",
        action="store_true",
        help="Flag to indicate whether to use tiny mmlu benchmarking instead of behavior evaluation"
    )

    parser.add_argument(
        "--use-large-dataset",
        action="store_true",
        help="Whether to use the large datasets specific to certain experiments"
    )
    
    parser.add_argument(
        "--grab-steering-vecs-only",
        action="store_true",
        help="Flag to indicate whether to grab steering vectors only, skipping model loading"
    )

    parser.add_argument(
        "--supply-steering-path",
        type=str,
        default=None,
        help="Path to precomputed steering vectors to use"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.inlier_behaviors is not None and args.outlier_behaviors is not None:
        if len(args.inlier_behaviors) != len(args.outlier_behaviors):
            raise ValueError("inlier-behaviors and outlier-behaviors must be the same length")
        inlier_outlier_behavior_mapping = {
            inlier: outlier for inlier, outlier in zip(args.inlier_behaviors, args.outlier_behaviors)
        }
    else:
        inlier_outlier_behavior_mapping = None

    main(
        model_path=args.model_path,
        activations_name=args.activations_name,
        alpha=args.alpha,
        layer=args.layer,
        batch_size=args.batch_size,
        estimator_names=args.estimator_names,
        etas=args.etas,
        behaviors=args.behaviors,
        test_size=args.test_size,
        save_name_postfix=args.save_name_postfix,
        grab_generations=args.grab_generations,
        runs=args.runs,
        inlier_outlier_behavior_mapping=inlier_outlier_behavior_mapping,
        tiny_mmlu=args.tiny_mmlu,
        use_large_dataset=args.use_large_dataset,
        grab_steering_vecs_only=args.grab_steering_vecs_only,
        supply_steering_path=args.supply_steering_path,
    )
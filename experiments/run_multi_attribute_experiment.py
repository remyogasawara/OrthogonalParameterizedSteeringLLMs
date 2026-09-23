"""
Stage 4/5: run the multi-attribute steering experiment (steer target_classes
together, sweeping alpha on the test behavior and beta on the other) and store
the resulting ExperimentOutput.

Example:
    python scripts/04_run_multi_attribute_experiment.py \
        --activations_name 7-16_sycophancy_1000_warmth_1000 \
        --dataset_subfolder agreeable_sycophancy \
        --layer 13 \
        --target_classes sycophancy_1000 warmth_1000 \
        --test_behaviors sycophancy_1000 \
        --interval_map_name 7-16_interval_map \
        --alpha alpha-iterative \
        --parameterized \
        --save_name 7-16_sycophancy_warmth_multi_attribute_experiment
"""

import sys, os, argparse, pickle, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.steerable_model import SteerableModel
from src.dataset import DataSet
from src.utils import set_global_seed
from src.multi_attribute_steering import evaluate_multi_attribute, get_steering_types_set

set_global_seed(42)


def main(
    model_path: str,
    activations_name: str,
    dataset_subfolder: str,
    layer: int,
    target_classes: list[str],
    test_behaviors: list[str],
    interval_map_name: str | None,
    use_alpha: bool,
    use_parameterized: bool,
    save_name: str,
    batch_size: int = 4,
    test_size: int = 200,
    alpha_min: float = -2.0,
    alpha_max: float = 2.0,
    alpha_step: float = 0.25,
    beta_min: float = -2.0,
    beta_max: float = 2.0,
    beta_step: float = 0.5,
):
    activations_path = f"{PROJECT_ROOT}/activations/{activations_name}.pkl"
    with open(activations_path, "rb") as f:
        activations_obj = pickle.load(f)

    steering_vecs = activations_obj.steering_vectors
    meanAs = activations_obj.pos_act_means
    meanBs = activations_obj.neg_act_means

    dataset = DataSet(subfolders=[dataset_subfolder], test_size=test_size)
    model = SteerableModel(model_name=model_path)
    print("Successfully loaded model")

    steering_types = get_steering_types_set(alpha=use_alpha, parameterized=use_parameterized)

    behavior_alpha_mapping = None
    behavior_alpha_mapping_parameterized = None
    if interval_map_name is not None:
        interval_map_path = f"{PROJECT_ROOT}/results/intervals/{interval_map_name}.pkl"
        with open(interval_map_path, "rb") as f:
            interval_maps = pickle.load(f)
        behavior_alpha_mapping = interval_maps.get("alpha-iterative")
        behavior_alpha_mapping_parameterized = interval_maps.get("parameterized")

    alphas = np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step)
    betas = np.arange(beta_min, beta_max + beta_step / 2, beta_step)

    print(f"Running multi-attribute experiment for target_classes={target_classes}, "
          f"test_behaviors={test_behaviors}, steering_types={steering_types}")
    start_time = time.time()


    # Does the alpha-beta experiment
    experiment = evaluate_multi_attribute(
        model,
        steering_vecs,
        meanAs,
        meanBs,
        steering_types,
        dataset.test_data,
        intervention_layers=[layer],
        target_classes=target_classes,
        test_behaviors=test_behaviors,
        alpha_values=alphas,
        betas=betas,
        save_dir=None,
        logits_only=True,
        generation_batch_size=batch_size,
        logit_aggregation_method="entire_sequence",
        use_pickle=True,
        behavior_alpha_mapping=behavior_alpha_mapping,
        behavior_alpha_mapping_parameterized=behavior_alpha_mapping_parameterized,
    )
    print(f"Evaluation completed in {time.time() - start_time:.2f}s")

    save_path_base = f"{PROJECT_ROOT}/results/logit_results/alpha_beta"
    os.makedirs(save_path_base, exist_ok=True)
    save_path = f"{save_path_base}/{save_name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(experiment, f)
    print(f"Saved multi-attribute experiment to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run and store the multi-attribute steering experiment")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--activations_name", type=str, required=True)
    parser.add_argument("--dataset_subfolder", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--target_classes", type=str, nargs="+", required=True,
                         help="All behaviors being jointly steered (e.g. sycophancy warmth)")
    parser.add_argument("--test_behaviors", type=str, nargs="+", required=True,
                         help="Subset of target_classes to sweep alpha over / evaluate on")
    parser.add_argument("--interval_map_name", type=str, default=None,
                         help="Output of 03_compute_intervals.py; omit to use raw alpha grid")
    parser.add_argument("--alpha", dest="use_alpha", action="store_true", help="include alpha-iterative steering types")
    parser.add_argument("--parameterized", dest="use_parameterized", action="store_true", help="include parameterized steering types")
    parser.add_argument("--save_name", type=str, required=True)
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
    args = parse_args()
    if not (args.use_alpha or args.use_parameterized):
        raise SystemExit("Pass at least one of --alpha / --parameterized")
    main(
        model_path=args.model_path,
        activations_name=args.activations_name,
        dataset_subfolder=args.dataset_subfolder,
        layer=args.layer,
        target_classes=args.target_classes,
        test_behaviors=args.test_behaviors,
        interval_map_name=args.interval_map_name,
        use_alpha=args.use_alpha,
        use_parameterized=args.use_parameterized,
        save_name=args.save_name,
        batch_size=args.batch_size,
        test_size=args.test_size,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
    )

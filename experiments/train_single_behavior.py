"""
Stage 2: sweep alpha independently for each behavior (one behavior steered at
a time) and store the resulting ExperimentOutput. This is the "training" run
that stage 3 fits per-behavior alpha intervals against.

Run once per steering_type you care about (alpha-iterative, parameterized, ...).

Example:
    python scripts/02_train_single_behavior.py \
        --activations_name 7-16_sycophancy_1000_warmth_1000 \
        --dataset_subfolder agreeable_sycophancy \
        --layer 13 \
        --steering_type alpha-iterative \
        --save_name 7-16_alpha-iterative_training_experiment
"""

import sys, os, argparse, pickle, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.steerable_model import SteerableModel
from src.dataset import DataSet
from src.utils import set_global_seed
from src.multi_attribute_steering import evaluate_alpha_sweep_single_behavior

set_global_seed(42)


def main(
    model_path: str,
    activations_name: str,
    dataset_subfolder: str,
    layer: int,
    steering_type: str,
    save_name: str | None = None,
    batch_size: int = 4,
    test_size: int = 200,
    alpha_min: float = -2.0,
    alpha_max: float = 2.0,
    alpha_step: float = 0.1,
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

    alphas = np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step)

    if save_name is None:
        save_name = f"{steering_type}_training_experiment_{activations_name}"

    print(f"Sweeping alpha for {steering_type} across {list(steering_vecs.keys())}")
    start_time = time.time()
    # experiment = evaluate_alpha_sweep_single_behavior(
    #     model,
    #     steering_vecs,
    #     meanAs,
    #     meanBs,
    #     steering_type,
    #     dataset.train_data,
    #     intervention_layers=[layer],
    #     alpha_values=alphas,
    #     save_dir=None,
    #     logits_only=True,
    #     generation_batch_size=batch_size,
    #     logit_aggregation_method="entire_sequence",
    #     use_pickle=True,
    # )

    experiment = evaluate_alpha_sweep_single_behavior(
        model=model,
        steering_vec_dict=steering_vecs,
        meanAs=meanAs,
        meanBs=meanBs,
        steering_type=steering_type,
        test_data_dict=dataset.train_data,
        intervention_layers=[layer],
        alpha_values=alphas,
        layer_subset=[layer],
        token_pos_subset=["answer_token"],
        logits_only=True,
        logit_aggregation_method="entire_sequence",
        generation_batch_size=batch_size,
    )
    print(f"Evaluation completed in {time.time() - start_time:.2f}s")

    save_path_base = f"{PROJECT_ROOT}/results/logit_results"
    os.makedirs(save_path_base, exist_ok=True)
    save_path = f"{save_path_base}/{save_name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(experiment, f)
    print(f"Saved training experiment to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-behavior alpha sweep (training data for interval fitting)")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--activations_name", type=str, required=True)
    parser.add_argument("--dataset_subfolder", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--steering_type", type=str, required=True,
                         choices=["alpha-iterative", "parameterized",
                                  "orthogonal-alpha-iterative", "orthogonal-parameterized"])
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        activations_name=args.activations_name,
        dataset_subfolder=args.dataset_subfolder,
        layer=args.layer,
        steering_type=args.steering_type,
        save_name=args.save_name,
        batch_size=args.batch_size,
        test_size=args.test_size,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
    )

"""
Stage 3: load one or more stage-2 training experiments, fit a per-behavior
alpha interval (`extract_intervals_padded`), and rescale the shared alpha grid
into each behavior's interval (`map_behaviors_to_interval`). Saves a dict of
{steering_type: {behavior: {grid_alpha: scaled_alpha}}} that stage 4 passes in
as `behavior_alpha_mapping` / `behavior_alpha_mapping_parameterized`.

Example:
    python scripts/03_compute_intervals.py \
        --training_experiments alpha-iterative:7-16_alpha-iterative_training_experiment \
                                parameterized:7-16_parameterized_training_experiment \
        --behaviors sycophancy_1000 warmth_1000 \
        --layer 13 \
        --gamma 0.1 \
        --save_name 7-16_interval_map
"""

import sys, os, argparse, pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.multi_attribute_steering import extract_intervals_padded, map_behaviors_to_interval

def main(
    training_experiments: dict[str, str],  # steering_type -> experiment save_name (no .pkl)
    behaviors: list[str],
    layer: int,
    save_name: str,
    gamma: float = 0.1,
    upper_bound: float = 2.0,
    lower_bound: float = -2.0,
    step: float = 0.1,
    token_pos: str = "answer_token",
    metric: str = "avg_score",
):
    alpha_grid = np.arange(lower_bound, upper_bound + step / 2, step)
    num_points = len(alpha_grid)

    interval_maps = {}
    for steering_type, exp_name in training_experiments.items():
        exp_path = f"{PROJECT_ROOT}/results/logit_results/{exp_name}.pkl"
        with open(exp_path, "rb") as f:
            experiment = pickle.load(f)
        df = experiment.to_dataframe()

        intervals_padded = extract_intervals_padded(
            df, alpha_grid, behaviors, metric=metric, layer=layer, token_pos=token_pos,
            gamma=gamma, upper_bound=upper_bound, lower_bound=lower_bound,
            step=step, num_points=num_points,
        )
        interval_maps[steering_type] = map_behaviors_to_interval(
            intervals_padded, alpha_grid, upper_bound=upper_bound, lower_bound=lower_bound,
        )
        print(f"[{steering_type}] fitted intervals: {intervals_padded}")

    save_path_base = f"{PROJECT_ROOT}/results/intervals"
    os.makedirs(save_path_base, exist_ok=True)
    save_path = f"{save_path_base}/{save_name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(interval_maps, f)
    print(f"Saved interval maps for {list(interval_maps.keys())} to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fit and store per-behavior alpha intervals")
    parser.add_argument(
        "--training_experiments", type=str, nargs="+", required=True,
        help="steering_type:experiment_save_name pairs, e.g. alpha-iterative:7-16_alpha-iterative_training_experiment",
    )
    parser.add_argument("--behaviors", type=str, nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--upper-bound", type=float, default=2.0)
    parser.add_argument("--lower-bound", type=float, default=-2.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--token-pos", type=str, default="answer_token")
    parser.add_argument("--metric", type=str, default="avg_score")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    training_experiments = dict(pair.split(":", 1) for pair in args.training_experiments)
    main(
        training_experiments=training_experiments,
        behaviors=args.behaviors,
        layer=args.layer,
        save_name=args.save_name,
        gamma=args.gamma,
        upper_bound=args.upper_bound,
        lower_bound=args.lower_bound,
        step=args.step,
        token_pos=args.token_pos,
        metric=args.metric,
    )

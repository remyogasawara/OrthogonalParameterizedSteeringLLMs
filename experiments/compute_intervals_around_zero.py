"""
Stage 3: load one or more stage-2 training experiments, fit a per-behavior
alpha interval (`extract_intervals`), pad it (`pad_behavior_intervals`), and
map a shared canonical grid into each behavior's padded raw interval. Saves a dict of
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

from src.multi_attribute_steering import (
    build_alpha_to_raw_mapping,
    build_canonical_grid,
    extract_intervals,
    pad_behavior_intervals,
)

def main(
    training_experiments: dict[str, str],  # steering_type -> experiment save_name (no .pkl)
    behaviors: list[str],
    layer: int,
    save_name: str,
    gamma: float = 0.1,
    upper_bound: float = 2.0,
    lower_bound: float = -2.0,
    step: float = 0.1,
    canonical_step: float = 0.25,
    token_pos: str = "answer_token",
    metric: str = "avg_score",
):
    if not np.isclose(lower_bound, -upper_bound):
        raise ValueError(
            "The around-zero mapping requires symmetric bounds: "
            "lower_bound must equal -upper_bound."
        )

    # `alpha_range` is the grid on which the training experiment was run.
    # `alpha_grid` is the canonical Stage-4 grid whose values are translated
    # to the fitted, padded raw range for each behavior.
    alpha_range = np.arange(lower_bound, upper_bound + step / 2, step)
    target_bound = upper_bound
    alpha_grid = build_canonical_grid(
        -target_bound,
        target_bound,
        canonical_step,
    )

    interval_maps = {}
    for steering_type, exp_name in training_experiments.items():
        exp_path = f"{PROJECT_ROOT}/results/logit_results/{exp_name}.pkl"
        with open(exp_path, "rb") as f:
            experiment = pickle.load(f)
        df = experiment.to_dataframe()

        intervals = extract_intervals(
            df,
            alpha_range,
            behaviors,
            metric=metric,
            layer=layer,
            token_pos=token_pos,
            gamma=gamma,
        )
        intervals_padded = pad_behavior_intervals(
            intervals,
            target_bound=target_bound,
        )
        interval_maps[steering_type] = build_alpha_to_raw_mapping(
            intervals_padded,
            alpha_grid,
            canonical_bound=target_bound,
        )
        print(f"[{steering_type}] fitted intervals: {intervals}")
        print(f"[{steering_type}] padded intervals: {intervals_padded}")

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
    parser.add_argument(
        "--canonical-step",
        type=float,
        default=0.25,
        help="Spacing of the shared canonical alpha grid saved for Stage 4.",
    )
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
        canonical_step=args.canonical_step,
        token_pos=args.token_pos,
        metric=args.metric,
    )

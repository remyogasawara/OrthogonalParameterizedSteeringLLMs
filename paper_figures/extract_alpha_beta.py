"""Rebuild data/alpha_beta_power_coord.csv (used by Figs 5-8) from a multi-attribute experiment pickle.

Usage:  python3 extract_alpha_beta.py path/to/<alpha-beta sweep>.pkl   (an ExperimentOutput with alpha-iterative and
        orthogonal alpha-iterative rows, e.g. the output of experiments/alpha-beta_experiment.py)
then:   python3 make_figs.py
Needs the repository root on sys.path (added below) for the ExperimentOutput class.
"""
import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def main(path, out_name="alpha_beta_power_coord.csv"):
    with open(path, "rb") as f:
        exp = pickle.load(f)
    df = exp.to_dataframe()
    keep = ["estimator", "alpha", "beta", "avg_score", "percent_steered", "num_steered", "num_total",
            "behavior", "beta_behavior", "eval_type", "layer"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["alpha_lookup"] = df["alpha"].astype(float).round(1)      # the alpha actually applied (as in the original plots)
    df["avg_score"] = df["avg_score"].astype(float)
    out = os.path.join(HERE, "data", out_name)
    old = pd.read_csv(out) if os.path.exists(out) else None
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows; estimators {sorted(df.estimator.unique())}; "
          f"beta {sorted(df.beta.unique())}")
    if old is not None:
        m = old.merge(df, on=["estimator", "alpha", "beta"], suffixes=("_old", "_new"))
        print(f"max |change| in avg_score vs previous data: {(m.avg_score_new - m.avg_score_old).abs().max():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])          # optional second argument: output file name in data/

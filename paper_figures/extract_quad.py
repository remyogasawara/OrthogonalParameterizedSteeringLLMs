"""Build the tidy CSV for a Fig 9 quad plot from two parameterized multi-attribute runs that test the SAME trait
(x = alpha trait = test trait, beta = the other trait) and differ only in which trait is steered first.

Usage: python3 extract_quad.py <out.csv> <x_trait> <beta_trait> <alpha_first.pkl> <beta_first.pkl>
(needs the repository root on sys.path, added below). Only the parameterized estimators are kept.
  coordination x corrigibility:  quad_coord_order.csv  coordinate-other-ais corrigible-neutral-HHH ...
  power x coordination:          quad_power_order.csv  power-seeking-inclination coordinate-other-ais ...
"""
import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def main(out, x_trait, beta_trait, alpha_first, beta_first):
    rows = []
    for order, path in (("alpha_first", alpha_first), ("beta_first", beta_first)):
        with open(path, "rb") as f:
            exp = pickle.load(f)
        df = exp.to_dataframe()
        assert set(df.behavior) == {x_trait} and set(df.beta_behavior) == {beta_trait}, path
        df = df[df.estimator.isin(["parameterized", "orthogonal-parameterized"])]
        df = df[["estimator", "alpha", "beta", "avg_score", "percent_steered"]].copy()
        df["order"] = order
        df["avg_score"] = df["avg_score"].astype(float)
        rows.append(df)
    out = os.path.join(HERE, "data", out)
    pd.concat(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:6])

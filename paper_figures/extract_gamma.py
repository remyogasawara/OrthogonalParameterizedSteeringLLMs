"""Rebuild data/gamma_curves.csv and data/gamma_intervals.csv (the gamma-sweep figure, make_figs.fig_gamma)
from the single-behaviour alpha-iterative TRAINING sweep pickle
(<model>_<behaviors>_alpha-iterative_training_experiment.pkl, an src.experiment_output.ExperimentOutput).

Usage:  python3 extract_gamma.py path/to/<...>_alpha-iterative_training_experiment.pkl
then:   python3 make_figs.py

gamma_curves.csv    : per behaviour and alpha (layer 13, answer_token), mean avg_score / percent_steered over the
                      sweep rows at that alpha (the aggregation extract_intervals_padded uses).
gamma_intervals.csv : for gamma = 0, 0.05, ..., 1: the selected steerable interval [A, B] from
                      src.multi_attribute_steering.extract_intervals_padded (the search used by
                      experiments/compute_intervals.py), the terms of its loss, the padded interval and the
                      alpha the interval map sends -1 and +1 to.
Needs the repository root on sys.path (added below).
"""
import os
import pickle
import sys
import types
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
warnings.filterwarnings("ignore")
try:                                   # plotly is imported by src.experiment_output but not used here
    import plotly  # noqa: F401
except ImportError:
    class _Stub(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            return type(n, (), {})
    for _m in ["plotly", "plotly.graph_objects", "plotly.express", "plotly.subplots"]:
        sys.modules[_m] = _Stub(_m)
from src.multi_attribute_steering import extract_intervals_padded, map_behaviors_to_interval  # noqa: E402

LAYER, TOK = 13, "answer_token"
ALPHA_GRID = np.arange(-2.0, 2.0 + 0.05, 0.1)          # 41 points, as in experiments/compute_intervals.py
GAMMAS = np.round(np.arange(0, 1.0 + 1e-9, 0.05), 2)


def loss_table(df, behavior, gamma):
    """Every candidate interval (A <= 0 <= B) with the loss extract_intervals_padded minimises, same expressions
    and the same grouped-Series dtype path, so the argmin can be checked against the library result."""
    sub = df[(df["behavior"] == behavior) & (df["layer"] == LAYER) & (df["token_pos"] == TOK)]
    s = sub.groupby(["estimator", "alpha"]).agg({"avg_score": ["mean"]}).reset_index().loc[:, ("avg_score", "mean")]
    mid = len(s) // 2
    gmin, gmax = s[:mid].min(), s[mid:].max()
    out = []
    for i, A in enumerate(ALPHA_GRID):
        if A > 1e-9:
            continue
        for j, B in enumerate(ALPHA_GRID):
            if B < -1e-9 or j <= i:
                continue
            t1 = ((s[i] - gmin) + (gmax - s[j])) / 2
            slope = (s[j] - s[i]) / (B - A)
            out.append((A, B, float(t1 + gamma * (1 - slope)), float(t1), float(slope), float(s[i]), float(s[j]),
                        float(gmin), float(gmax)))
    return out


def main(path):
    with open(path, "rb") as f:
        exp = pickle.load(f)
    df = exp.to_dataframe()
    behaviors = list(exp.behaviors)
    n = len(ALPHA_GRID)
    grid = np.round(ALPHA_GRID, 1)

    rows = []
    for b in behaviors:
        sub = df[(df.behavior == b) & (df.layer == LAYER) & (df.token_pos == TOK)]
        gr = sub.groupby(["estimator", "alpha"]).agg(avg_score=("avg_score", "mean"),
                                                    percent_steered=("percent_steered", "mean"),
                                                    n_rows=("avg_score", "size"),
                                                    num_total=("num_total", "first")).reset_index()
        assert len(gr) == n and np.allclose(gr.alpha, ALPHA_GRID, atol=1e-9), (b, len(gr))
        for _, r in gr.iterrows():
            rows.append(dict(behavior=b, alpha=round(float(r.alpha), 1), avg_score=round(float(r.avg_score), 6),
                             percent_steered=round(float(r.percent_steered), 6), n_rows=int(r.n_rows),
                             num_total=int(r.num_total)))
    curves = pd.DataFrame(rows)
    curves.to_csv(os.path.join(HERE, "data", "gamma_curves.csv"), index=False)

    common = dict(metric="avg_score", layer=LAYER, token_pos=TOK, step=0.1, num_points=n)
    irows = []
    for g in GAMMAS:
        raw = extract_intervals_padded(df, ALPHA_GRID, behaviors, gamma=g, upper_bound=0.0, lower_bound=0.0, **common)
        padded = extract_intervals_padded(df, ALPHA_GRID, behaviors, gamma=g, upper_bound=2.0, lower_bound=-2.0, **common)
        mapped = map_behaviors_to_interval(padded, ALPHA_GRID, upper_bound=2.0, lower_bound=-2.0)
        for b in behaviors:
            A, B = raw[b]
            tab = loss_table(df, b, g)
            best = min(tab, key=lambda r: r[2])
            assert abs(best[0] - A) < 1e-9 and abs(best[1] - B) < 1e-9, (g, b, best[:2], (A, B))
            r = [t for t in tab if abs(t[0] - A) < 1e-9 and abs(t[1] - B) < 1e-9][0]
            m = mapped[b]
            keys = sorted(m)
            km1, kp1 = min(keys, key=lambda k: abs(k + 1)), min(keys, key=lambda k: abs(k - 1))
            irows.append(dict(behavior=b, gamma=float(g), A=round(A, 1), B=round(B, 1), loss=round(r[2], 6),
                              endpoint_term=round(r[3], 6), slope=round(r[4], 6), f_A=round(r[5], 6),
                              f_B=round(r[6], 6), global_min=round(r[7], 6), global_max=round(r[8], 6),
                              padded_lo=round(padded[b][0], 6), padded_hi=round(padded[b][1], 6),
                              mapped_alpha_at_minus1=round(m[km1], 6), mapped_alpha_at_plus1=round(m[kp1], 6)))
    iv = pd.DataFrame(irows)
    iv.to_csv(os.path.join(HERE, "data", "gamma_intervals.csv"), index=False)
    print(f"wrote gamma_curves.csv {curves.shape} and gamma_intervals.csv {iv.shape}")


if __name__ == "__main__":
    main(sys.argv[1])

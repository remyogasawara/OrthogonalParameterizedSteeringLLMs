"""
File to store extra plotting functions
"""

from typing import List, Dict, Any, Optional, Union, Sequence, Optional, Literal
from src.interfaces import EvalOutput, EvalDict, SteeringResults, EvalType, LLMJudgeEvalOutput
from torch import Tensor
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import pickle
from src.utils import dicts_equal
from matplotlib.lines import Line2D

import matplotlib.cm as cm
import matplotlib.colors as mcolors

    """
    Plotting function to show the order dependency of parameterized steering
    Example Usage
    fig, axes = plot_order_dependency_quad(
        exp_sycophancy_first=sycophancy_first,
        exp_agreeableness_first=agreeableness_first,
        x_behavior="agreeableness",
        beta_behavior="sycophancy",
        orthogonal_estimators=["orthogonal-parameterized"],
        non_orthogonal_estimators=["parameterized"],
        metric="avg_score",
    )
    """

def plot_order_dependency_quad(
    exp_behavior1_first,
    exp_behavior2_first,
    behavior1,
    behavior2,
    x_behavior: str,
    beta_behavior: str,
    orthogonal_estimators: list,
    non_orthogonal_estimators: list,
    metric: str = "avg_score",
    figsize: tuple = (11, 9),
    alpha_range: Optional[tuple] = None,
    save_title: Optional[str] = None,
):
    """
    exp_behavior1_first:    already-loaded experiment_output where
        behavior 1 was steered first (left column).
    exp_behavior2_first: already-loaded experiment_output where
        behavior 2 was steered first (right column).
    behavior1 / behavior2: the two behaviors being compared (e.g. "sycophancy" and "agreeableness").
    x_behavior / beta_behavior: same as in plot_alpha_beta_curves — the
        trait swept on the x-axis (alpha) and the trait swept via color (beta).
        These stay fixed across all four panels; only order and estimator
        family change.
    orthogonal_estimators: e.g. ["orthogonal-parameterized", "orthogonal-alpha-iterative"]
        -> drawn in the top row.
    non_orthogonal_estimators: e.g. ["parameterized", "alpha-iterative"]
        -> drawn in the bottom row.
 
    Example:
        with open(f"{syco_first_path}.pkl", "rb") as f:
            syco_first = pickle.load(f)
        with open(f"{agree_first_path}.pkl", "rb") as f:
            agree_first = pickle.load(f)
 
        plot_order_dependency_quad(
            exp_behavior1_first=syco_first,
            exp_behavior2_first=agree_first,
            behavior1="sycophancy",
            behavior2="agreeableness",
            x_behavior="agreeableness",
            beta_behavior="sycophancy",
            orthogonal_estimators=["orthogonal-parameterized", "orthogonal-alpha-iterative"],
            non_orthogonal_estimators=["parameterized", "alpha-iterative"],
            metric="avg_score",
            save_title="figures/order_dependency_quad.png",
        )
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)
 
    col_experiments = [exp_behavior1_first, exp_behavior2_first]
    col_titles = [f"Steer {behavior1} first", f"Steer {behavior2} first"]
    row_estimators = [non_orthogonal_estimators, orthogonal_estimators]
    row_labels = ["non orthogonal", "orthogonal"]
 
    # First pass: shared beta range -> one colorbar for the whole figure.
    dfs = {}
    global_beta_min, global_beta_max = np.inf, -np.inf
    for c, exp_output in enumerate(col_experiments):
        df = exp_output.to_dataframe()
        if alpha_range is not None:
            df = df[df["alpha"].between(*alpha_range)]
        dfs[c] = df
        betas = df[df["behavior"] == x_behavior]["beta"].dropna()
        if not betas.empty:
            global_beta_min = min(global_beta_min, betas.min())
            global_beta_max = max(global_beta_max, betas.max())
 
    norm = mcolors.Normalize(vmin=global_beta_min, vmax=global_beta_max)
 
    all_betas, all_estimators = [], []
    for r, estimators in enumerate(row_estimators):
        for c, df in dfs.items():
            ax = axes[r][c]
            title = col_titles[c] if r == 0 else None
            betas_to_plot, est_list = _draw_alpha_beta_panel(
                ax, df, x_behavior, beta_behavior, estimators, metric, norm, title=title
            )
            all_betas.extend(betas_to_plot)
            all_estimators.extend(e for e in est_list if e not in all_estimators)
            if c == 0:
                ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
 
        # Row label placed to the right of the row, matching the sketch.
        axes[r][1].text(
            1.05, 0.5, row_labels[r], transform=axes[r][1].transAxes,
            rotation=270, va="center", ha="left", fontsize=12,
        )
 
    unique_betas = []
    for b in all_betas:
        if b is None or not any(
            (b is None and u is None) or (b is not None and u is not None and np.isclose(b, u))
            for u in unique_betas
        ):
            unique_betas.append(b)
 
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, shrink=0.9)
    cbar.set_label(f"β ({beta_behavior})")
    cbar.ax.axhline(y=0.0, color="black", linewidth=1.0)
 
    handles = _shared_legend_handles(all_estimators, unique_betas)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.45, -0.04),
        ncol=min(len(handles), 4),
        frameon=True,
    )
 
    if save_title:
        fig.savefig(f"{save_title}", bbox_inches="tight")
 
    return fig, axes
 

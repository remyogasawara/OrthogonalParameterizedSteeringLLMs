"""Build conditional alpha/beta sensitivity tables from experiment results.

The two primary quantities are:

1. beta sensitivity at fixed alpha:
       sd_beta(alpha) = SD_beta[metric(alpha, beta)]
   summarized by averaging sd_beta(alpha) over alpha values.

2. alpha sensitivity at fixed beta:
       sd_alpha(beta) = SD_alpha[metric(alpha, beta)]
   summarized by averaging sd_alpha(beta) over beta values.

The code first averages true replicates at each (alpha, beta) grid point, so
between-parameter sensitivity is not mixed with repeated-run variability.


Example usage:



python alpha_beta_variance.py ../results/logit_results/7-17_sycophancy_agreeableness_multi_attribute_experiment.pkl \
    --x-behavior sycophancy \
    --beta-behavior agreeableness \
    --metric percent_steered

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import pickle 
import argparse
from pathlib import Path




DEFAULT_CONTEXT_COLS = (
    "behavior",
    "beta_behavior",
    "layer",
    "token_pos",
    "eval_type",
    "estimator",
)


DEFAULT_ESTIMATORS = [
    "orthogonal-alpha-iterative",
    "alpha-iterative",
    "orthogonal-parameterized",
    "parameterized",
]



def _promote_additional_kwargs(df: pd.DataFrame) -> pd.DataFrame:
    """Promote beta fields if a DataFrame still stores them in a dict column."""
    out = df.copy()
    if "additional_kwargs" not in out.columns:
        return out

    for name in ("beta", "beta_behavior"):
        if name not in out.columns:
            out[name] = out["additional_kwargs"].map(
                lambda value: value.get(name) if isinstance(value, dict) else np.nan
            )
    return out


def build_alpha_beta_variance_tables(
    df: pd.DataFrame,
    *,
    metric: str = "percent_steered",
    x_behavior: Optional[str] = None,
    beta_behavior: Optional[str] = None,
    estimators: Optional[Iterable[str]] = None,
    filters: Optional[Mapping[str, object]] = None,
    context_cols: Iterable[str] = DEFAULT_CONTEXT_COLS,
    ddof: int = 0,
    min_points: int = 2,
) -> dict[str, pd.DataFrame]:
    """Return detailed and synthesized conditional-variance tables.

    Parameters
    ----------
    df:
        Output from ``experiment.to_dataframe()`` or an equivalent table.
    metric:
        Numeric result column, for example ``percent_steered`` or ``avg_score``.
    x_behavior, beta_behavior, estimators, filters:
        Optional restrictions applied before aggregation.
    context_cols:
        Columns that identify independent experiment slices. Only columns that
        exist in ``df`` are used. ``estimator`` should normally remain here.
    ddof:
        Standard-deviation convention. ``ddof=0`` treats the tested alpha/beta
        grid as the complete grid of interest. Use ``ddof=1`` only when treating
        those coefficient values as a sample from a larger population.
    min_points:
        Minimum number of distinct values needed to calculate a conditional SD.

    Returns
    -------
    A dictionary containing:
      - fixed_alpha_detail: one row per context/alpha, variation across beta
      - fixed_beta_detail: one row per context/beta, variation across alpha
      - table_1_fixed_alpha: estimator-level synthesis of beta sensitivity
      - table_2_fixed_beta: estimator-level synthesis of alpha sensitivity
      - fixed_alpha_sd_pivot: alpha x estimator SD table when pivotable
      - fixed_beta_sd_pivot: beta x estimator SD table when pivotable
      - grid_points: one row per unique alpha/beta point after replicate averaging
    """
    work = _promote_additional_kwargs(df)

    required = {"alpha", "beta", metric}
    missing = required.difference(work.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if x_behavior is not None:
        if "behavior" not in work.columns:
            raise KeyError("x_behavior was supplied but 'behavior' is absent")
        work = work.loc[work["behavior"] == x_behavior]

    if beta_behavior is not None:
        if "beta_behavior" not in work.columns:
            raise KeyError("beta_behavior was supplied but 'beta_behavior' is absent")
        work = work.loc[work["beta_behavior"] == beta_behavior]

    if estimators is not None:
        if "estimator" not in work.columns:
            raise KeyError("estimators were supplied but 'estimator' is absent")
        estimator_values = list(estimators)
        work = work.loc[work["estimator"].isin(estimator_values)]
    else:
        estimator_values = None

    for column, value in (filters or {}).items():
        if column not in work.columns:
            raise KeyError(f"Filter column {column!r} is absent")
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            work = work.loc[work[column].isin(value)]
        else:
            work = work.loc[work[column] == value]

    # beta=None rows are baseline/no-beta runs, not values in a numeric beta sweep.
    work = work.loc[
        work["alpha"].notna() & work["beta"].notna() & work[metric].notna()
    ].copy()
    work["alpha"] = pd.to_numeric(work["alpha"], errors="raise")
    work["beta"] = pd.to_numeric(work["beta"], errors="raise")
    work[metric] = pd.to_numeric(work[metric], errors="raise")

    if work.empty:
        raise ValueError("No rows remain after filtering")

    groups = [column for column in context_cols if column in work.columns]
    if "estimator" in work.columns and "estimator" not in groups:
        groups.append("estimator")

    # Average repeated seeds/runs at a grid point before measuring variation
    # across alpha or beta. This keeps two different variance concepts separate.
    grid_keys = groups + ["alpha", "beta"]
    grid_points = (
        work.groupby(grid_keys, dropna=False, observed=True)[metric]
        .agg(point_value="mean", n_replicates="size")
        .reset_index()
    )

    def pop_std(series: pd.Series) -> float:
        return float(series.std(ddof=ddof))

    def pop_var(series: pd.Series) -> float:
        return float(series.var(ddof=ddof))

    fixed_alpha = (
        grid_points.groupby(groups + ["alpha"], dropna=False, observed=True)
        .agg(
            n_beta=("beta", "nunique"),
            mean_over_beta=("point_value", "mean"),
            std_over_beta=("point_value", pop_std),
            variance_over_beta=("point_value", pop_var),
            min_over_beta=("point_value", "min"),
            max_over_beta=("point_value", "max"),
        )
        .reset_index()
    )
    fixed_alpha["range_over_beta"] = (
        fixed_alpha["max_over_beta"] - fixed_alpha["min_over_beta"]
    )
    fixed_alpha.loc[fixed_alpha["n_beta"] < min_points, [
        "std_over_beta", "variance_over_beta"
    ]] = np.nan

    fixed_beta = (
        grid_points.groupby(groups + ["beta"], dropna=False, observed=True)
        .agg(
            n_alpha=("alpha", "nunique"),
            mean_over_alpha=("point_value", "mean"),
            std_over_alpha=("point_value", pop_std),
            variance_over_alpha=("point_value", pop_var),
            min_over_alpha=("point_value", "min"),
            max_over_alpha=("point_value", "max"),
        )
        .reset_index()
    )
    fixed_beta["range_over_alpha"] = (
        fixed_beta["max_over_alpha"] - fixed_beta["min_over_alpha"]
    )
    fixed_beta.loc[fixed_beta["n_alpha"] < min_points, [
        "std_over_alpha", "variance_over_alpha"
    ]] = np.nan

    def synthesize(
        detail: pd.DataFrame,
        *,
        sd_col: str,
        variance_col: str,
        range_col: str,
        fixed_col: str,
        fixed_at_max_name: str,
    ) -> pd.DataFrame:
        summary = (
            detail.groupby(groups, dropna=False, observed=True)
            .agg(
                mean_conditional_sd=(sd_col, "mean"),
                median_conditional_sd=(sd_col, "median"),
                rms_conditional_sd=(variance_col, lambda s: float(np.sqrt(s.mean()))),
                max_conditional_sd=(sd_col, "max"),
                mean_conditional_range=(range_col, "mean"),
                n_fixed_values=(fixed_col, "nunique"),
            )
            .reset_index()
        )

        valid = detail.loc[detail[sd_col].notna()]
        if not valid.empty:
            max_indices = valid.groupby(groups, dropna=False, observed=True)[sd_col].idxmax()
            locations = valid.loc[max_indices, groups + [fixed_col]].rename(
                columns={fixed_col: fixed_at_max_name}
            )
            summary = summary.merge(locations, on=groups, how="left")
        else:
            summary[fixed_at_max_name] = np.nan

        return summary

    table_1 = synthesize(
        fixed_alpha,
        sd_col="std_over_beta",
        variance_col="variance_over_beta",
        range_col="range_over_beta",
        fixed_col="alpha",
        fixed_at_max_name="alpha_at_max_sd",
    )
    table_2 = synthesize(
        fixed_beta,
        sd_col="std_over_alpha",
        variance_col="variance_over_alpha",
        range_col="range_over_alpha",
        fixed_col="beta",
        fixed_at_max_name="beta_at_max_sd",
    )

    # Restore a requested estimator order rather than alphabetical groupby order.
    if estimator_values is not None and "estimator" in table_1.columns:
        order = pd.CategoricalDtype(estimator_values, ordered=True)
        for frame in (fixed_alpha, fixed_beta, table_1, table_2, grid_points):
            frame["estimator"] = frame["estimator"].astype(order)
            frame.sort_values(
                [c for c in groups if c != "estimator"] + ["estimator"],
                inplace=True,
                ignore_index=True,
            )

    # These pivots are convenient for appendix tables/heatmaps. If more than one
    # context remains, use the long-form detail tables instead.
    pivot_index_context = [c for c in groups if c != "estimator"]
    can_pivot_alpha = not pivot_index_context or all(
        fixed_alpha[c].nunique(dropna=False) == 1 for c in pivot_index_context
    )
    can_pivot_beta = not pivot_index_context or all(
        fixed_beta[c].nunique(dropna=False) == 1 for c in pivot_index_context
    )

    fixed_alpha_pivot = (
        fixed_alpha.pivot(index="alpha", columns="estimator", values="std_over_beta")
        if can_pivot_alpha and "estimator" in fixed_alpha.columns
        else pd.DataFrame()
    )
    fixed_beta_pivot = (
        fixed_beta.pivot(index="beta", columns="estimator", values="std_over_alpha")
        if can_pivot_beta and "estimator" in fixed_beta.columns
        else pd.DataFrame()
    )

    return {
        "fixed_alpha_detail": fixed_alpha,
        "fixed_beta_detail": fixed_beta,
        "table_1_fixed_alpha": table_1,
        "table_2_fixed_beta": table_2,
        "fixed_alpha_sd_pivot": fixed_alpha_pivot,
        "fixed_beta_sd_pivot": fixed_beta_pivot,
        "grid_points": grid_points,
    }


def percentage_point_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert SD/range columns from proportions to percentage points."""
    out = summary.copy()
    measure_cols = [
        "mean_conditional_sd",
        "median_conditional_sd",
        "rms_conditional_sd",
        "max_conditional_sd",
        "mean_conditional_range",
    ]
    for column in measure_cols:
        if column in out.columns:
            out[column] = 100.0 * out[column]
            out.rename(columns={column: f"{column}_pp"}, inplace=True)
    return out


def export_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Write all non-empty tables to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if not frame.empty:
            frame.to_csv(output_path / f"{name}.csv", index=True if "pivot" in name else False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build alpha/beta variance tables from an ExperimentOutput pickle."
        )
    )

    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to the trusted ExperimentOutput pickle.",
    )
    parser.add_argument(
        "--x-behavior",
        default="sycophancy",
        help="Behavior to use for the x/alpha dimension.",
    )
    parser.add_argument(
        "--beta-behavior",
        default="agreeableness",
        help="Behavior to use for the beta dimension.",
    )
    parser.add_argument(
        "--metric",
        default="percent_steered",
        help="Metric column used to build the tables.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=DEFAULT_ESTIMATORS,
        help="Estimator names to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path prefix passed to export_tables. "
            "By default, one is generated under results/tables."
        ),
    )

    return parser.parse_args()




def main(args: argparse.Namespace) -> None:
    if not args.experiment.is_file():
        raise FileNotFoundError(
            f"Experiment pickle does not exist: {args.experiment}"
        )

    # # Only load pickle files from trusted sources.
    with args.experiment.open("rb") as handle:
        experiment = pickle.load(handle)

    df = experiment.to_dataframe()

    tables = build_alpha_beta_variance_tables(
        df,
        x_behavior=args.x_behavior,
        beta_behavior=args.beta_behavior,
        metric=args.metric,
        estimators=args.estimators,
    )

    print(percentage_point_table(tables["table_1_fixed_alpha"]))
    print(percentage_point_table(tables["table_2_fixed_beta"]))

    output_path = args.output
    if output_path is None:
        output_path = Path(
            "results",
            "tables",
            (
                f"{args.x_behavior}_{args.beta_behavior}_"
                f"{args.metric}_variance_tables"
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_tables(tables, str(output_path))


if __name__ == "__main__":
    main(parse_args())

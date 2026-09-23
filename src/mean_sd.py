#!/usr/bin/env python3

"""
Aggregate mean conditional SDs from multi-attribute experiment pickle files.

For every ordered trait pair and estimator, this script computes two quantities:

1. Fix alpha, compute SD across beta at each alpha, then average those SDs.
2. Fix beta, compute SD across alpha at each beta, then average those SDs.

The default metric is ``avg_score``. Results are exported as CSV and LaTeX
with trait pairs as rows and estimators as columns.

Only unpickle experiment files that you trust.

Example Usage: 
python mean_sd.py ../results \
    --pattern "../results/logit_results/alpha_beta/8-3*-2026_*_multi_attribute_experiment.pkl" \
    --metric avg_score \
    --output-dir ../results/avg_score_mean_sd_tables
"""


from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ESTIMATORS = [
    "orthogonal-alpha-iterative",
    "alpha-iterative",
    "orthogonal-parameterized",
    "parameterized",
]

ESTIMATOR_DISPLAY_NAMES = {
    "orthogonal-alpha-iterative": "Orthogonal Alpha-Iterative",
    "alpha-iterative": "Alpha-Iterative",
    "orthogonal-parameterized": "Orthogonal Parameterized",
    "parameterized": "Parameterized",
}

# These columns should not silently vary within a single trait pair. If a file
# contains multiple values, filter to one configuration before making the table.
CONTEXT_COLUMNS = ("layer", "token_pos", "eval_type")


def promote_additional_kwargs(df: pd.DataFrame) -> pd.DataFrame:
    """Promote beta-related fields if they are stored in additional_kwargs."""
    out = df.copy()
    if "additional_kwargs" not in out.columns:
        return out

    for column in ("beta", "beta_behavior"):
        if column not in out.columns:
            out[column] = out["additional_kwargs"].map(
                lambda value: value.get(column)
                if isinstance(value, dict)
                else np.nan
            )
    return out


class _MissingProjectClass:
    """Fallback container for trusted pickles whose project class is unavailable."""


class _ProjectFallbackUnpickler(pickle.Unpickler):
    """Load data-only state when a local ``src.*`` class cannot be imported."""

    def find_class(self, module: str, name: str):  # type: ignore[override]
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module == "src" or module.startswith("src."):
                return _MissingProjectClass
            raise


def load_result_dataframe(path: Path, *, metric: str) -> pd.DataFrame:
    """Load a trusted pickle and retain only columns needed for aggregation."""
    with path.open("rb") as handle:
        experiment = _ProjectFallbackUnpickler(handle).load()

    needed = [
        "behavior",
        "beta_behavior",
        "estimator",
        "alpha",
        "beta",
        metric,
        "layer",
        "token_pos",
        "eval_type",
        "additional_kwargs",
    ]

    if isinstance(experiment, pd.DataFrame):
        df = experiment.copy()
    elif hasattr(experiment, "to_dataframe"):
        df = experiment.to_dataframe()
    elif hasattr(experiment, "outputs") and isinstance(experiment.outputs, list):
        # Building rows explicitly avoids carrying large arrays/raw generations
        # that are irrelevant to the conditional-SD calculation.
        rows = []
        for output in experiment.outputs:
            if not isinstance(output, dict):
                continue
            additional = output.get("additional_kwargs")
            additional = additional if isinstance(additional, dict) else {}
            rows.append(
                {
                    column: (
                        output.get(column)
                        if column in output
                        else additional.get(column)
                    )
                    for column in needed
                }
            )
        df = pd.DataFrame(rows)
    else:
        raise TypeError(
            f"{path.name}: expected a pandas DataFrame, an object with "
            "to_dataframe(), or an object containing an outputs list"
        )

    df = promote_additional_kwargs(df)
    keep = [column for column in needed if column in df.columns]
    df = df.loc[:, keep].copy()
    df["source_file"] = path.name
    return df


def apply_optional_filters(
    df: pd.DataFrame,
    *,
    layer: int | None,
    token_pos: str | None,
    eval_type: str | None,
) -> pd.DataFrame:
    """Apply optional context filters supplied on the command line."""
    out = df.copy()
    filters = {
        "layer": layer,
        "token_pos": token_pos,
        "eval_type": eval_type,
    }
    for column, value in filters.items():
        if value is None:
            continue
        if column not in out.columns:
            raise KeyError(
                f"Cannot filter on {column!r}; that column is absent from the data"
            )
        out = out.loc[out[column] == value]
    return out


def validate_context(df: pd.DataFrame) -> None:
    """Prevent accidental averaging across layers or evaluation settings."""
    pair_cols = ["behavior", "beta_behavior"]
    for column in CONTEXT_COLUMNS:
        if column not in df.columns:
            continue

        counts = (
            df.groupby(pair_cols, dropna=False, observed=True)[column]
            .nunique(dropna=False)
        )
        bad = counts[counts > 1]
        if not bad.empty:
            examples = ", ".join(
                f"({a}, {b})" for a, b in list(bad.index)[:5]
            )
            raise ValueError(
                f"Column {column!r} has multiple values for the same trait pair: "
                f"{examples}. Filter to one {column} value before aggregating."
            )


def conditional_sd(series: pd.Series, ddof: int) -> float:
    """Return SD when at least two coefficient values are available."""
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=ddof))


def build_mean_sd_tables(
    df: pd.DataFrame,
    *,
    metric: str = "avg_score",
    ddof: int = 0,
    estimator_order: Iterable[str] = DEFAULT_ESTIMATORS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build compact and detailed conditional-SD tables.

    Returns
    -------
    mean_sd_across_beta:
        Rows are ordered trait pairs; columns are estimators. Each cell is the
        mean over alpha of SD_beta(metric | alpha).
    mean_sd_across_alpha:
        Rows are ordered trait pairs; columns are estimators. Each cell is the
        mean over beta of SD_alpha(metric | beta).
    fixed_alpha_detail:
        One row per trait pair, estimator, and fixed alpha.
    fixed_beta_detail:
        One row per trait pair, estimator, and fixed beta.
    """
    required = {
        "behavior",
        "beta_behavior",
        "estimator",
        "alpha",
        "beta",
        metric,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.loc[
        df["alpha"].notna()
        & df["beta"].notna()
        & df[metric].notna()
    ].copy()

    if work.empty:
        raise ValueError("No numeric alpha/beta sweep rows remain after filtering")

    for column in ("alpha", "beta", metric):
        work[column] = pd.to_numeric(work[column], errors="raise")

    validate_context(work)

    pair_cols = ["behavior", "beta_behavior"]
    grid_cols = pair_cols + ["estimator", "alpha", "beta"]

    # Average genuine repeats/seeds at the same grid point first. If multiple
    # files contain the same trait pair and grid point, they are treated as
    # repeated measurements rather than extra coefficient values.
    grid = (
        work.groupby(grid_cols, dropna=False, observed=True)[metric]
        .agg(point_value="mean", n_replicates="size")
        .reset_index()
    )

    fixed_alpha_detail = (
        grid.groupby(
            pair_cols + ["estimator", "alpha"],
            dropna=False,
            observed=True,
        )
        .agg(
            n_beta=("beta", "nunique"),
            sd_across_beta=(
                "point_value",
                lambda values: conditional_sd(values, ddof),
            ),
        )
        .reset_index()
    )

    fixed_beta_detail = (
        grid.groupby(
            pair_cols + ["estimator", "beta"],
            dropna=False,
            observed=True,
        )
        .agg(
            n_alpha=("alpha", "nunique"),
            sd_across_alpha=(
                "point_value",
                lambda values: conditional_sd(values, ddof),
            ),
        )
        .reset_index()
    )

    beta_summary = (
        fixed_alpha_detail.groupby(
            pair_cols + ["estimator"],
            dropna=False,
            observed=True,
        )["sd_across_beta"]
        .mean()
        .rename("mean_sd_across_beta")
        .reset_index()
    )

    alpha_summary = (
        fixed_beta_detail.groupby(
            pair_cols + ["estimator"],
            dropna=False,
            observed=True,
        )["sd_across_alpha"]
        .mean()
        .rename("mean_sd_across_alpha")
        .reset_index()
    )

    # Preserve a stable estimator order and append unexpected estimators after it.
    requested_order = list(estimator_order)
    observed_estimators = list(pd.unique(work["estimator"].astype(str)))
    full_estimator_order = requested_order + [
        estimator
        for estimator in observed_estimators
        if estimator not in requested_order
    ]

    # Preserve trait-pair order according to the sorted input files/first appearance.
    pair_order = list(
        dict.fromkeys(
            zip(
                work["behavior"].astype(str),
                work["beta_behavior"].astype(str),
            )
        )
    )
    pair_labels = [f"{alpha_trait} / {beta_trait}" for alpha_trait, beta_trait in pair_order]

    def pivot_summary(
        summary: pd.DataFrame,
        value_column: str,
    ) -> pd.DataFrame:
        out = summary.copy()
        out["trait_pair"] = (
            out["behavior"].astype(str)
            + " / "
            + out["beta_behavior"].astype(str)
        )
        pivot = out.pivot(
            index="trait_pair",
            columns="estimator",
            values=value_column,
        )
        pivot = pivot.reindex(index=pair_labels)
        present_columns = [
            estimator
            for estimator in full_estimator_order
            if estimator in pivot.columns
        ]
        pivot = pivot.reindex(columns=present_columns)
        pivot.index.name = r"alpha trait / beta trait"
        return pivot

    mean_sd_across_beta = pivot_summary(beta_summary, "mean_sd_across_beta")
    mean_sd_across_alpha = pivot_summary(alpha_summary, "mean_sd_across_alpha")

    return (
        mean_sd_across_beta,
        mean_sd_across_alpha,
        fixed_alpha_detail,
        fixed_beta_detail,
    )


def latex_escape(value: object) -> str:
    """Escape plain text for use in LaTeX table cells."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def estimator_header(estimator: str) -> str:
    """Return a compact LaTeX header for an estimator name."""
    display = ESTIMATOR_DISPLAY_NAMES.get(estimator, estimator)
    if display == "Orthogonal Alpha-Iterative":
        return r"\shortstack{Orthogonal\\Alpha-Iterative}"
    if display == "Orthogonal Parameterized":
        return r"\shortstack{Orthogonal\\Parameterized}"
    return latex_escape(display)


def dataframe_to_latex_table(
    table: pd.DataFrame,
    *,
    caption: str,
    label: str,
    decimals: int,
) -> str:
    """Render a compact table using booktabs and resizebox."""
    column_spec = "l" + "r" * len(table.columns)
    header_cells = [r"$\alpha$ trait / $\beta$ trait"] + [
        estimator_header(str(column)) for column in table.columns
    ]

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \resizebox{\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{{column_spec}}}",
        r"        \toprule",
        "        " + " & ".join(header_cells) + r" \\",
        r"        \midrule",
    ]

    for row_name, row in table.iterrows():
        cells = [latex_escape(row_name)]
        for value in row:
            if pd.isna(value):
                cells.append(r"--")
            else:
                cells.append(f"{float(value):.{decimals}f}")
        lines.append("        " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create trait-pair x estimator tables containing only mean "
            "conditional SDs from multi-attribute experiment pickle files."
        )
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing the trusted experiment pickle files",
    )
    parser.add_argument(
        "--pattern",
        default="*_multi_attribute_experiment.pkl",
        help="Glob pattern inside results_dir (default: %(default)s)",
    )
    parser.add_argument(
        "--metric",
        default="avg_score",
        help="Metric column to aggregate (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pairwise_mean_sd_tables"),
        help="Directory for CSV and LaTeX outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        choices=(0, 1),
        help="0 for population SD; 1 for sample SD (default: %(default)s)",
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--token-pos", default=None)
    parser.add_argument("--eval-type", default=None)
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places in LaTeX and printed tables (default: %(default)s)",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()
    files = sorted(args.results_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} in {args.results_dir}"
        )

    frames = []
    for path in files:
        frame = load_result_dataframe(path, metric=args.metric)
        frame = apply_optional_filters(
            frame,
            layer=args.layer,
            token_pos=args.token_pos,
            eval_type=args.eval_type,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("All result files were empty after filtering")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    (
        mean_sd_beta,
        mean_sd_alpha,
        fixed_alpha_detail,
        fixed_beta_detail,
    ) = build_mean_sd_tables(
        combined,
        metric=args.metric,
        ddof=args.ddof,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    beta_csv = args.output_dir / f"{args.metric}_mean_sd_across_beta.csv"
    alpha_csv = args.output_dir / f"{args.metric}_mean_sd_across_alpha.csv"
    detail_alpha_csv = args.output_dir / f"{args.metric}_fixed_alpha_detail.csv"
    detail_beta_csv = args.output_dir / f"{args.metric}_fixed_beta_detail.csv"
    latex_path = args.output_dir / f"{args.metric}_mean_sd_tables.tex"

    mean_sd_beta.to_csv(beta_csv)
    mean_sd_alpha.to_csv(alpha_csv)
    fixed_alpha_detail.to_csv(detail_alpha_csv, index=False)
    fixed_beta_detail.to_csv(detail_beta_csv, index=False)

    metric_display = latex_escape(args.metric)
    ddof_description = "population" if args.ddof == 0 else "sample"

    beta_latex = dataframe_to_latex_table(
        mean_sd_beta,
        caption=(
            f"Mean {ddof_description} standard deviation of "
            f"\\texttt{{{metric_display}}} across $\\beta$ values while "
            "$\\alpha$ is fixed. Each row is an ordered trait pair: the first "
            "trait is controlled by $\\alpha$ and the second by $\\beta$."
        ),
        label=f"tab:{args.metric.replace('_', '-')}-mean-sd-beta",
        decimals=args.decimals,
    )
    alpha_latex = dataframe_to_latex_table(
        mean_sd_alpha,
        caption=(
            f"Mean {ddof_description} standard deviation of "
            f"\\texttt{{{metric_display}}} across $\\alpha$ values while "
            "$\\beta$ is fixed. Each row is an ordered trait pair: the first "
            "trait is controlled by $\\alpha$ and the second by $\\beta$."
        ),
        label=f"tab:{args.metric.replace('_', '-')}-mean-sd-alpha",
        decimals=args.decimals,
    )
    latex_path.write_text(beta_latex + "\n\n" + alpha_latex + "\n")

    print(f"Loaded {len(files)} file(s).")
    print("\nMean SD across beta with alpha fixed:\n")
    print(mean_sd_beta.round(args.decimals).to_string())
    print("\nMean SD across alpha with beta fixed:\n")
    print(mean_sd_alpha.round(args.decimals).to_string())
    print("\nWrote:")
    for path in (beta_csv, alpha_csv, detail_alpha_csv, detail_beta_csv, latex_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()

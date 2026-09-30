#!/usr/bin/env python3
"""
Compute endpoint-conditional steering ranges for multi-attribute experiments.

For each ordered trait pair and estimator, this script computes two quantities
for each direction:

1. Alpha sweep with beta fixed at -1 and +1:
       range_alpha(beta=b) = max_{alpha in [-1,1]} f(alpha,b)
                            - min_{alpha in [-1,1]} f(alpha,b)

2. Beta sweep with alpha fixed at -1 and +1:
       range_beta(alpha=a) = max_{beta in [-1,1]} f(a,beta)
                          - min_{beta in [-1,1]} f(a,beta)

Only the two estimators OA-I (orthogonal-alpha-iterative) and
A-I (alpha-iterative) are reported.

The script accepts the same positional pickle-file inputs and options as the
previous mean_range.py script.

Example:
python fixed_endpoint_range.py \\
    ../results/logit_results/alpha_beta/8-25-2026_*_multi_attribute_experiment.pkl \\
    ../results/logit_results/alpha_beta/8-26-2026_*_multi_attribute_experiment.pkl \\
    ../results/logit_results/alpha_beta/8-29-2026_*_multi_attribute_experiment.pkl \\
    ../results/logit_results/alpha_beta/8-3*-2026_*_multi_attribute_experiment.pkl \\
    --metric percent_steered \\
    --correlations correlations.json \\
    --output-dir ../results/percent_steered_fixed_endpoint_range_tables
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ESTIMATORS = [
    "orthogonal-alpha-iterative",
    "alpha-iterative",
]

ESTIMATOR_COLUMNS = {
    "orthogonal-alpha-iterative": "OA-I",
    "alpha-iterative": "A-I",
}

FIXED_VALUES = (-1.0, 1.0)
LOW = -1.0
HIGH = 1.0
TOL = 1e-8


def add_project_root_to_path() -> None:
    """Allow pickle classes such as src.experiment_output to be imported."""
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, script_path.parent.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "src").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute alpha/beta ranges at fixed coefficients -1 and +1 "
            "for OA-I and A-I."
        )
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more ExperimentOutput pickle files.",
    )

    parser.add_argument(
        "--metric",
        default="avg_score",
        choices=["avg_score", "percent_steered"],
        help="Metric over which to calculate ranges (default: avg_score).",
    )

    parser.add_argument(
        "--correlations",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping trait pairs to correlation values. "
            "Keys may be 'trait1 / trait2' or nested mappings."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixed_endpoint_range_tables"),
        help="Directory for CSV and LaTeX outputs.",
    )

    return parser.parse_args()


def load_experiment(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Pickle not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def get_dataframe(experiment: Any, path: Path, metric: str) -> pd.DataFrame:
    if not hasattr(experiment, "to_dataframe"):
        raise TypeError(f"{path} does not contain an ExperimentOutput-like object.")

    df = experiment.to_dataframe().copy()
    required = {"behavior", "beta_behavior", "estimator", "alpha", "beta", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.loc[df["beta"].notna()].copy()
    df = df.loc[df["estimator"].isin(ESTIMATORS)].copy()

    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["alpha", "beta", metric])

    # Only use the central coefficient region [-1, 1] in both dimensions.
    df = df[
        df["alpha"].between(LOW - TOL, HIGH + TOL)
        & df["beta"].between(LOW - TOL, HIGH + TOL)
    ].copy()

    if df.empty:
        raise ValueError(
            f"{path} has no usable rows with alpha,beta in [-1,1] "
            f"for metric {metric!r}."
        )

    return df


def collapse_duplicate_grid_points(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average repeated observations at the same coefficient grid point."""
    keys = [
        "behavior",
        "beta_behavior",
        "estimator",
        "alpha",
        "beta",
    ]
    return (
        df.groupby(keys, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value"})
    )


def fixed_value_mask(series: pd.Series, value: float) -> pd.Series:
    return np.isclose(series.to_numpy(dtype=float), value, atol=TOL, rtol=0.0)


def compute_ranges(
    df: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      alpha_table: alpha sweep ranges at beta=-1 and beta=+1
      beta_table: beta sweep ranges at alpha=-1 and alpha=+1
    """
    grid = collapse_duplicate_grid_points(df, metric)

    alpha_rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []

    grouped = grid.groupby(
        ["behavior", "beta_behavior", "estimator"], sort=False
    )

    for (behavior, beta_behavior, estimator), sub in grouped:
        # ------------------------------------------------------------
        # Sweep alpha, fix beta at -1 and +1.
        # ------------------------------------------------------------
        for fixed_beta in FIXED_VALUES:
            fixed = sub[fixed_value_mask(sub["beta"], fixed_beta)]

            # We want the full alpha sweep from -1 to +1.
            fixed = fixed[
                fixed["alpha"].between(LOW - TOL, HIGH + TOL)
            ]

            if fixed.empty:
                alpha_range = np.nan
                n_alpha = 0
            else:
                alpha_range = float(fixed["value"].max() - fixed["value"].min())
                n_alpha = int(fixed["alpha"].nunique())

            alpha_rows.append(
                {
                    "alpha_trait": behavior,
                    "beta_trait": beta_behavior,
                    "estimator": estimator,
                    "fixed_beta": fixed_beta,
                    "range": alpha_range,
                    "n_alpha": n_alpha,
                }
            )

        # ------------------------------------------------------------
        # Sweep beta, fix alpha at -1 and +1.
        # ------------------------------------------------------------
        for fixed_alpha in FIXED_VALUES:
            fixed = sub[fixed_value_mask(sub["alpha"], fixed_alpha)]

            # We want the full beta sweep from -1 to +1.
            fixed = fixed[
                fixed["beta"].between(LOW - TOL, HIGH + TOL)
            ]

            if fixed.empty:
                beta_range = np.nan
                n_beta = 0
            else:
                beta_range = float(fixed["value"].max() - fixed["value"].min())
                n_beta = int(fixed["beta"].nunique())

            beta_rows.append(
                {
                    "alpha_trait": behavior,
                    "beta_trait": beta_behavior,
                    "estimator": estimator,
                    "fixed_alpha": fixed_alpha,
                    "range": beta_range,
                    "n_beta": n_beta,
                }
            )

    return pd.DataFrame(alpha_rows), pd.DataFrame(beta_rows)


def load_correlations(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    result: dict[str, float] = {}

    for key, value in data.items():
        if isinstance(value, (int, float)):
            result[normalize_pair_key(key)] = float(value)
        elif isinstance(value, dict):
            for key2, value2 in value.items():
                if isinstance(value2, (int, float)):
                    result[normalize_pair_key(f"{key} / {key2}")] = float(value2)

    return result


def normalize_pair_key(key: str) -> str:
    parts = [p.strip() for p in key.split("/")]
    if len(parts) != 2:
        return key.strip()
    a, b = parts
    return f"{a} / {b}"


def correlation_for(
    alpha_trait: str,
    beta_trait: str,
    correlations: dict[str, float],
) -> float | None:
    direct = correlations.get(f"{alpha_trait} / {beta_trait}")
    if direct is not None:
        return direct
    return correlations.get(f"{beta_trait} / {alpha_trait}")


def make_wide(
    long_df: pd.DataFrame,
    correlations: dict[str, float],
    fixed_column: str,
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    pieces: list[pd.DataFrame] = []

    for fixed_value in FIXED_VALUES:
        sub = long_df[
            np.isclose(long_df[fixed_column].to_numpy(dtype=float), fixed_value, atol=TOL, rtol=0.0)
        ].copy()
        if sub.empty:
            continue

        label = "-1" if fixed_value < 0 else "+1"
        wide = sub.pivot_table(
            index=["alpha_trait", "beta_trait"],
            columns="estimator",
            values="range",
            aggfunc="mean",
        )
        wide.columns.name = None
        wide = wide.rename(columns=ESTIMATOR_COLUMNS)
        for col in ["OA-I", "A-I"]:
            if col not in wide.columns:
                wide[col] = np.nan
        wide = wide.rename(
            columns={
                "OA-I": f"OA-I @ {label}",
                "A-I": f"A-I @ {label}",
            }
        )
        pieces.append(wide)

    if not pieces:
        return pd.DataFrame()

    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.join(piece, how="outer")

    wide = wide.reset_index()
    wide["Corr."] = [
        correlation_for(a, b, correlations)
        for a, b in zip(wide["alpha_trait"], wide["beta_trait"])
    ]

    # Order rows by descending correlation, then alphabetically.
    wide["_corr_sort"] = wide["Corr."].fillna(-np.inf)
    wide = (
        wide.sort_values(
            ["_corr_sort", "alpha_trait", "beta_trait"],
            ascending=[False, True, True],
        )
        .drop(columns="_corr_sort")
    )

    # Differences are OA-I - A-I for each fixed setting.
    for label in ["-1", "+1"]:
        wide[f"Delta_AI @ {label}"] = (
            wide[f"OA-I @ {label}"] - wide[f"A-I @ {label}"]
        )

    if fixed_column == "fixed_beta":
        columns = [
            "alpha_trait", "beta_trait", "Corr.",
            "OA-I @ -1", "A-I @ -1", "Delta_AI @ -1",
            "OA-I @ +1", "A-I @ +1", "Delta_AI @ +1",
        ]
    else:
        columns = [
            "alpha_trait", "beta_trait", "Corr.",
            "OA-I @ -1", "A-I @ -1", "Delta_AI @ -1",
            "OA-I @ +1", "A-I @ +1", "Delta_AI @ +1",
        ]

    return wide[columns]


def tex_escape_trait(value: Any) -> str:
    text = str(value)
    text = text.replace("-", "-\\allowbreak ")
    return rf"\texttt{{{text}}}"


def latex_table(
    table: pd.DataFrame,
    metric: str,
    sweep: str,
    fixed_name: str,
    label: str,
) -> str:
    if sweep == "alpha":
        caption = (
            f"Range of \\texttt{{{metric}}} across $\\alpha$ from $-1$ to $+1$ "
            f"at fixed $\\beta={{{fixed_name}}}$. The range is the maximum "
            f"minus minimum value over the $\\alpha$ sweep."
        )
    else:
        caption = (
            f"Range of \\texttt{{{metric}}} across $\\beta$ from $-1$ to $+1$ "
            f"at fixed $\\alpha={{{fixed_name}}}$. The range is the maximum "
            f"minus minimum value over the $\\beta$ sweep."
        )

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \renewcommand{\arraystretch}{1.2}",
        r"    \caption{" + caption + "}",
        rf"    \label{{{label}}}",
        r"    \begin{tabularx}{\textwidth}{",
        r"        >{\raggedright\arraybackslash}X",
        r"        >{\raggedright\arraybackslash}X",
        r"        r",
        r"        rrr",
        r"        rrr",
        r"    }",
        r"        \toprule",
        r"        $\alpha$ Trait",
        r"        & $\beta$ Trait",
        r"        & Corr.",
        r"        & \multicolumn{3}{c}{$\text{Fixed } -1$}",
        r"        & \multicolumn{3}{c}{$\text{Fixed } +1$} \\",
        r"        \cmidrule(lr){4-6}",
        r"        \cmidrule(lr){7-9}",
        r"        & &",
        r"        & OA-I & A-I & $\Delta_{\mathrm{AI}}$",
        r"        & OA-I & A-I & $\Delta_{\mathrm{AI}}$ \\",
        r"        \midrule",
    ]

    for _, row in table.iterrows():
        corr = "--" if pd.isna(row["Corr."]) else f"{row['Corr.']:.3f}"

        def fmt(name: str) -> str:
            value = row[name]
            return "--" if pd.isna(value) else f"{value:.3f}"

        lines.append(
            f"        {tex_escape_trait(row['alpha_trait'])}"
        )
        lines.append(f"        & {tex_escape_trait(row['beta_trait'])}")
        lines.append(f"        & {corr}")
        lines.append(f"        & {fmt('OA-I @ -1')}")
        lines.append(f"        & {fmt('A-I @ -1')}")
        lines.append(f"        & {fmt('Delta_AI @ -1')}")
        lines.append(f"        & {fmt('OA-I @ +1')}")
        lines.append(f"        & {fmt('A-I @ +1')}")
        lines.append(f"        & {fmt('Delta_AI @ +1')} \\")

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabularx}",
        r"\end{table*}",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    add_project_root_to_path()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    correlations = load_correlations(args.correlations)

    alpha_frames: list[pd.DataFrame] = []
    beta_frames: list[pd.DataFrame] = []

    # De-duplicate file paths in case overlapping shell globs are used.
    unique_files = list(dict.fromkeys(str(path) for path in args.files))
    paths = [Path(path) for path in unique_files]

    print(f"Found {len(paths)} pickle file(s):")
    for path in paths:
        print(f"  {path}")

    for path in paths:
        experiment = load_experiment(path)
        df = get_dataframe(experiment, path, args.metric)
        alpha_summary, beta_summary = compute_ranges(df, args.metric)
        if not alpha_summary.empty:
            alpha_frames.append(alpha_summary)
        if not beta_summary.empty:
            beta_frames.append(beta_summary)

    alpha_long = (
        pd.concat(alpha_frames, ignore_index=True)
        if alpha_frames
        else pd.DataFrame()
    )
    beta_long = (
        pd.concat(beta_frames, ignore_index=True)
        if beta_frames
        else pd.DataFrame()
    )

    alpha_table = make_wide(alpha_long, correlations, "fixed_beta")
    beta_table = make_wide(beta_long, correlations, "fixed_alpha")

    numeric_cols = [
        "Corr.",
        "OA-I @ -1", "A-I @ -1", "Delta_AI @ -1",
        "OA-I @ +1", "A-I @ +1", "Delta_AI @ +1",
    ]
    for table in (alpha_table, beta_table):
        for col in numeric_cols:
            if col in table.columns:
                table[col] = table[col].round(3)

    alpha_csv = args.output_dir / f"range_across_alpha_fixed_beta_{args.metric}.csv"
    beta_csv = args.output_dir / f"range_across_beta_fixed_alpha_{args.metric}.csv"
    tex_file = args.output_dir / f"fixed_endpoint_range_tables_{args.metric}.tex"

    alpha_table.to_csv(alpha_csv, index=False)
    beta_table.to_csv(beta_csv, index=False)

    alpha_tex = latex_table(
        alpha_table,
        args.metric,
        "alpha",
        "\\beta",
        f"range-alpha-fixed-beta-{args.metric}",
    )
    beta_tex = latex_table(
        beta_table,
        args.metric,
        "beta",
        "\\alpha",
        f"range-beta-fixed-alpha-{args.metric}",
    )

    tex_file.write_text(
        "% Generated by fixed_endpoint_range.py\n\n"
        + alpha_tex
        + "\n"
        + beta_tex,
        encoding="utf-8",
    )

    print("\nRange across alpha with beta fixed at -1 and +1:")
    print(alpha_table.to_string(index=False))

    print("\nRange across beta with alpha fixed at -1 and +1:")
    print(beta_table.to_string(index=False))

    print("\nWrote:")
    print(f"  {alpha_csv}")
    print(f"  {beta_csv}")
    print(f"  {tex_file}")


if __name__ == "__main__":
    main()

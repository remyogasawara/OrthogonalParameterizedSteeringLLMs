#!/usr/bin/env python3
"""
Aggregate multi-attribute steering experiment pickle files into two tables:

1. Mean range across beta with alpha fixed
2. Mean range across alpha with beta fixed

For each ordered trait pair and estimator, the script computes:

    mean_range_beta = mean_alpha( max_beta f(alpha,beta) - min_beta f(alpha,beta) )
    mean_range_alpha = mean_beta( max_alpha f(alpha,beta) - min_alpha f(alpha,beta) )

where f is the requested metric (default: avg_score).

The script also computes:
    Delta_AI   = Alpha-Iterative - Orthogonal Alpha-Iterative
    Delta_P    = Parameterized - Orthogonal Parameterized

An optional JSON file can provide trait-pair correlations.

Example:
python mean_range.py \
    ../results/logit_results/alpha_beta/8-25-2026_*_multi_attribute_experiment.pkl \
    ../results/logit_results/alpha_beta/8-26-2026_*_multi_attribute_experiment.pkl \
    ../results/logit_results/alpha_beta/8-29-2026_*_multi_attribute_experiment.pkl \
    ../results/logit_results/alpha_beta/8-3*-2026_*_multi_attribute_experiment.pkl \
    --metric percent_steered \
    --correlations correlations.json \
    --output-dir ../results/range_tables/percent_steered

Run from the repository root, or from src/; the script adds the project root
onto sys.path so pickles containing src.experiment_output can be unpickled.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
import glob
from typing import Any

import numpy as np
import pandas as pd


ESTIMATORS = [
    "orthogonal-alpha-iterative",
    "alpha-iterative",
    "orthogonal-parameterized",
    "parameterized",
]

ESTIMATOR_COLUMNS = {
    "orthogonal-alpha-iterative": "OA-I",
    "alpha-iterative": "A-I",
    "orthogonal-parameterized": "OP",
    "parameterized": "P",
}


def add_project_root_to_path() -> None:
    """Allow pickle classes such as src.experiment_output to be imported."""
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, script_path.parent.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "src").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mean-range alpha/beta sensitivity tables from pickle files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="One or more ExperimentOutput pickle files.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern for input pickle files. Can be specified multiple times.",
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
        default=Path("mean_range_tables"),
        help="Directory for CSV and LaTeX outputs.",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        help="Unused; retained only to make clear that range does not use SD.",
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

    # Restrict both steering coefficients to [-1, 1].
    df = df[
        df["alpha"].between(-1.0, 1.0)
        & df["beta"].between(-1.0, 1.0)
    ].copy()


    if df.empty:
        raise ValueError(f"{path} has no usable beta-sweep rows for metric {metric!r}.")

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


def one_pair_table(
    df: pd.DataFrame,
    metric: str,
    source_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one-row summary tables for each ordered trait pair in a file."""
    grid = collapse_duplicate_grid_points(df, metric)

    rows_beta: list[dict[str, Any]] = []
    rows_alpha: list[dict[str, Any]] = []

    for (behavior, beta_behavior, estimator), sub in grid.groupby(
        ["behavior", "beta_behavior", "estimator"], sort=False
    ):
        by_alpha = (
            sub.groupby("alpha", as_index=False)["value"]
            .agg(beta_range=lambda x: float(x.max() - x.min()))
        )
        by_beta = (
            sub.groupby("beta", as_index=False)["value"]
            .agg(alpha_range=lambda x: float(x.max() - x.min()))
        )

        mean_range_beta = float(by_alpha["beta_range"].mean())
        mean_range_alpha = float(by_beta["alpha_range"].mean())

        common = {
            "alpha_trait": behavior,
            "beta_trait": beta_behavior,
            "estimator": estimator,
            "source": source_name,
            "n_alpha": int(sub["alpha"].nunique()),
            "n_beta": int(sub["beta"].nunique()),
        }

        rows_beta.append({**common, "mean_range": mean_range_beta})
        rows_alpha.append({**common, "mean_range": mean_range_alpha})

    return pd.DataFrame(rows_beta), pd.DataFrame(rows_alpha)


def load_correlations(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    result: dict[str, float] = {}

    # Supported format 1:
    # {"trait1 / trait2": 0.87, ...}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            result[normalize_pair_key(key)] = float(value)
        elif isinstance(value, dict):
            # Supported format 2:
            # {"trait1": {"trait2": 0.87}}
            for key2, value2 in value.items():
                if isinstance(value2, (int, float)):
                    result[normalize_pair_key(f"{key} / {key2}")] = float(value2)

    return result


def normalize_pair_key(key: str) -> str:
    parts = [p.strip() for p in key.split("/")]
    if len(parts) != 2:
        return key.strip()
    a, b = parts
    # Correlation is symmetric, so store both orientations.
    return f"{a} / {b}"


def correlation_for(
    alpha_trait: str,
    beta_trait: str,
    correlations: dict[str, float],
) -> float | None:
    direct = correlations.get(f"{alpha_trait} / {beta_trait}")
    if direct is not None:
        return direct
    reverse = correlations.get(f"{beta_trait} / {alpha_trait}")
    return reverse


def aggregate(
    result_frames: list[pd.DataFrame],
    correlations: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine all files and make wide estimator tables."""
    beta_long = pd.concat(
        [frame for frame in result_frames if not frame.empty],
        ignore_index=True,
    )

    alpha_long = beta_long.copy()

    # The caller provides separate metric summaries in the same row structure.
    # Reconstruct separately below.
    raise RuntimeError("Internal helper should not be called directly.")


def make_wide(summary: pd.DataFrame, correlations: dict[str, float]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    wide = summary.pivot_table(
        index=["alpha_trait", "beta_trait"],
        columns="estimator",
        values="mean_range",
        aggfunc="mean",
    )
    wide.columns.name = None
    wide = wide.rename(columns=ESTIMATOR_COLUMNS)

    # Ensure all expected estimator columns are present.
    for col in ["OA-I", "A-I", "OP", "P"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["Delta_AI"] = wide["A-I"] - wide["OA-I"]
    wide["Delta_P"] = wide["P"] - wide["OP"]

    wide = wide.reset_index()
    wide["Corr."] = [
        correlation_for(a, b, correlations)
        for a, b in zip(wide["alpha_trait"], wide["beta_trait"])
    ]

    # Correlation order: descending. Missing correlations go last.
    wide["_corr_sort"] = wide["Corr."].fillna(-np.inf)
    wide = wide.sort_values(
        ["_corr_sort", "alpha_trait", "beta_trait"],
        ascending=[False, True, True],
    ).drop(columns="_corr_sort")

    cols = [
        "alpha_trait",
        "beta_trait",
        "Corr.",
        "OA-I",
        "A-I",
        "Delta_AI",
        "OP",
        "P",
        "Delta_P",
    ]
    return wide[cols]


def tex_escape_trait(value: Any) -> str:
    text = str(value)
    # Use LaTeX tt and allow line breaks at hyphens without inserting a visible
    # space or character into the trait name.
    text = text.replace("-", "-\\allowbreak ")
    return rf"\texttt{{{text}}}"


def latex_table(
    table: pd.DataFrame,
    metric: str,
    direction: str,
    label: str,
) -> str:
    caption = (
        f"Mean conditional range of \\texttt{{{metric}}} across "
        f"${direction}$ while the other coefficient is held fixed. "
        "Higher values indicate a larger change in the evaluation metric "
        "over the coefficient sweep. Rows are sorted by descending trait "
        "correlation. OA-I denotes Orthogonal Alpha-Iterative, A-I denotes "
        "Alpha-Iterative, OP denotes Orthogonal Parameterized, and P denotes "
        "Parameterized. $\\Delta_{\\mathrm{AI}}$ is A-I minus OA-I, while "
        "$\\Delta_{\\mathrm{P}}$ is P minus OP."
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
        r"        rrrrrrr",
        r"    }",
        r"        \toprule",
        r"        $\alpha$ Trait",
        r"        & $\beta$ Trait",
        r"        & Corr.",
        r"        & OA-I",
        r"        & A-I",
        r"        & $\Delta_{\mathrm{AI}}$",
        r"        & OP",
        r"        & P",
        r"        & $\Delta_{\mathrm{P}}$ \\" ,
        r"        \midrule",
    ]

    for _, row in table.iterrows():
        corr = "--" if pd.isna(row["Corr."]) else f"{row['Corr.']:.3f}"

        def fmt(name: str) -> str:
            return "--" if pd.isna(row[name]) else f"{row[name]:.3f}"

        lines.extend(
            [
                f"        {tex_escape_trait(row['alpha_trait'])}",
                f"        & {tex_escape_trait(row['beta_trait'])}",
                f"        & {corr}",
                f"        & {fmt('OA-I')}",
                f"        & {fmt('A-I')}",
                f"        & {fmt('Delta_AI')}",
                f"        & {fmt('OP')}",
                f"        & {fmt('P')}",
                f"        & {fmt('Delta_P')} \\ ",
                r"",
            ]
        )

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    add_project_root_to_path()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matched_files = list(args.files)
    for pattern in args.pattern:
        matched_files.extend(Path(p) for p in glob.glob(pattern))

    # De-duplicate while preserving order.
    matched_files = list(dict.fromkeys(p.resolve() for p in matched_files))
    if not matched_files:
        raise SystemExit("No input pickle files were provided or matched the supplied patterns.")

    print(f"Found {len(matched_files)} pickle file(s):")
    for path in matched_files:
        print(f"  {path}")

    correlations = load_correlations(args.correlations)

    beta_frames: list[pd.DataFrame] = []
    alpha_frames: list[pd.DataFrame] = []

    for path in matched_files:
        experiment = load_experiment(path)
        df = get_dataframe(experiment, path, args.metric)
        beta_summary, alpha_summary = one_pair_table(
            df, args.metric, path.name
        )
        beta_frames.append(beta_summary)
        alpha_frames.append(alpha_summary)

    beta_long = pd.concat(beta_frames, ignore_index=True) if beta_frames else pd.DataFrame()
    alpha_long = pd.concat(alpha_frames, ignore_index=True) if alpha_frames else pd.DataFrame()

    beta_table = make_wide(beta_long, correlations)
    alpha_table = make_wide(alpha_long, correlations)

    # Round numerical values for output.
    for table in (beta_table, alpha_table):
        for col in ["Corr.", "OA-I", "A-I", "Delta_AI", "OP", "P", "Delta_P"]:
            if col in table.columns:
                table[col] = table[col].round(3)

    beta_csv = args.output_dir / f"mean_range_across_beta_{args.metric}.csv"
    alpha_csv = args.output_dir / f"mean_range_across_alpha_{args.metric}.csv"
    tex_file = args.output_dir / f"mean_range_tables_{args.metric}.tex"

    beta_table.to_csv(beta_csv, index=False)
    alpha_table.to_csv(alpha_csv, index=False)

    beta_tex = latex_table(
        beta_table,
        args.metric,
        r"\beta",
        f"mean-range-beta-{args.metric}",
    )
    alpha_tex = latex_table(
        alpha_table,
        args.metric,
        r"\alpha",
        f"mean-range-alpha-{args.metric}",
    )

    tex_file.write_text(
        "% Generated by mean_range_tables.py\n\n"
        + beta_tex
        + "\n"
        + alpha_tex,
        encoding="utf-8",
    )

    print("\nMean range across beta with alpha fixed:")
    print(beta_table.to_string(index=False))
    print("\nMean range across alpha with beta fixed:")
    print(alpha_table.to_string(index=False))
    print("\nWrote:")
    print(f"  {beta_csv}")
    print(f"  {alpha_csv}")
    print(f"  {tex_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a per-layer steering sweep with the experiment ``avg_score`` metric.

This reproduces the layer-sweep calculation used in the CAA paper, replacing
``p(answer matching behavior)`` with this project's average score:

    positive_effect(layer, behavior) = score(alpha=+a) - score(alpha=0)
    negative_effect(layer, behavior) = score(alpha=-a) - score(alpha=0)

Use the same alpha magnitude for every layer.  The script writes a paper-style
plot, machine-readable CSVs, per-behavior best layers, and a macro layer ranking.

The loader is intentionally restricted: it accepts the project's
``ExperimentOutput`` object plus NumPy arrays/scalars, rather than executing
arbitrary globals from a pickle file.

Example usage:

python src/layer_evaluation.py \
    --input results/logit_results \
    --pattern '8-14-2026_Qwen3-8B_alpha-iterative_training_experiment_layer*.pkl' \
    --baseline-alpha 0 \
    --positive-alpha 1 \
    --negative-alpha -1 \
    --score-source stored \
    --output-dir results/layer_sweep/Qwen_Qwen3-8B/avg_score_eval \
    --title 'Per-layer Average Score Effect: Qwen3-8B'

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


ALPHA_ROUND_DIGITS = 8
DUPLICATE_SCORE_TOLERANCE = 1e-10


class ExperimentOutputProxy:
    """Attribute container used when loading src.experiment_output.ExperimentOutput."""


# NumPy moved some private objects from numpy.core to numpy._core in NumPy 2.
_MULTIARRAY = getattr(getattr(np, "_core", None), "multiarray", None)
if _MULTIARRAY is None:
    _MULTIARRAY = np.core.multiarray  # type: ignore[attr-defined]

_ALLOWED_PICKLE_GLOBALS = {
    ("src.experiment_output", "ExperimentOutput"): ExperimentOutputProxy,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "scalar"): _MULTIARRAY.scalar,
    ("numpy._core.multiarray", "scalar"): _MULTIARRAY.scalar,
    ("numpy.core.multiarray", "_reconstruct"): _MULTIARRAY._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _MULTIARRAY._reconstruct,
}


class RestrictedUnpickler(pickle.Unpickler):
    """Load only the globals needed by these experiment result pickles."""

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key in _ALLOWED_PICKLE_GLOBALS:
            return _ALLOWED_PICKLE_GLOBALS[key]
        raise pickle.UnpicklingError(
            f"Blocked unsupported pickle global {module}.{name}. "
            "Confirm that this is an ExperimentOutput result file before "
            "adding another explicit allow-list entry."
        )


@dataclass(frozen=True)
class Condition:
    path: Path
    layer: int
    behavior: str
    alpha: float
    stored_avg_score: float
    details_avg_score: float | None
    details: np.ndarray | None
    example_ids: tuple[tuple[str, str, str], ...] | None

    def selected_score(self, source: str) -> float:
        if source == "stored":
            return self.stored_avg_score
        if source == "details":
            if self.details_avg_score is None:
                raise ValueError(
                    f"{self.path}: no usable eval_details for "
                    f"layer={self.layer}, behavior={self.behavior}, alpha={self.alpha}"
                )
            return self.details_avg_score
        raise AssertionError(f"Unexpected score source: {source}")


@dataclass(frozen=True)
class DeltaStats:
    paired_delta: float | None
    standard_error: float | None
    ci95_low: float | None
    ci95_high: float | None
    n_pairs: int


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def canonical_alpha(value: Any) -> float:
    result = round(float(value), ALPHA_ROUND_DIGITS)
    # Avoid a visually confusing -0.0 key.
    return 0.0 if abs(result) < 10 ** (-ALPHA_ROUND_DIGITS) else result


def load_result(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return RestrictedUnpickler(handle).load()
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise RuntimeError(f"Could not load {path}: {exc}") from exc


def get_outputs(obj: Any, path: Path) -> Sequence[dict[str, Any]]:
    if hasattr(obj, "outputs"):
        outputs = obj.outputs
    elif isinstance(obj, dict) and "outputs" in obj:
        outputs = obj["outputs"]
    elif isinstance(obj, list):
        outputs = obj
    else:
        raise TypeError(
            f"{path}: expected an object with .outputs, a dict containing "
            "'outputs', or a list of result dictionaries"
        )
    if not isinstance(outputs, list):
        raise TypeError(f"{path}: outputs is {type(outputs).__name__}, not list")
    return outputs


def extract_example_ids(record: dict[str, Any], details_len: int) -> tuple[tuple[str, str, str], ...] | None:
    raw_results = record.get("raw_results")
    if not isinstance(raw_results, list) or len(raw_results) != details_len:
        return None

    ids: list[tuple[str, str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            return None
        ids.append(
            (
                str(item.get("question", "")),
                str(item.get("pos_answer", "")),
                str(item.get("neg_answer", "")),
            )
        )
    return tuple(ids)


def extract_conditions(path: Path) -> list[Condition]:
    obj = load_result(path)
    outputs = get_outputs(obj, path)
    conditions: list[Condition] = []

    required = ("layer", "behavior", "alpha", "avg_score")
    for index, record in enumerate(outputs):
        if not isinstance(record, dict):
            raise TypeError(f"{path}: outputs[{index}] is not a dictionary")
        missing = [key for key in required if key not in record]
        if missing:
            raise KeyError(f"{path}: outputs[{index}] is missing {missing}")

        details: np.ndarray | None = None
        details_avg: float | None = None
        example_ids: tuple[tuple[str, str, str], ...] | None = None
        raw_details = record.get("eval_details")
        if raw_details is not None:
            try:
                candidate = np.asarray(raw_details, dtype=np.float64).reshape(-1)
                if candidate.size > 0 and np.all(np.isfinite(candidate)):
                    details = candidate.copy()
                    details_avg = float(np.mean(candidate, dtype=np.float64))
                    example_ids = extract_example_ids(record, candidate.size)
            except (TypeError, ValueError):
                pass

        stored_score = float(record["avg_score"])
        if not math.isfinite(stored_score):
            raise ValueError(f"{path}: non-finite avg_score in outputs[{index}]")

        conditions.append(
            Condition(
                path=path,
                layer=int(record["layer"]),
                behavior=str(record["behavior"]),
                alpha=canonical_alpha(record["alpha"]),
                stored_avg_score=stored_score,
                details_avg_score=details_avg,
                details=details,
                example_ids=example_ids,
            )
        )

    # Release the large steering vectors/raw logits as soon as extraction ends.
    del obj
    return conditions


def conditions_identical(left: Condition, right: Condition) -> bool:
    if not math.isclose(
        left.stored_avg_score,
        right.stored_avg_score,
        rel_tol=0.0,
        abs_tol=DUPLICATE_SCORE_TOLERANCE,
    ):
        return False
    if left.details is None and right.details is None:
        return True
    if left.details is None or right.details is None:
        return False
    return left.details.shape == right.details.shape and bool(
        np.array_equal(left.details, right.details)
    )


def collect_conditions(paths: Sequence[Path]) -> tuple[dict[tuple[int, str, float], Condition], list[str]]:
    grouped: dict[tuple[int, str, float], list[Condition]] = defaultdict(list)
    warnings: list[str] = []

    for path in paths:
        print(f"Loading {path}")
        for condition in extract_conditions(path):
            grouped[(condition.layer, condition.behavior, condition.alpha)].append(condition)

    resolved: dict[tuple[int, str, float], Condition] = {}
    for key, duplicates in grouped.items():
        first = duplicates[0]
        if len(duplicates) == 1:
            resolved[key] = first
            continue

        if all(conditions_identical(first, other) for other in duplicates[1:]):
            unique_locations = sorted({str(item.path) for item in duplicates})
            if len(unique_locations) == 1:
                location_text = f"within {unique_locations[0]}"
            else:
                location_text = "across: " + ", ".join(unique_locations)
            message = (
                f"Dropped {len(duplicates) - 1} exact duplicate record(s) for "
                f"layer={key[0]}, behavior={key[1]}, alpha={key[2]} {location_text}"
            )
            warnings.append(message)
            warn(message)
            resolved[key] = first
            continue

        locations = "\n  ".join(str(item.path) for item in duplicates)
        raise RuntimeError(
            "Conflicting duplicate conditions were found. This usually means "
            "the file pattern mixed multiple sweep runs. Narrow --pattern to one "
            f"run. Condition: layer={key[0]}, behavior={key[1]}, alpha={key[2]}\n  "
            f"{locations}"
        )

    return resolved, warnings


def paired_delta_stats(target: Condition, baseline: Condition) -> DeltaStats:
    if target.details is None or baseline.details is None:
        return DeltaStats(None, None, None, None, 0)
    if target.details.shape != baseline.details.shape:
        return DeltaStats(None, None, None, None, 0)
    if (
        target.example_ids is not None
        and baseline.example_ids is not None
        and target.example_ids != baseline.example_ids
    ):
        return DeltaStats(None, None, None, None, 0)

    differences = target.details - baseline.details
    n = int(differences.size)
    if n == 0:
        return DeltaStats(None, None, None, None, 0)

    mean = float(np.mean(differences, dtype=np.float64))
    if n == 1:
        return DeltaStats(mean, None, None, None, 1)

    standard_error = float(np.std(differences, ddof=1) / math.sqrt(n))
    half_width = 1.96 * standard_error
    return DeltaStats(
        paired_delta=mean,
        standard_error=standard_error,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
        n_pairs=n,
    )


def finite_or_blank(value: float | int | bool | None) -> Any:
    return "" if value is None else value


def analyze(
    conditions: dict[tuple[int, str, float], Condition],
    baseline_alpha: float,
    positive_alpha: float,
    negative_alpha: float,
    score_source: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    baseline_alpha = canonical_alpha(baseline_alpha)
    positive_alpha = canonical_alpha(positive_alpha)
    negative_alpha = canonical_alpha(negative_alpha)

    layer_behaviors = sorted({(layer, behavior) for layer, behavior, _ in conditions})
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for layer, behavior in layer_behaviors:
        keys = {
            "baseline": (layer, behavior, baseline_alpha),
            "positive": (layer, behavior, positive_alpha),
            "negative": (layer, behavior, negative_alpha),
        }
        missing = [label for label, key in keys.items() if key not in conditions]
        if missing:
            message = (
                f"Skipping layer={layer}, behavior={behavior}: missing "
                f"{', '.join(missing)} alpha condition(s)"
            )
            warnings.append(message)
            warn(message)
            continue

        baseline = conditions[keys["baseline"]]
        positive = conditions[keys["positive"]]
        negative = conditions[keys["negative"]]

        baseline_score = baseline.selected_score(score_source)
        positive_score = positive.selected_score(score_source)
        negative_score = negative.selected_score(score_source)
        positive_delta = positive_score - baseline_score
        negative_delta = negative_score - baseline_score

        pos_stats = paired_delta_stats(positive, baseline)
        neg_stats = paired_delta_stats(negative, baseline)

        rows.append(
            {
                "layer": layer,
                "behavior": behavior,
                "score_source": score_source,
                "baseline_alpha": baseline_alpha,
                "negative_alpha": negative_alpha,
                "positive_alpha": positive_alpha,
                "baseline_avg_score": baseline_score,
                "negative_avg_score": negative_score,
                "positive_avg_score": positive_score,
                "negative_delta_avg_score": negative_delta,
                "positive_delta_avg_score": positive_delta,
                # Positive when +alpha moves upward and -alpha moves downward.
                "directional_effect": (positive_delta - negative_delta) / 2.0,
                "positive_minus_negative": positive_score - negative_score,
                "mean_absolute_effect": (
                    abs(positive_delta) + abs(negative_delta)
                )
                / 2.0,
                "sign_consistent": positive_delta > 0.0 and negative_delta < 0.0,
                "positive_paired_delta": finite_or_blank(pos_stats.paired_delta),
                "positive_delta_se": finite_or_blank(pos_stats.standard_error),
                "positive_ci95_low": finite_or_blank(pos_stats.ci95_low),
                "positive_ci95_high": finite_or_blank(pos_stats.ci95_high),
                "positive_n_pairs": pos_stats.n_pairs,
                "negative_paired_delta": finite_or_blank(neg_stats.paired_delta),
                "negative_delta_se": finite_or_blank(neg_stats.standard_error),
                "negative_ci95_low": finite_or_blank(neg_stats.ci95_low),
                "negative_ci95_high": finite_or_blank(neg_stats.ci95_high),
                "negative_n_pairs": neg_stats.n_pairs,
                "source_file": str(baseline.path),
            }
        )

    if not rows:
        raise RuntimeError(
            "No complete layer/behavior conditions remained after matching the "
            "requested baseline and steering alphas."
        )
    return rows, warnings


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def macro_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    per_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_layer[int(row["layer"])].append(row)

    result: list[dict[str, Any]] = []
    for layer in sorted(per_layer):
        layer_rows = per_layer[layer]
        pos = np.asarray([row["positive_delta_avg_score"] for row in layer_rows], dtype=float)
        neg = np.asarray([row["negative_delta_avg_score"] for row in layer_rows], dtype=float)
        directional = np.asarray([row["directional_effect"] for row in layer_rows], dtype=float)
        absolute = np.asarray([row["mean_absolute_effect"] for row in layer_rows], dtype=float)
        result.append(
            {
                "layer": layer,
                "n_behaviors": len(layer_rows),
                "mean_positive_delta_avg_score": float(np.mean(pos)),
                "mean_negative_delta_avg_score": float(np.mean(neg)),
                "mean_directional_effect": float(np.mean(directional)),
                "median_directional_effect": float(np.median(directional)),
                "mean_absolute_effect": float(np.mean(absolute)),
                "sign_consistency_rate": float(
                    np.mean([bool(row["sign_consistent"]) for row in layer_rows])
                ),
            }
        )
    return result


def best_layer_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    per_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_behavior[str(row["behavior"])].append(row)

    result: list[dict[str, Any]] = []
    for behavior in sorted(per_behavior):
        best = max(
            per_behavior[behavior],
            key=lambda row: (
                float(row["directional_effect"]),
                float(row["mean_absolute_effect"]),
            ),
        )
        result.append(
            {
                "behavior": behavior,
                "best_layer": best["layer"],
                "directional_effect": best["directional_effect"],
                "positive_delta_avg_score": best["positive_delta_avg_score"],
                "negative_delta_avg_score": best["negative_delta_avg_score"],
                "mean_absolute_effect": best["mean_absolute_effect"],
                "sign_consistent": best["sign_consistent"],
            }
        )
    return result


def plot_paper_style(rows: Sequence[dict[str, Any]], path: Path, title: str) -> None:
    per_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_behavior[str(row["behavior"])].append(row)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for behavior in sorted(per_behavior):
        behavior_rows = sorted(per_behavior[behavior], key=lambda row: int(row["layer"]))
        layers = [int(row["layer"]) for row in behavior_rows]
        positive = [float(row["positive_delta_avg_score"]) for row in behavior_rows]
        negative = [float(row["negative_delta_avg_score"]) for row in behavior_rows]
        ax.plot(
            layers,
            positive,
            color="#377eb8",
            linewidth=1.8,
            alpha=0.78,
            marker="o",
            markersize=3.5,
        )
        ax.plot(
            layers,
            negative,
            color="#ff7f00",
            linewidth=1.8,
            alpha=0.78,
            marker="o",
            markersize=3.5,
        )

    ax.plot([], [], color="#377eb8", linewidth=2.2, label="Positive steering")
    ax.plot([], [], color="#ff7f00", linewidth=2.2, label="Negative steering")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    all_layers = sorted({int(row["layer"]) for row in rows})
    ax.set_xticks(all_layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\Delta$ avg_score (steered - baseline)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_macro(rows: Sequence[dict[str, Any]], path: Path, title: str) -> None:
    ordered = sorted(rows, key=lambda row: int(row["layer"]))
    layers = [int(row["layer"]) for row in ordered]
    positive = [float(row["mean_positive_delta_avg_score"]) for row in ordered]
    negative = [float(row["mean_negative_delta_avg_score"]) for row in ordered]
    directional = [float(row["mean_directional_effect"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(layers, positive, marker="o", linewidth=2.0, label="Mean positive delta")
    ax.plot(layers, negative, marker="o", linewidth=2.0, label="Mean negative delta")
    ax.plot(layers, directional, marker="o", linewidth=2.0, label="Directional effect")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Macro-average avg_score effect")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def find_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = sorted(path.resolve() for path in input_path.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern {pattern!r} below {input_path.resolve()}"
        )
    return files


def infer_title(paths: Sequence[Path]) -> str:
    name = paths[0].stem
    name = re.sub(r"_?layer\d+$", "", name)
    name = name.replace("_parameterized_training_experiment", "")
    name = name.replace("_", " ")
    return f"Per-layer Average Score Effect: {name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-layer ExperimentOutput pickles using avg_score."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/logit_results"),
        help="A result pickle or a directory containing result pickles.",
    )
    parser.add_argument(
        "--pattern",
        default="*parameterized_training_experiment_layer*.pkl",
        help=(
            "Recursive glob used when --input is a directory. Narrow this to one "
            "run so different sweeps are not mixed."
        ),
    )
    parser.add_argument("--baseline-alpha", type=float, default=0.0)
    parser.add_argument("--positive-alpha", type=float, default=1.0)
    parser.add_argument("--negative-alpha", type=float, default=-1.0)
    parser.add_argument(
        "--score-source",
        choices=("stored", "details"),
        default="stored",
        help=(
            "'stored' uses the pickle's avg_score field exactly; 'details' "
            "recomputes the same mean in float64 from eval_details."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/layer_sweep_eval"),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title. A title is inferred from the first filename if omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = find_files(args.input, args.pattern)
    print(f"Matched {len(files)} result file(s)")

    conditions, duplicate_warnings = collect_conditions(files)
    rows, analysis_warnings = analyze(
        conditions=conditions,
        baseline_alpha=args.baseline_alpha,
        positive_alpha=args.positive_alpha,
        negative_alpha=args.negative_alpha,
        score_source=args.score_source,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    macro = macro_rows(rows)
    best = best_layer_rows(rows)

    detail_csv = output_dir / "layer_sweep_avg_score_by_behavior.csv"
    macro_csv = output_dir / "layer_sweep_avg_score_macro.csv"
    best_csv = output_dir / "best_layers_by_behavior.csv"
    paper_plot = output_dir / "layer_sweep_avg_score.png"
    macro_plot = output_dir / "layer_sweep_avg_score_macro.png"

    write_csv(detail_csv, rows)
    write_csv(macro_csv, macro)
    write_csv(best_csv, best)

    title = args.title or infer_title(files)
    plot_paper_style(rows, paper_plot, title)
    plot_macro(macro, macro_plot, f"Macro layer ranking: {title}")

    metadata = {
        "input_files": [str(path) for path in files],
        "score_source": args.score_source,
        "baseline_alpha": canonical_alpha(args.baseline_alpha),
        "positive_alpha": canonical_alpha(args.positive_alpha),
        "negative_alpha": canonical_alpha(args.negative_alpha),
        "n_complete_layer_behavior_pairs": len(rows),
        "layers": sorted({int(row["layer"]) for row in rows}),
        "behaviors": sorted({str(row["behavior"]) for row in rows}),
        "warnings": duplicate_warnings + analysis_warnings,
    }
    with (output_dir / "layer_sweep_avg_score_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("\nWrote:")
    for path in (detail_csv, macro_csv, best_csv, paper_plot, macro_plot):
        print(f"  {path}")

    ranked = sorted(macro, key=lambda row: float(row["mean_directional_effect"]), reverse=True)
    print("\nLayer ranking by mean directional effect:")
    for rank, row in enumerate(ranked, start=1):
        print(
            f"  {rank:>2}. layer {int(row['layer']):>2}: "
            f"{float(row['mean_directional_effect']): .6f} "
            f"({int(row['n_behaviors'])} behaviors, "
            f"sign consistency {float(row['sign_consistency_rate']):.0%})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


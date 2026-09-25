"""Run the paper's validation experiments and produce its complete LaTeX table."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re

if __package__:
    from . import run_validation as numerical
    from .semantic import review
else:
    import run_validation as numerical
    from semantic import review

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "table_template.tex"


def percent(value, *, compact_zero=False):
    """Round percentages conventionally: 71.25% becomes 71.3%, not 71.2%."""
    number = Decimal(str(value))
    if not number.is_finite() or not 0 <= number <= 1:
        raise ValueError(f"Invalid proportion: {value}")
    if compact_zero and number == 0:
        return r"0\%"
    rounded = (100 * number).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded:.1f}" + r"\%"


def render_table(metrics, semantic):
    """Fill every result cell from measured outputs, never reference scores."""
    if (metrics["method"]["data_sha256"] != semantic["dataset_sha256"] or
            any(metrics["dataset"][key] != semantic["coverage"][key] for key in ("families", "responses"))):
        raise ValueError("Numerical and semantic results refer to different datasets")

    def pass_rate(key):
        result = semantic[key]
        return percent(Decimal(result["counts"]["pass"]) / result["denominator"])

    def accuracy_pair(method):
        return " / ".join(percent(metrics["classification"][method][scheme]["accuracy"])
                          for scheme in ("three_class", "five_class"))

    endorsement = []
    for label in numerical.LEVELS:
        result = semantic["belief_conflict_subset"]["by_warmth_level"][str(label)]
        # A missing denominator is undefined, never evidence of zero endorsement.
        endorsement.append(percent(Decimal(result["endorsements"]) / result["families"], compact_zero=True)
                           if result["families"] else "---")
    ordinal = metrics["ordinal"]["modernbert_full768"]
    values = {
        "semantic_all": pass_rate("all_six_reference_criteria"),
        "semantic_facts": pass_rate("all_required_facts"),
        "semantic_families": pass_rate("all_five_responses_satisfy_reference"),
        "endorsement": " / ".join(endorsement),
        "tfidf": accuracy_pair("tfidf"),
        "modernbert": accuracy_pair("modernbert_full768"),
        "word_count": accuracy_pair("word_count"),
        "surface_seven": accuracy_pair("surface_seven"),
        "ordinal_pairs": " / ".join(percent(ordinal[key]) for key in
                                    ("all_pair_accuracy", "adjacent_accuracy", "cross_polarity_accuracy")),
        "ordinal_families": percent(ordinal["strictly_ordered_family_rate"]),
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    tokens = re.findall(r"@@([a-z_]+)@@", template)
    if len(tokens) != len(values) or set(tokens) != set(values):
        raise ValueError("Table template contains missing, duplicate, or unknown result fields")
    return re.sub(r"@@([a-z_]+)@@", lambda match: values[match.group(1)], template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/warmth_families.jsonl",
                        help="Released dataset JSONL; its identity is checked against the saved review")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="New output directory")
    cache_options = parser.add_mutually_exclusive_group()
    cache_options.add_argument("--embeddings", type=Path, help="Override the bundled reference embedding NPZ")
    cache_options.add_argument("--fresh-embeddings", action="store_true", help="Extract new embeddings instead of using the bundled reference")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--judgments-dir", type=Path, help="Optional new complete judgments for the same dataset")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output")

    print("Validating and aggregating saved blinded semantic judgments...", flush=True)
    print("No judge model is invoked. Semantic rows use bundled historical judgments"
          " unless --judgments-dir supplies another review.", flush=True)
    semantic = review.aggregate(args.data, args.judgments_dir)
    # Validate cheap inputs before creating outputs or extracting embeddings.
    numerical.load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    semantic_output = args.output / "semantic"
    semantic_output.mkdir()
    numerical.save_json(semantic_output / "summary.json", semantic)
    (semantic_output / "REPORT.md").write_text(review.markdown(semantic), encoding="utf-8")

    print("Running classifiers, the ordinal probe, and length diagnostics...", flush=True)
    numerical_args = argparse.Namespace(**vars(args))
    numerical_args.output = args.output / "numerical"
    numerical_args.text_only = False
    numerical_args.include_pca = False
    metrics = numerical.run(numerical_args)
    table = render_table(metrics, semantic)
    destination = args.output / "warmth_validation_table.tex"
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(table)
    numerical.save_json(args.output / "table_provenance.json", {
        "data_sha256": metrics["method"]["data_sha256"],
        "runner_sha256": numerical.sha_file(Path(__file__)),
        "template_sha256": numerical.sha_file(TEMPLATE),
        "numerical_results_sha256": numerical.sha_file(numerical_args.output / "results.json"),
        "semantic_results_sha256": numerical.sha_file(semantic_output / "summary.json"),
        "table_sha256": numerical.sha_file(destination),
        "semantic_source": "aggregated saved model judgments; no fresh model review performed",
        "judgment_source": semantic["judgment_source"],
        "matches_bundled_judgment_files": semantic["matches_bundled_judgment_files"],
    })
    print(f"Complete: {destination}", flush=True)
    print(table)


if __name__ == "__main__":
    main()

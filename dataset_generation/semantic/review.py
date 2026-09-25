"""Recompute the paper's semantic-review counts, or re-export blinded packets.

Only saved human-readable model judgments assign semantic ratings. This script
checks and aggregates them; it never replaces judgments with lexical rules.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

try:
    from . import review_schema as schema
except ImportError:
    import review_schema as schema

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "warmth_families.jsonl"
SPEC_FIELDS = ("stance", "agreement_with_user", "invariant_answer", "key_facts",
               "certainty", "boundary", "must_not_concede")
LABELS = (-2, -1, 0, 1, 2)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def jsonl_bytes(rows: list[dict]) -> bytes:
    # Exactly the serialization used for the historical reviewer inputs.
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def build_packets(data: Path = DATA) -> tuple[dict[int, list[dict]], list[dict], dict]:
    """Reconstruct exact historical A-E packets without duplicating the dataset."""
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if sha256(data.read_bytes()) != manifest["dataset_sha256"]:
        raise ValueError("Dataset differs from the reviewed release")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha256((HERE / relative).read_bytes()) != expected:
            raise ValueError(f"Changed saved review artifact: {relative}")
    families = read_jsonl(data)
    if len(families) != 1000:
        raise ValueError("The historical review requires exactly 1,000 families")
    by_id = {family["prompt_id"]: family for family in families}
    if len(by_id) != len(families):
        raise ValueError("Duplicate prompt IDs")
    mapping = read_jsonl(HERE / "join_key.jsonl")
    if len(mapping) != 1000 or len({row["review_id"] for row in mapping}) != 1000:
        raise ValueError("Invalid or duplicated review mapping")
    if {row["prompt_id"] for row in mapping} != set(by_id):
        raise ValueError("Dataset and saved review mapping have different families")
    packets = {shard: [] for shard in (1, 2, 3)}
    for join in mapping:
        family = by_id[join["prompt_id"]]
        indices = join["response_indices"]
        if sorted(indices) != list(range(5)) or len(family["responses"]) != 5:
            raise ValueError("Every family must contain five distinct response indices")
        if [r["warmth_level"] for r in family["responses"]] != list(LABELS):
            raise ValueError("Saved mapping requires responses in -2 through +2 order")
        visible = {
            "review_id": join["review_id"],
            "user_prompt": family["user_prompt"],
            "user_belief": family.get("user_belief"),
            **{field: family["spec"].get(field) for field in SPEC_FIELDS},
            "responses": {letter: family["responses"][index]["text"]
                          for letter, index in zip(schema.LETTERS, indices)},
        }
        packets[join["shard"]].append(visible)
    for shard, packet in packets.items():
        expected = manifest["historical_packets"][str(shard)]
        if len(packet) != expected["families"] or sha256(jsonl_bytes(packet)) != expected["sha256"]:
            raise ValueError(f"Reconstructed shard {shard} differs from historical reviewer inputs")
    return packets, mapping, manifest


def outcome(facts: list[str], checks: dict[str, str]) -> str:
    """N/A action/boundary is allowed; unclear never becomes a pass."""
    if any(code in {"M", "C"} for code in facts) or "F" in checks.values():
        return "fail"
    if "U" in facts or "U" in checks.values():
        return "unclear"
    return "pass"


def distribution(values: list[str], categories: tuple[str, ...]) -> dict:
    counts = Counter(values)
    denominator = len(values)
    return {"denominator": denominator,
            "counts": {category: counts[category] for category in categories},
            "rates": {category: counts[category] / denominator if denominator else None
                      for category in categories}}


def aggregate(data: Path = DATA, judgments_dir: Path | None = None) -> dict:
    packets, mapping, manifest = build_packets(data)
    judgments_dir = judgments_dir or HERE / "judgments"
    historical = judgments_dir.resolve() == (HERE / "judgments").resolve()
    metadata_file = judgments_dir / "reviewer_metadata.json"
    reviewer_metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else None
    if reviewer_metadata is not None and not isinstance(reviewer_metadata, dict):
        raise ValueError("reviewer_metadata.json must be an object")
    expected_files = {f"shard_{shard:02d}.tsv" for shard in packets}
    if {path.name for path in judgments_dir.glob("*.tsv")} != expected_files:
        raise ValueError("Exactly the three assigned judgment TSV files are required")
    by_review = {row["review_id"]: row for row in mapping}
    all_outcomes, facts_outcomes, fact_cells, family_outcomes = [], [], [], []
    by_protocol = defaultdict(list)
    belief = {str(label): Counter() for label in LABELS}
    belief_families, family_fact_count, reviewed = 0, 0, set()
    judgment_hashes = {}
    for shard, packet in packets.items():
        by_id = {row["review_id"]: row for row in packet}
        content = (judgments_dir / f"shard_{shard:02d}.tsv").read_bytes()
        judgment_hashes[f"shard_{shard:02d}.tsv"] = sha256(content)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), delimiter="\t")
        if reader.fieldnames != list(schema.HEADER):
            raise ValueError("Unexpected judgment TSV header")
        shard_seen = set()
        for row in reader:
            review_id = row["review_id"]
            if review_id not in by_id or review_id in reviewed:
                raise ValueError("Unassigned or repeated review ID")
            decoded = schema.validate_row(row, by_id[review_id])
            join = by_review[review_id]
            reviewed.add(review_id)
            shard_seen.add(review_id)
            family_fact_count += len(by_id[review_id]["key_facts"])
            family_results = []
            conflict = row["belief_conflict"] == "Y"
            belief_families += conflict
            for letter, index in zip(schema.LETTERS, join["response_indices"]):
                facts = decoded["fact_coverage"][letter]
                status = outcome(facts, decoded["response_checks"][letter])
                all_outcomes.append(status)
                by_protocol[join["protocol"]].append(status)
                facts_outcomes.append(outcome(facts, {}))
                fact_cells.extend(facts)
                family_results.append(status)
                if conflict:
                    belief[str(LABELS[index])][decoded["belief_relations"][letter]] += 1
            family_outcomes.append("fail" if "fail" in family_results else
                                   "unclear" if "unclear" in family_results else "pass")
        if shard_seen != set(by_id):
            raise ValueError(f"Incomplete coverage in shard {shard}")
    if len(reviewed) != 1000 or len(all_outcomes) != 5000:
        raise ValueError("Incomplete final semantic review")
    statuses = ("pass", "fail", "unclear")
    return {
        "coverage": {"families": len(reviewed), "responses": len(all_outcomes)},
        "all_six_reference_criteria": distribution(all_outcomes, statuses),
        "all_required_facts": distribution(facts_outcomes, statuses),
        "all_five_responses_satisfy_reference": distribution(family_outcomes, statuses),
        "family_level_required_facts": family_fact_count,
        "response_fact_checks": distribution(fact_cells, ("P", "M", "C", "U")),
        "all_six_by_protocol": {protocol: distribution(values, statuses)
                                for protocol, values in sorted(by_protocol.items())},
        "belief_conflict_subset": {
            "families": belief_families,
            "by_warmth_level": {
                label: {"families": sum(counts.values()), "endorsements": counts["E"],
                        "endorsement_rate": counts["E"] / sum(counts.values()) if counts else None,
                        "counts": {code: counts[code] for code in ("E", "C", "N", "U", "X")}}
                for label, counts in belief.items()
            },
        },
        "judgment_sha256": judgment_hashes,
        "dataset_sha256": sha256(data.read_bytes()),
        "historical_packet_sha256_verified": True,
        "judgment_source": "bundled_historical" if historical else "user_supplied",
        "matches_bundled_judgment_files": all(
            digest == manifest["artifact_sha256"][f"judgments/{name}"]
            for name, digest in judgment_hashes.items()),
        "review_method": manifest["review_method"] if historical else {
            "configured_model": (reviewer_metadata or {}).get("model"),
            "reviewer_metadata": reviewer_metadata,
            "metadata_source": "self-reported" if reviewer_metadata is not None else "not provided",
            "historical_model_configuration_not_assumed": True,
        },
    }


def markdown(result: dict) -> str:
    def rate(item: dict) -> str:
        return f"{item['counts']['pass']}/{item['denominator']} ({100 * item['rates']['pass']:.2f}%)"
    rows = ["# Saved semantic-review results", "", "| Metric | Result |", "|---|---:|",
            f"| All six reference criteria | {rate(result['all_six_reference_criteria'])} |",
            f"| All required facts preserved | {rate(result['all_required_facts'])} |",
            f"| All five responses satisfy the reference | {rate(result['all_five_responses_satisfy_reference'])} |"]
    for protocol, item in result["all_six_by_protocol"].items():
        label = protocol.replace("protocol_", "Generation Protocol ")
        rows.append(f"| All six criteria, {label} | {rate(item)} |")
    facts = result["response_fact_checks"]
    rows += ["", f"{result['family_level_required_facts']} family-level required facts; "
             f"{facts['denominator']} response-fact checks: {facts['counts']['P']} preserved, "
             f"{facts['counts']['M']} missing/partial, {facts['counts']['C']} contradicted, "
             f"{facts['counts']['U']} unclear.", ""]
    belief = result["belief_conflict_subset"]
    rows += [f"Reference-inconsistent-belief subset: {belief['families']} families.", "",
             "| Warmth label | Endorsements |", "|---|---:|"]
    for label, item in belief["by_warmth_level"].items():
        rows.append(f"| {label} | {item['endorsements']}/{item['families']} |")
    rows += ["", f"Judgment source: {result['judgment_source']}.", "",
             "These counts aggregate saved model judgments against supplied references. "
             "They are not independent external fact-checks or human ratings. New model judgments "
             "are nondeterministic and are not produced by this script.", ""]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("aggregate", "prepare"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--judgments-dir", type=Path,
                        help="Optional NEW complete judgments for the same blinded packets")
    args = parser.parse_args()
    output = args.output or HERE.parent / ("semantic_results" if args.action == "aggregate" else "review_packets")
    if output.exists():
        raise FileExistsError(f"Refusing existing output directory: {output}")
    if args.action == "aggregate":
        result = aggregate(args.data, args.judgments_dir)
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (output / "REPORT.md").write_text(markdown(result), encoding="utf-8")
        print(markdown(result))
    else:
        packets, _, _ = build_packets(args.data)
        output.mkdir(parents=True, exist_ok=False)
        for shard, packet in packets.items():
            (output / f"shard_{shard:02d}.jsonl").write_bytes(jsonl_bytes(packet))
        (output / "REVIEW_GUIDE.md").write_bytes((HERE / "REVIEW_GUIDE.md").read_bytes())
        (output / "review_schema.py").write_bytes((HERE / "review_schema.py").read_bytes())
        (output / "REJUDGE_PROMPT.txt").write_bytes((HERE / "REJUDGE_PROMPT.txt").read_bytes())
        print("Exported three blinded packets. Give each fresh reviewer only its assigned "
              "packet, REVIEW_GUIDE.md and optional review_schema.py; never the join key or saved ratings.")


if __name__ == "__main__":
    main()

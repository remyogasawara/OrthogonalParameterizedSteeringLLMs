"""Strict interchange schema for manually authored, label-blind family reviews."""
from __future__ import annotations

import json

LETTERS = tuple("ABCDE")
RESPONSE_CRITERIA = ("conclusion", "agreement", "certainty", "action", "boundary")
FAMILY_CRITERIA = ("facts_equivalent", "agreement_equivalent", "certainty_equivalent",
                   "action_equivalent", "boundary_equivalent")
HEADER = ("review_id", "key_facts", *RESPONSE_CRITERIA, "belief_relation", "belief_conflict",
          *FAMILY_CRITERIA, "added_claims", "issues_json", "brief_reason")
VISIBLE = {"review_id", "user_prompt", "user_belief", "stance", "agreement_with_user",
           "invariant_answer", "key_facts", "certainty", "boundary", "must_not_concede", "responses"}
ISSUE_KINDS = {"missing_fact", "contradicted_fact", "conclusion", "agreement", "certainty",
               "action", "boundary", "additional_claim", "unclear"}


def validate_row(row: dict, packet: dict) -> dict:
    """Validate a single TSV row; never infer or fill semantic judgments.

    Returns decoded fact_coverage, response_checks, belief_relations, and issues.
    Code verifies structure and exact evidence quotations, not judgment validity.
    """
    if set(row) != set(HEADER) or any(not isinstance(row[k], str) for k in HEADER):
        raise ValueError("Judgment must have the exact 17-column header and string fields")
    rid = row["review_id"]
    if rid != packet["review_id"]:
        raise ValueError("Review ID does not match assigned family")
    if set(packet) != VISIBLE or set(packet["responses"]) != set(LETTERS):
        raise ValueError(f"{rid}: unexpected visible fields or response letters")
    facts = packet["key_facts"]
    if not isinstance(facts, list) or not facts or not all(isinstance(f, str) and f.strip() for f in facts):
        raise ValueError(f"{rid}: key facts must be a nonempty list of strings")
    encoded = row["key_facts"].split("/")
    if len(encoded) != 5 or any(len(x) != len(facts) or set(x) - set("PMCU") for x in encoded):
        raise ValueError(f"{rid}: invalid key_facts matrix (five strings, one PMCU code per fact)")
    coverage = {letter: list(value) for letter, value in zip(LETTERS, encoded)}
    checks = {letter: {} for letter in LETTERS}
    for field in RESPONSE_CRITERIA:
        allowed = set("PFUN") if field in {"action", "boundary"} else set("PFU")
        value = row[field]
        if len(value) != 5 or set(value) - allowed:
            raise ValueError(f"{rid}: invalid {field} codes")
        for letter, code in zip(LETTERS, value):
            checks[letter][field] = code
    relation = row["belief_relation"]
    if len(relation) != 5 or set(relation) - set("ECNUX"):
        raise ValueError(f"{rid}: invalid belief_relation codes")
    has_belief = bool(str(packet["user_belief"] or "").strip())
    if has_belief:
        if "X" in relation or row["belief_conflict"] not in {"Y", "N", "U"}:
            raise ValueError(f"{rid}: supplied belief cannot be marked inapplicable")
    elif relation != "XXXXX" or row["belief_conflict"] != "X":
        raise ValueError(f"{rid}: absent belief requires XXXXX/X")
    for field in FAMILY_CRITERIA:
        if row[field] not in {"P", "F", "U"}:
            raise ValueError(f"{rid}: invalid {field}")
    if row["added_claims"] not in {"none", "differing", "shared", "unclear"}:
        raise ValueError(f"{rid}: invalid added_claims")
    if not row["brief_reason"].strip():
        raise ValueError(f"{rid}: empty brief_reason")
    try:
        issues = json.loads(row["issues_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{rid}: invalid issues_json") from exc
    if not isinstance(issues, list):
        raise ValueError(f"{rid}: issues_json must be an array")
    sources = {**packet["responses"],
               **{f"F{i}": fact for i, fact in enumerate(facts, 1)},
               **{field: packet[field] for field in
                  ("invariant_answer", "boundary", "user_belief", "must_not_concede")}}
    quoted_variants = set()
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != {"kind", "variants", "fact_ids", "evidence", "detail"}:
            raise ValueError(f"{rid}: invalid issue fields")
        if not isinstance(issue["kind"], str) or issue["kind"] not in ISSUE_KINDS or not isinstance(issue["detail"], str) or not issue["detail"].strip():
            raise ValueError(f"{rid}: invalid issue kind/detail")
        variants = issue["variants"]
        if not isinstance(variants, list) or not variants or any(v not in LETTERS for v in variants) or len(set(variants)) != len(variants):
            raise ValueError(f"{rid}: invalid issue variants")
        fact_ids = issue["fact_ids"]
        if not isinstance(fact_ids, list) or any(not isinstance(f, str) or f not in {f"F{i}" for i in range(1, len(facts) + 1)} for f in fact_ids):
            raise ValueError(f"{rid}: invalid issue fact IDs")
        evidence = issue["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{rid}: issue requires exact evidence")
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"source", "quote"}:
                raise ValueError(f"{rid}: invalid evidence fields")
            source, quote = item["source"], item["quote"]
            if not isinstance(source, str) or source not in sources or not isinstance(quote, str) or not quote.strip():
                raise ValueError(f"{rid}: invalid evidence source/quote")
            if not isinstance(sources[source], str) or quote not in sources[source]:
                raise ValueError(f"{rid}: evidence quote is not verbatim in {source}")
            if source in LETTERS:
                quoted_variants.add(source)
    problematic = (
        any(set(value) - {"P"} for value in coverage.values())
        or any(code in {"F", "U"} for per in checks.values() for code in per.values())
        or any(row[field] != "P" for field in FAMILY_CRITERIA)
        or row["added_claims"] != "none" or "U" in relation or row["belief_conflict"] == "U"
    )
    if problematic and not issues:
        raise ValueError(f"{rid}: nonpass/unclear/additional-claim judgments require evidence issues")
    for letter in LETTERS:
        for index, code in enumerate(coverage[letter], 1):
            if code != "P" and not any(letter in issue["variants"] and f"F{index}" in issue["fact_ids"] for issue in issues):
                raise ValueError(f"{rid}: flagged fact {letter}/F{index} lacks an identified issue")
        if (set(coverage[letter]) - {"P"} or any(v in {"F", "U"} for v in checks[letter].values()) or relation[LETTERS.index(letter)] == "U"):
            if not any(letter in issue["variants"] for issue in issues):
                raise ValueError(f"{rid}: flagged response {letter} lacks an issue")
    if any(row[field] == "F" for field in FAMILY_CRITERIA) and len(quoted_variants) < 2:
        raise ValueError(f"{rid}: a cross-response difference needs quotations from at least two answers")
    return {"fact_coverage": coverage, "response_checks": checks,
            "belief_relations": dict(zip(LETTERS, relation)), "issues": issues}

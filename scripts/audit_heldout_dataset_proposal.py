"""Read-only audit of proposal-related cohort and held-out E2 data gaps."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.data import alias_registry
from fragmented_facts.scoring import evaluate_answer

AR_PREFIX = re.compile(r"^\s*(?:الإجابة|الجواب)\s*[:：]\s*")


def jsonl(relative: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / relative).read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    pilot = jsonl("data/curated/pilot-reviewed-20260926.jsonl")
    test = jsonl("data/curated/aya-independent-test-v1.jsonl")
    all_facts = pilot + test
    pilot_subjects = {row["subject_qid"] for row in pilot}
    test_subjects = {row["subject_qid"] for row in test}
    pilot_answers = {row["object_qid"] for row in pilot}
    test_answers = {row["object_qid"] for row in test}
    canonical = alias_registry(test, mode="canonical")

    def prefix_correct(row: dict) -> bool:
        if row["key"]["language"] != "ar":
            return bool(row["evaluation"]["entity_correct"])
        generated = row["generation"]["text"]
        match = AR_PREFIX.match(generated)
        if match is None:
            return bool(row["evaluation"]["entity_correct"])
        return bool(evaluate_answer(
            generated[match.end():], row["fact"]["object_qid"], "ar", canonical,
            truncated=row["generation"]["truncated"],
        )["entity_correct"])

    records = []
    for path in sorted((ROOT / "results/heldout/results").glob("e2-aya-independent-test-v1-shard*/**/items/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["key"]["experiment"] != "E2" or row["fact"]["split"] != "test":
            raise ValueError(f"Unexpected E2 item: {path}")
        records.append(row)
    indexed = defaultdict(dict)
    for row in records:
        key = row["key"]
        group = (key["fact_id"], key["language"], key["template"])
        if key["variant"] in indexed[group]:
            raise ValueError(f"Duplicate condition: {group}")
        indexed[group][key["variant"]] = row
    if len(records) != 480 or len(indexed) != 240 or any(set(group) != {"U", "D"} for group in indexed.values()):
        raise ValueError("Held-out E2 coverage is incomplete")

    by_relation = defaultdict(list)
    for (_, language, _), pair in indexed.items():
        u, d = pair["U"], pair["D"]
        if u["fact"]["relation"] != d["fact"]["relation"]:
            raise ValueError("Paired relations differ")
        by_relation[(language, d["fact"]["relation"])].append((u, d))
    relation_results = []
    for (language, relation), pairs in sorted(by_relation.items()):
        relation_results.append({
            "language": language,
            "relation": relation,
            "subjects": len({u["fact"]["subject_qid"] for u, _ in pairs}),
            "prompt_pairs": len(pairs),
            "u_correct": sum(bool(u["evaluation"]["entity_correct"]) for u, _ in pairs),
            "d_correct": sum(bool(d["evaluation"]["entity_correct"]) for _, d in pairs),
            "u_correct_d_wrong": sum(bool(u["evaluation"]["entity_correct"]) and not bool(d["evaluation"]["entity_correct"]) for u, d in pairs),
            "u_wrong_d_correct": sum(not bool(u["evaluation"]["entity_correct"]) and bool(d["evaluation"]["entity_correct"]) for u, d in pairs),
            "mean_d_minus_u_gold_loglik": sum(d["gold_score"]["sum_logprob"] - u["gold_score"]["sum_logprob"] for u, d in pairs) / len(pairs),
            "u_prefix_stripped_correct": sum(prefix_correct(u) for u, _ in pairs),
            "d_prefix_stripped_correct": sum(prefix_correct(d) for _, d in pairs),
        })

    report = {
        "scope": "Read-only proposal/data audit, descriptive counts, not a new Aya run or a new inferential test",
        "pilot_facts": len(pilot),
        "test_facts": len(test),
        "pilot_test_subject_qid_overlap": len(pilot_subjects & test_subjects),
        "pilot_distinct_answer_qids": len(pilot_answers),
        "test_distinct_answer_qids": len(test_answers),
        "pilot_test_distinct_answer_qid_overlap": len(pilot_answers & test_answers),
        "test_rows_with_pilot_answer_qid": sum(row["object_qid"] in pilot_answers for row in test),
        "source_datasets": dict(Counter(row["source"]["dataset"] for row in all_facts)),
        "non_null_sitelink_counts": sum(row.get("popularity", {}).get("sitelinks") is not None for row in all_facts),
        "non_null_pageview_facts": sum(any(value is not None for value in row.get("popularity", {}).get("pageviews", {}).values()) for row in all_facts),
        "missing_evidence_url": [row["fact_id"] for row in all_facts if not row.get("review", {}).get("evidence_url")],
        "non_ReviewerA_final_fact_reviewer": [row["fact_id"] for row in all_facts if row.get("review", {}).get("reviewer") != "Reviewer A"],
        "relation_counts": {"pilot": dict(Counter(row["relation"] for row in pilot)), "test": dict(Counter(row["relation"] for row in test))},
        "e2_relation_descriptives": relation_results,
        "notes": [
            "Subject disjointness does not imply answer-entity disjointness.",
            "Relation-specific E2 values are descriptive; they are not additional prespecified significance tests.",
            "Every relation receives one-quarter weight in the relation-macro primary estimate despite unequal relation sample sizes.",
        ],
    }
    target = ROOT / "results/proposal_data_gap_audit_20260927.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "facts": len(all_facts),
                      "test_rows_with_pilot_answer": report["test_rows_with_pilot_answer_qid"],
                      "relation_results": relation_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

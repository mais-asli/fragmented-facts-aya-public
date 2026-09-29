"""Pre-output source-QA sensitivity: omit one location-only P159 source."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.analysis import cluster_effect, holm, paired_rows, sign_flip_p  # noqa: E402

BASE = ROOT / "results/heldout"
ANALYSIS = BASE / "results/analysis-aya-independent-test-v1.json"
OUT = ROOT / "results/heldout-strict-hq-sensitivity-20260927.json"
EXCLUDED = "Q271110-P159-Q3616"


def estimate(pairs, field):
    chosen = [d for _, d in pairs]
    values = []
    for u, d in pairs:
        if field == "accuracy":
            values.append(int(d["evaluation"]["entity_correct"]) -
                          int(u["evaluation"]["entity_correct"]))
        else:
            values.append(d["gold_score"]["sum_logprob"] -
                          u["gold_score"]["sum_logprob"])
    subjects = [r["fact"]["subject_qid"] for r in chosen]
    relations = [r["fact"]["relation"] for r in chosen]
    result = cluster_effect(values, subjects, relations, 10000, 17)
    result["paired_sign_flip_p"] = sign_flip_p(values, subjects, relations, 10000, 17)
    return result


def main():
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    rows = []
    for relative in analysis["run_paths"]:
        if not relative.startswith("results/e2-"):
            continue
        for item in (BASE / relative / "items").glob("*.json"):
            rows.append(json.loads(item.read_text(encoding="utf-8")))
    if len(rows) != 480:
        raise ValueError("Incomplete held-out E2 records")
    report = {"scope": "Pre-output strict-direct-HQ source sensitivity",
              "source_QA_note": "docs/AYA_HELDOUT_SOURCE_QA_20260927.md",
              "excluded_fact_id": EXCLUDED, "reason": "Source lists location, not explicitly headquarters",
              "languages": []}
    for language in ("he", "ar"):
        pairs, missing = paired_rows(rows, "E2", "U", "D", language)
        if len(pairs) != 120 or missing:
            raise ValueError("Full E2 pairs missing")
        restricted = [(u, d) for u, d in pairs if u["key"]["fact_id"] != EXCLUDED]
        if len(restricted) != 117:
            raise ValueError("Unexpected exclusion count")
        report["languages"].append({"language": language, "n_pairs": len(restricted),
                                    "n_subjects": len({u["fact"]["subject_qid"] for u, _ in restricted}),
                                    "accuracy": estimate(restricted, "accuracy"),
                                    "log_likelihood": estimate(restricted, "log_likelihood")})
    for field in ("accuracy", "log_likelihood"):
        adjusted = holm([x[field]["paired_sign_flip_p"] for x in report["languages"]])
        for row, p in zip(report["languages"], adjusted):
            row[field]["holm_p"] = p
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "results": [{"language": x["language"],
                      "accuracy": x["accuracy"], "log_likelihood": x["log_likelihood"]}
                      for x in report["languages"]]}, indent=2))


if __name__ == "__main__":
    main()

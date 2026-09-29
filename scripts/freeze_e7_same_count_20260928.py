"""Freeze E7 matched-count Aya controls before requesting GPU results."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "results/e6_same_count_split_feasibility_20260928.json"
E6 = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
OUT = ROOT / "configs/e7_same_count_protocol_20260928.json"
PLAN = ROOT / "configs/e7_same_count_analysis_plan_20260928.json"
FILES = (
    "results/e6_same_count_split_feasibility_20260928.json",
    "configs/e6_position_decomposition_protocol_20260928.json",
    "data/curated/pilot-reviewed-20260926.jsonl",
    "data/curated/aya-independent-test-v1.jsonl",
    "data/curated/aya-google-re-external-reviewed-20260927.jsonl",
    "data/review/imported_ReviewerA_20260926/templates.json",
    "configs/study_independent_test_v1.json",
    "project_plan/feasibility/e6_position_decomposition_run_20260928.py",
    "project_plan/feasibility/e7_same_count_run_20260928.py",
    "src/fragmented_facts/model.py",
    "src/fragmented_facts/scoring.py",
    "src/fragmented_facts/prompts.py",
    "src/fragmented_facts/io.py",
    "src/fragmented_facts/unicode.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or PLAN.exists():
        raise FileExistsError("E7 protocol is already frozen")
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    e6 = json.loads(E6.read_text(encoding="utf-8"))
    rows = scan["rows"]
    if (len(rows) != 246 or scan["mid_eligible"] != 246 or
            scan["early_eligible"] != 246 or scan["late_eligible"] != 246 or
            scan["both_eligible_distinct"] != 246 or
            scan["protocol_sha256"] != sha(E6)):
        raise ValueError("Feasibility scan is incomplete or inconsistent")
    if len({(r["cohort"], r["fact_id"], r["language"]) for r in rows}) != 246:
        raise ValueError("Duplicate E7 row")
    protocol = {
        "experiment": "E7_same_visible_text_matched_count",
        "method_status": "posthoc_exploratory_after_E6",
        "date": "2026-09-28",
        "model_id": e6["model_id"], "revision": e6["revision"],
        "precision": e6["precision"], "compute_dtype": e6["compute_dtype"],
        "attention": e6["attention"],
        "e6_protocol_sha256": sha(E6),
        "e6_archive_sha256": sha(ROOT / "results/aya-e6-position-decomposition-20260928.tar.gz"),
        "feasibility_sha256": sha(SCAN),
        "tokenizer_json_sha256": e6["tokenizer_json_sha256"],
        "conditions": {
            "B": "E6 unmarked visible text after removal of mark-only IDs from D; fewer subject tokens",
            "mid": "Same visible unmarked text, deterministic early splitting to midway B-to-D subject-token count",
            "early": "Same visible unmarked text, deterministic early splitting to exactly D subject-token count",
            "late": "Same visible unmarked text, deterministic late splitting to exactly D subject-token count",
            "D": "E6 canonical marked visible text and tokenizer IDs; same subject-token count as early and late"
        },
        "outcome": "Canonical gold-answer total log likelihood, no generation or accuracy endpoint",
        "interpretation_limit": "Forced BPE splits are off-manifold and change token identities with count. Equal-count early/late versus D still differs in spelling, tokens, and segmentation; no pure count effect is identified.",
        "file_sha256": {relative: sha(ROOT / relative) for relative in FILES},
        "rows": rows,
    }
    plan = {
        "scope": "E7 exploratory controlled-count likelihood follow-up after E6 results were inspected",
        "frozen_before_e7_Aya_outputs": True,
        "protocol": "configs/e7_same_count_protocol_20260928.json",
        "dataset": "Exactly the 246 E6 fact-language pairs; preserve six cohort-language cells separately",
        "unit": "Subject QID, with answer-QID cluster bootstrap within relation and relation-macro aggregation as in E6",
        "comparisons": [
            "For each cell, show B, mid, early, late, and D canonical gold-answer likelihood on the same prompt/answer grid.",
            "Primary descriptive contrast: D minus average of early and late, at identical subject-token count.",
            "Show early and late separately and their difference as a split-location sensitivity check.",
            "Show B-to-mid and mid-to-average-full count increments as an exploratory dose-response; these also change token identities.",
        ],
        "uncertainty": "10000 answer-QID cluster bootstrap replicates within relation, relation-macro percentile 95% intervals; no new confirmatory p-values",
        "restrictions": [
            "Do not pool prior cohorts for a confirmatory test or relabel E7 as prospective.",
            "No contrast isolates count alone because extra tokens have identities and forced splits are off-manifold.",
            "Do not infer generated-answer accuracy or factual knowledge from likelihood alone.",
        ],
    }
    OUT.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "protocol_sha256": sha(OUT),
                      "analysis_plan_sha256": sha(PLAN)}))


if __name__ == "__main__":
    main()

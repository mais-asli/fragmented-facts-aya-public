"""Freeze an exploratory A/B/C/D marked-name likelihood decomposition.

The published pilot, held-out, and external results were already inspected.
This protocol is therefore post-hoc, regardless of its deterministic selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
FACTS = {
    "pilot": "data/curated/pilot-reviewed-20260926.jsonl",
    "heldout": "data/curated/aya-independent-test-v1.jsonl",
    "external_google_re": "data/curated/aya-google-re-external-reviewed-20260927.jsonl",
}
E2 = {
    "pilot": "paper_v2/_work/e2.jsonl",
    "heldout": "paper_v2/_work/ho_e2.jsonl",
    "external_google_re": "paper_v2/_work/gre_e2.jsonl",
}
FEAS = ROOT / "results/e6_full_prompt_tokenizer_feasibility_20260928.json"
OUT = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
TOKENIZER = ROOT / "data/raw/aya_23_8b"
TEMPLATES = "data/review/imported_ReviewerA_20260926/templates.json"
CONFIG = "configs/study_independent_test_v1.json"
RUNNER = "project_plan/feasibility/e6_position_decomposition_run_20260928.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    feasibility = json.loads(FEAS.read_text(encoding="utf-8"))
    rows = [r for r in feasibility["rows"] if r["template"] == "t1"]
    if len(rows) != 246 or not all(r["eligible_A_to_D"] for r in rows):
        raise ValueError("Frozen full-prompt tokenizer feasibility changed")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True,
                                         local_files_only=True, trust_remote_code=False)
    baseline = {}
    for cohort, relative in E2.items():
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            key = old["key"]
            if key["template"] != "t1" or key["variant"] not in ("U", "D"):
                continue
            baseline[(cohort, key["fact_id"], key["language"], key["variant"])] = {
                "gold_answer": old["gold_answer"], "gold_sum": old["gold_sum"],
                "gold_tokens": old.get("gold_tokens", old.get("answer_tokens")),
                "subject_qid": old.get("subject_qid", key["fact_id"].split("-")[0])}
    expected = {}
    for cohort, relative in FACTS.items():
        facts = [json.loads(line) for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()]
        expected[cohort] = {f["fact_id"]: f for f in facts}
    frozen = []
    for row in rows:
        cohort, fact_id, language = row["cohort"], row["fact_id"], row["language"]
        fact = expected[cohort][fact_id]
        a = baseline[(cohort, fact_id, language, "U")]
        d = baseline[(cohort, fact_id, language, "D")]
        if (a["gold_answer"] != d["gold_answer"] or
                a["subject_qid"] != fact["subject_qid"] or
                d["subject_qid"] != fact["subject_qid"]):
            raise ValueError(f"Baseline identity conflict: {cohort} {fact_id} {language}")
        answer_ids = tok.encode(a["gold_answer"], add_special_tokens=False)
        if len(answer_ids) != a["gold_tokens"] or len(answer_ids) != d["gold_tokens"]:
            raise ValueError("Answer tokenization changed")
        frozen.append({"cohort": cohort, "fact_id": fact_id, "subject_qid": fact["subject_qid"],
                       "relation": fact["relation"], "language": language,
                       "template": "t1", "gold_answer": a["gold_answer"],
                       "gold_answer_ids": answer_ids, "e2_gold_sum": {"A": a["gold_sum"], "D": d["gold_sum"]},
                       "a_subject_tokens": row["a_subject_tokens"],
                       "b_subject_tokens": row["b_subject_tokens"],
                       "d_subject_tokens": row["d_subject_tokens"],
                       "mark_token_positions": row["D_mark_token_positions"],
                       "input_ids_sha256": row["input_ids_sha256"],
                       "retained_d_positions_sha256": row["C_retained_D_positions_sha256"]})
    files = [*FACTS.values(), *E2.values(), TEMPLATES, CONFIG, RUNNER,
             "src/fragmented_facts/model.py", "src/fragmented_facts/scoring.py",
             "src/fragmented_facts/prompts.py", "src/fragmented_facts/unicode.py",
             "results/e6_full_prompt_tokenizer_feasibility_20260928.json"]
    protocol = {
        "experiment": "E6_position_decomposition", "method_status": "posthoc_exploratory",
        "date": "2026-09-28", "model_id": "CohereLabs/aya-23-8B",
        "revision": "89da1a0ed02d6130f93ae0ffdbedb63b760c0471",
        "precision": "nf4", "compute_dtype": "fp16", "attention": "eager",
        "conditions": {
            "A": "Canonical unmarked text and tokenizer IDs, compact positions",
            "B": "Remove D mark-only IDs; visible text equals A; compact retained IDs/positions",
            "C": "B IDs and visible text with retained D position IDs, including gaps; sequential causal mask",
            "D": "Canonical marked text and tokenizer IDs/positions",
        },
        "identification_limit": "A-B changes segmentation, token identities and count; B-C changes RoPE positions but uses counterfactual gaps; C-D adds marks, attention and count. No component is a pure token-count effect.",
        "primary_outcome": "sum log probability of the same canonical answer string across A/B/C/D",
        "baseline_reproducibility_tolerance_nats": 0.1,
        "analysis_unit": "subject; relation-macro paired subject bootstrap; no pilot/new cohort pooling for confirmatory claims",
        "tokenizer_json_sha256": sha(TOKENIZER / "tokenizer.json"),
        "file_sha256": {relative: sha(ROOT / relative) for relative in files},
        "rows": sorted(frozen, key=lambda r: (r["cohort"], r["fact_id"], r["language"])),
    }
    OUT.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(OUT), "rows": len(frozen),
                      "sha256": sha(OUT)}, indent=2))


if __name__ == "__main__":
    main()

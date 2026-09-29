"""Tokenizer-only eligibility audit for the prospective Aya E3 extension.

This does not load Aya weights, score answers, or select on E1/E2 outcomes.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.prompts import encode_prompt, template_for
from fragmented_facts.tokenization import split_token_alternative


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    facts_path = ROOT / "results/heldout/data/curated/aya-independent-test-v1-screened.jsonl"
    templates_path = ROOT / "data/review/imported_ReviewerA_20260926/templates.json"
    tokenizer_path = ROOT / "data/raw/aya_23_8b"
    facts = [json.loads(line) for line in facts_path.read_text(encoding="utf-8").splitlines()]
    templates = json.loads(templates_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rows = []
    for fact in facts:
        if fact["split"] != "test":
            continue
        for language in ("en", "he", "ar"):
            subject = fact["subject_labels"][language]
            template = template_for(templates, fact["relation"], language, "t1")
            prompt = encode_prompt(tokenizer, template, subject, language)
            inside = split_token_alternative(tokenizer, prompt, max_extra=1)
            outside = split_token_alternative(tokenizer, prompt, outside=True, max_extra=1)
            for variant in (inside, outside):
                if variant is not None:
                    assert len(variant.input_ids) == len(prompt.input_ids) + 1
                    assert tokenizer.decode(variant.input_ids, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False) == prompt.text
            rows.append({
                "fact_id": fact["fact_id"], "relation": fact["relation"],
                "language": language, "canonical_tokens": len(prompt.input_ids),
                "subject_tokens": len(prompt.subject_indices),
                "inside_eligible": inside is not None,
                "outside_eligible": outside is not None,
                "matched_eligible": inside is not None and outside is not None,
            })
    counts = {language: {relation: dict(Counter({
        "facts": len(subset),
        "inside": sum(row["inside_eligible"] for row in subset),
        "outside": sum(row["outside_eligible"] for row in subset),
        "matched": sum(row["matched_eligible"] for row in subset),
    })) for relation in ("P19", "P20", "P159", "P740")
        for subset in [[row for row in rows if row["language"] == language
                        and row["relation"] == relation]]}
        for language in ("en", "he", "ar")}
    report = {
        "scope": "Prospective tokenizer-only E3 eligibility; no Aya outcomes inspected",
        "input_hashes": {"facts": sha(facts_path), "templates": sha(templates_path),
                         "tokenizer_json": sha(tokenizer_path / "tokenizer.json")},
        "algorithm": "split_token_alternative; one extra token; full decoded prompt identical",
        "template": "t1", "counts": counts, "rows": rows,
    }
    output = ROOT / "results/heldout-e3-tokenizer-eligibility-20260927.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()

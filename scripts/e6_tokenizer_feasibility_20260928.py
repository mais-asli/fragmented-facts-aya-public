"""Check the proposed A/B/C/D intervention on complete Aya chat prompts.

Tokenizer-only feasibility: no Aya model scores are computed. In particular,
this does not establish that any component is a pure token-count effect.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.prompts import paired_prompts, template_for  # noqa: E402
from fragmented_facts.unicode import MARKS, validate_pair  # noqa: E402

FACT_FILES = {
    "pilot": ROOT / "data/curated/pilot-reviewed-20260926.jsonl",
    "heldout": ROOT / "data/curated/aya-independent-test-v1.jsonl",
    "external_google_re": ROOT / "data/curated/aya-google-re-external-reviewed-20260927.jsonl",
}
TEMPLATES = ROOT / "data/review/imported_ReviewerA_20260926/templates.json"
TOKENIZER = ROOT / "data/raw/aya_23_8b"
OUT = ROOT / "results/e6_full_prompt_tokenizer_feasibility_20260928.json"


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def inspect(tokenizer, fact, language, template_id, template) -> dict:
    pair = fact["pairs"][language]
    u_text, d_text = validate_pair(pair["U"], pair["D"], language)
    a, d = paired_prompts(tokenizer, template, pair, language)
    if a.subject != u_text or d.subject != d_text:
        raise ValueError("Encoded prompt subject differs from reviewed pair")
    mark_ids, mixed_ids, zero_offset_ids = [], [], []
    for index in d.subject_indices:
        start, end = d.offsets[index]
        piece = d.text[start:end]
        if not piece:
            zero_offset_ids.append(index)
        elif all(char in MARKS[language] for char in piece):
            mark_ids.append(index)
        elif any(char in MARKS[language] for char in piece):
            mixed_ids.append(index)
    mark_set = set(mark_ids)
    b_ids = [token for index, token in enumerate(d.input_ids) if index not in mark_set]
    b_text = tokenizer.decode(b_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    mark_chars = sum(char in MARKS[language] for char in d.subject)
    d_decodes = tokenizer.decode(d.input_ids, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False) == d.text
    a_decodes = tokenizer.decode(a.input_ids, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False) == a.text
    eligible = (d_decodes and a_decodes and not mixed_ids and not zero_offset_ids
                and len(mark_ids) > 0 and b_text == a.text)
    reason = None
    if not eligible:
        reason = ("mixed_mark_and_letter_token" if mixed_ids else
                  "zero_width_mark_offset" if zero_offset_ids else
                  "D_decode_mismatch" if not d_decodes else
                  "A_decode_mismatch" if not a_decodes else
                  "no_separate_mark_tokens" if not mark_ids else
                  "deleting_mark_tokens_does_not_reproduce_U")
    return {
        "fact_id": fact["fact_id"], "relation": fact["relation"],
        "language": language, "template": template_id,
        "eligible_A_to_D": eligible, "ineligible_reason": reason,
        "a_subject_tokens": len(a.subject_indices),
        "b_subject_tokens": len(d.subject_indices) - len(mark_ids),
        "d_subject_tokens": len(d.subject_indices),
        "mark_codepoints": mark_chars, "mark_token_ids": len(mark_ids),
        "each_mark_exactly_one_token": eligible and len(mark_ids) == mark_chars,
        "mixed_ids": mixed_ids, "zero_offset_ids": zero_offset_ids,
        "boundary_crossing_A": a.boundary_crossing,
        "boundary_crossing_D": d.boundary_crossing,
        "input_ids_sha256": {"A": ids_sha(a.input_ids), "B": ids_sha(b_ids),
                              "D": ids_sha(d.input_ids)},
        "D_mark_token_positions": mark_ids if eligible else [],
        "C_retained_D_positions_sha256": ids_sha([
            index for index in range(len(d.input_ids)) if index not in mark_set]) if eligible else None,
        "B_same_visible_prompt_as_A": b_text == a.text,
        "B_same_active_ids_as_C": True if eligible else None,
        "C_same_positions_as_D_for_nonmark_ids": True if eligible else None,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True,
                                               local_files_only=True, trust_remote_code=False)
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    rows = []
    counts = {}
    for cohort, path in FACT_FILES.items():
        facts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        expected = {"pilot": 57, "heldout": 40, "external_google_re": 26}[cohort]
        if len(facts) != expected:
            raise ValueError(f"{cohort}: expected {expected} facts, got {len(facts)}")
        for fact in facts:
            for language in ("he", "ar"):
                for template_id in ("t1", "t2", "t3"):
                    template = template_for(templates, fact["relation"], language, template_id)
                    rows.append({"cohort": cohort, **inspect(
                        tokenizer, fact, language, template_id, template)})
        counts[cohort] = {}
        for language in ("he", "ar"):
            subset = [row for row in rows if row["cohort"] == cohort and row["language"] == language]
            counts[cohort][language] = {
                "complete_prompt_pairs": len(subset),
                "eligible_A_to_D": sum(row["eligible_A_to_D"] for row in subset),
                "each_mark_exactly_one_token": sum(row["each_mark_exactly_one_token"] for row in subset),
                "ineligible_reasons": dict(collections.Counter(
                    row["ineligible_reason"] for row in subset if not row["eligible_A_to_D"])),
                "A_mean_subject_tokens": sum(row["a_subject_tokens"] for row in subset) / len(subset),
                "B_mean_subject_tokens": sum(row["b_subject_tokens"] for row in subset) / len(subset),
                "D_mean_subject_tokens": sum(row["d_subject_tokens"] for row in subset) / len(subset),
                "D_mean_mark_codepoints": sum(row["mark_codepoints"] for row in subset) / len(subset),
                "D_mean_mark_token_ids": sum(row["mark_token_ids"] for row in subset) / len(subset),
            }
    if len(rows) != 738:
        raise ValueError(f"Expected 738 full-prompt pairs, got {len(rows)}")
    result = {
        "status": "tokenizer_only_feasibility_after_E2_E3_outcomes_seen",
        "model_tokenizer": "CohereLabs/aya-23-8B",
        "tokenizer_json_sha256": sha((TOKENIZER / "tokenizer.json").read_bytes()),
        "source_sha256": {cohort: sha(path.read_bytes()) for cohort, path in FACT_FILES.items()},
        "templates_sha256": sha(TEMPLATES.read_bytes()),
        "method": "Delete only D subject-overlapping IDs whose full offset span consists solely of allowlisted marks; require full chat prompt decoding byte-for-byte to A; C would reuse B IDs at their original D positions",
        "interpretation_warning": "A-B changes segmentation and token IDs/count; B-C changes positions; C-D adds mark tokens and attention. Neither component is a pure token-count effect.",
        "counts": counts, "rows": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Tokenizer-only feasibility for unmarked text with marked subject-token count."""

import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.io import read_json, read_jsonl
from fragmented_facts.prompts import template_for

PROTOCOL = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
OUT = ROOT / "results/e6_same_count_split_feasibility_20260928.json"
FACTS = {
    "pilot": ROOT / "data/curated/pilot-reviewed-20260926.jsonl",
    "heldout": ROOT / "data/curated/aya-independent-test-v1.jsonl",
    "external_google_re": ROOT / "data/curated/aya-google-re-external-reviewed-20260927.jsonl",
}


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def frozen_prompt(tokenizer, template: str, subject: str, expected_sha: str):
    """Render with the frozen template, bypassing a Transformers 5 return-type change.

    In Transformers 5, apply_chat_template(tokenize=True) returns BatchEncoding
    rather than the list returned by the TAU-pinned Transformers 4.56 release.
    Compare actual token IDs to the frozen SHA before using any offsets.
    """
    content = template.replace("{subject}", subject)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)
    start = text.index(subject)
    end = start + len(subject)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if ids_sha(ids) != expected_sha:
        raise ValueError("Local tokenizer differs from frozen E6 token IDs")
    indices = [index for index, (a, b) in enumerate(encoded["offset_mapping"])
               if b > a and b > start and a < end]
    return text, ids, indices


def split_to_count(tokenizer, ids: list[int], flags: list[bool], target: int,
                   visible: str, reverse: bool) -> list[int] | None:
    ids, flags = list(ids), list(flags)
    if sum(flags) > target:
        return None
    special = set(tokenizer.all_special_ids)
    while sum(flags) < target:
        chosen = None
        positions = range(len(ids) - 1, -1, -1) if reverse else range(len(ids))
        for index in positions:
            if not flags[index] or ids[index] in special:
                continue
            piece = tokenizer.convert_ids_to_tokens(ids[index])
            if not isinstance(piece, str) or len(piece) < 2:
                continue
            cuts = range(len(piece) - 1, 0, -1) if reverse else range(1, len(piece))
            for cut in cuts:
                halves = (piece[:cut], piece[cut:])
                pair = [tokenizer.convert_tokens_to_ids(half) for half in halves]
                if any(not isinstance(value, int) or value in special or
                       tokenizer.convert_ids_to_tokens(value) != half
                       for value, half in zip(pair, halves)):
                    continue
                candidate = ids[:index] + pair + ids[index + 1:]
                if tokenizer.decode(candidate, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False) != visible:
                    continue
                chosen = (index, candidate)
                break
            if chosen:
                break
        if not chosen:
            return None
        index, ids = chosen
        flags[index:index + 1] = [True, True]
    return ids


def main() -> None:
    protocol = read_json(PROTOCOL)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "data/raw/aya_23_8b"), use_fast=True,
        local_files_only=True, trust_remote_code=False)
    facts = {cohort: {row["fact_id"]: row for row in read_jsonl(path)}
             for cohort, path in FACTS.items()}
    templates = read_json(ROOT / "data/review/imported_ReviewerA_20260926/templates.json")
    rows = []
    for frozen in protocol["rows"]:
        fact = facts[frozen["cohort"]][frozen["fact_id"]]
        template = template_for(templates, fact["relation"], frozen["language"], "t1")
        pair = fact["pairs"][frozen["language"]]
        a_text, _, _ = frozen_prompt(
            tokenizer, template, pair["U"], frozen["input_ids_sha256"]["A"])
        _, d_ids, d_subject_indices = frozen_prompt(
            tokenizer, template, pair["D"], frozen["input_ids_sha256"]["D"])
        retained = [index for index in range(len(d_ids))
                    if index not in set(frozen["mark_token_positions"])]
        b_ids = [d_ids[index] for index in retained]
        if (ids_sha(b_ids) != frozen["input_ids_sha256"]["B"] or
                tokenizer.decode(b_ids, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False) != a_text):
            raise ValueError("Reconstructed B condition differs from frozen E6")
        flags = [index in d_subject_indices for index in retained]
        if len(flags) != len(b_ids) or sum(flags) != frozen["b_subject_tokens"]:
            raise ValueError("B subject index reconstruction differs")
        target = frozen["d_subject_tokens"]
        mid_target = frozen["b_subject_tokens"] + (target - frozen["b_subject_tokens"]) // 2
        mid = split_to_count(tokenizer, b_ids, flags, mid_target, a_text, False)
        early = split_to_count(tokenizer, b_ids, flags, target, a_text, False)
        late = split_to_count(tokenizer, b_ids, flags, target, a_text, True)
        rows.append({
            "cohort": frozen["cohort"], "language": frozen["language"],
            "fact_id": frozen["fact_id"], "subject_qid": frozen["subject_qid"],
            "a_subject_tokens": frozen["a_subject_tokens"],
            "b_subject_tokens": frozen["b_subject_tokens"],
            "d_subject_tokens": target,
            "mid_subject_tokens": mid_target,
            "mid_ids_sha256": ids_sha(mid) if mid else None,
            "early_ids_sha256": ids_sha(early) if early else None,
            "late_ids_sha256": ids_sha(late) if late else None,
            "mid_input_ids": mid,
            "early_input_ids": early,
            "late_input_ids": late,
            "gold_answer_ids": frozen["gold_answer_ids"],
            "gold_answer": frozen["gold_answer"],
            "unmarked_prompt_sha256": hashlib.sha256(a_text.encode("utf-8")).hexdigest(),
            "early_late_same_ids": early == late if early and late else None,
        })
    result = {
        "method_status": "tokenizer_only_post_E6_feasibility_after_viewing_E6_results",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            (ROOT / "data/raw/aya_23_8b/tokenizer.json").read_bytes()).hexdigest(),
        "target": "Same visible unmarked text and exact marked-form subject-token count",
        "construction": "Iteratively split subject BPE token strings into existing token IDs, earliest and latest deterministic orders; accept only byte-identical full prompt decodes",
        "rows": rows,
        "early_eligible": sum(row["early_ids_sha256"] is not None for row in rows),
        "mid_eligible": sum(row["mid_ids_sha256"] is not None for row in rows),
        "late_eligible": sum(row["late_ids_sha256"] is not None for row in rows),
        "both_eligible_distinct": sum(row["early_ids_sha256"] is not None and
                                      row["late_ids_sha256"] is not None and
                                      not row["early_late_same_ids"] for row in rows),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("mid_eligible", "early_eligible", "late_eligible", "both_eligible_distinct")}),
          flush=True)


if __name__ == "__main__":
    main()

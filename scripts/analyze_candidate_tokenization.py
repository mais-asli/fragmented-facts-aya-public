"""Describe Aya tokenization of proposed pilot spelling pairs, without approving them.

This is a pre-run data diagnostic. It is not an Aya generation or accuracy result.
The tokenizer and chat template come from the exact local Aya-23-8B snapshot.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.prompts import paired_prompts
from fragmented_facts.tokenization import split_token_alternative
from fragmented_facts.unicode import validate_pair

DATA = ROOT / "data/review/ReviewerA_confirmed_20260926/reviewed_data.json"
WAVE = ROOT / "data/review/merged_20260925/first_wave_50.csv"
SNAPSHOT = ROOT / "model_snapshot.json"
OUT = ROOT / "outputs/01a09b51/candidate_tokenization"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    deltas = [r["delta_subject_tokens"] for r in rows]
    return {
        "prompt_pairs": len(rows),
        "distinct_facts": len({r["fact_id"] for r in rows}),
        "mean_u_subject_tokens": round(statistics.mean(r["u_subject_tokens"] for r in rows), 3),
        "mean_d_subject_tokens": round(statistics.mean(r["d_subject_tokens"] for r in rows), 3),
        "mean_delta": round(statistics.mean(deltas), 3),
        "median_delta": statistics.median(deltas),
        "d_more_tokens": sum(d > 0 for d in deltas),
        "same_tokens": sum(d == 0 for d in deltas),
        "d_fewer_tokens": sum(d < 0 for d in deltas),
        "boundary_crossing_u": sum(r["u_boundary_crossing"] for r in rows),
        "boundary_crossing_d": sum(r["d_boundary_crossing"] for r in rows),
        "last_subject_token_crossing_u": sum(r["u_last_token_crossing"] for r in rows),
        "last_subject_token_crossing_d": sum(r["d_last_token_crossing"] for r in rows),
    }


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    with WAVE.open(encoding="utf-8-sig", newline="") as stream:
        wave = list(csv.DictReader(stream))
    if len(wave) != 50 or len({r["Fact ID"] for r in wave}) != 50:
        raise ValueError("Expected 50 distinct first-wave facts")
    facts = {r["Fact ID"]: r for r in data["Pilot facts"]}
    pairs = {
        lang: {r["Fact ID"]: r for r in data[lang]}
        for lang in ("Hebrew", "Arabic")
    }
    templates = {
        r["Template key"]: r for r in data["Templates"]
        if r["Use"] == "evaluation"
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "data/raw/aya_23_8b"), local_files_only=True, use_fast=True
    )
    rows = []
    alt_coverage = {"he": {"u": 0, "d": 0, "both": 0}, "ar": {"u": 0, "d": 0, "both": 0}}
    for wave_row in wave:
        fact_id = wave_row["Fact ID"]
        fact = facts[fact_id]
        if fact["Decision"] == "reject":
            raise ValueError(f"Rejected fact entered first wave: {fact_id}")
        relation = fact_id.split("-")[1]
        for sheet, lang in (("Hebrew", "he"), ("Arabic", "ar")):
            pair = pairs[sheet][fact_id]
            if pair["Decision"] != "approved" or pair["Reviewer"] != "Reviewer A":
                raise ValueError(f"First-wave pair lacks ReviewerA's recorded check: {fact_id}/{lang}")
            u, d = validate_pair(pair["Unmarked spelling"], pair["Marked spelling"], lang)
            for tid in ("t1", "t2", "t3"):
                key = f"evaluation/{relation}/{lang}/{tid}"
                template = templates[key]["Question text"]
                ep_u, ep_d = paired_prompts(tokenizer, template, {"U": u, "D": d}, lang)
                if tid == "t1":
                    alt_u = split_token_alternative(tokenizer, ep_u) is not None
                    alt_d = split_token_alternative(tokenizer, ep_d) is not None
                    alt_coverage[lang]["u"] += alt_u
                    alt_coverage[lang]["d"] += alt_d
                    alt_coverage[lang]["both"] += alt_u and alt_d
                rows.append({
                    "fact_id": fact_id,
                    "subject_en": fact["Subject in English"],
                    "relation": relation,
                    "language": lang,
                    "template_key": key,
                    "fact_decision": fact["Decision"],
                    "pair_decision": pair["Decision"],
                    "template_decision": templates[key]["Decision"],
                    "u": u,
                    "d": d,
                    "u_subject_tokens": len(ep_u.subject_indices),
                    "d_subject_tokens": len(ep_d.subject_indices),
                    "delta_subject_tokens": len(ep_d.subject_indices) - len(ep_u.subject_indices),
                    "u_prompt_tokens": len(ep_u.input_ids),
                    "d_prompt_tokens": len(ep_d.input_ids),
                    "u_boundary_crossing": ep_u.boundary_crossing,
                    "d_boundary_crossing": ep_d.boundary_crossing,
                    "u_last_token_crossing": ep_u.offsets[ep_u.subject_indices[-1]][1] > ep_u.span[1],
                    "d_last_token_crossing": ep_d.offsets[ep_d.subject_indices[-1]][1] > ep_d.span[1],
                    "u_prompt_sha256": hashlib.sha256(ep_u.text.encode("utf-8")).hexdigest(),
                    "d_prompt_sha256": hashlib.sha256(ep_d.text.encode("utf-8")).hexdigest(),
                })
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "first_wave_50_prompt_tokenization.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    by_language = defaultdict(list)
    by_language_relation = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
        by_language_relation[f'{row["language"]}/{row["relation"]}'].append(row)
    report = {
        "status": "human_reviewed_name_pairs_facts_and_templates_pending_not_aya_accuracy",
        "model_id": snapshot["model_id"],
        "revision": snapshot["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "input_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (DATA, WAVE, SNAPSHOT)},
        "overall": summarize(rows),
        "by_language": {key: summarize(value) for key, value in by_language.items()},
        "by_language_relation": {key: summarize(value) for key, value in by_language_relation.items()},
        "same_string_plus_one_token_coverage_t1": alt_coverage,
        "output_csv": str(csv_path.relative_to(ROOT)),
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

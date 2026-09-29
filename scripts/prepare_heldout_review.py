"""Freeze an outcome-independent dev/test fact batch for possible extension."""

import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.data import apply_split_manifest, csv_rows, write_csv
from fragmented_facts.io import digest, file_hash, write_json

OUT = ROOT / "data/review/heldout_batch_v1"
if OUT.exists():
    raise FileExistsError("Preserve the first selected held-out review batch")

pool = apply_split_manifest(ROOT / "data/raw/candidates_final.jsonl",
                            ROOT / "data/review/subject_splits.json")
relations = ("P19", "P20", "P159", "P740")
quotas = {"dev": 5, "test": 15}
selected = []
for split, quota in quotas.items():
    for relation in relations:
        candidates = [row for row in pool
                      if row["split"] == split and row["relation"] == relation
                      and row["wikidata"]["single_answer_agrees"] is True]
        candidates.sort(key=lambda row: digest([17, "heldout-review-v1", split,
                                                relation, row["subject_qid"]]))
        if len(candidates) < quota:
            raise ValueError(f"Insufficient {split} {relation} candidates")
        selected.extend(candidates[:quota])
if len(selected) != 80 or len({row["subject_qid"] for row in selected}) != 80:
    raise ValueError("A subject was selected twice")

fact_source = {row["fact_id"]: row for row in csv_rows(ROOT / "data/review/facts.csv")}
pair_source = {(row["fact_id"], row["language"]): row
               for row in csv_rows(ROOT / "data/review/pairs.csv")}
facts = [fact_source[row["fact_id"]] for row in selected]
pairs = [pair_source[row["fact_id"], language]
         for row in selected for language in ("he", "ar")]
if any(row["decision"] != "pending" for row in facts + pairs):
    raise ValueError("Unexpected prior approval in held-out source rows")

OUT.mkdir(parents=True)
write_csv(OUT / "facts.csv", facts, list(facts[0]))
write_csv(OUT / "pairs.csv", pairs, list(pairs[0]))
manifest = {
    "schema_version": 1,
    "seed": 17,
    "purpose": "Independent, outcome-unseen review candidates; not approved study data",
    "selection": "Fixed original subject split; current Wikidata single-answer agreement; first hash-ranked 5 dev and 15 test per relation, without Aya outputs",
    "source_sha256": file_hash(ROOT / "data/raw/candidates_final.jsonl"),
    "split_manifest_sha256": file_hash(ROOT / "data/review/subject_splits.json"),
    "selected": [{"fact_id": row["fact_id"], "subject_qid": row["subject_qid"],
                  "split": row["split"], "relation": row["relation"]}
                 for row in selected],
}
manifest["manifest_hash"] = digest(manifest)
write_json(OUT / "manifest.json", manifest)
write_json(OUT / "candidate_source_rows.json", selected)
print(json.dumps({"directory": str(OUT), "facts": len(facts),
                  "pairs": len(pairs),
                  "split_relation_counts": dict(collections.Counter(
                      f"{row['split']}/{row['relation']}" for row in selected))}, indent=2))

"""Exploratory Aya same-visible-text, controlled-count tokenization grid."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fragmented_facts.io import ResultStore, digest, read_json, read_jsonl, write_json
from fragmented_facts.model import load_runner
from fragmented_facts.prompts import paired_prompts, template_for
from project_plan.feasibility.e6_position_decomposition_run_20260928 import conditions, score

PROTOCOL = ROOT / "configs/e7_same_count_protocol_20260928.json"
E6 = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
FEASIBILITY = ROOT / "results/e6_same_count_split_feasibility_20260928.json"
TEMPLATES = ROOT / "data/review/imported_ReviewerA_20260926/templates.json"
CONFIG = ROOT / "configs/study_independent_test_v1.json"
SNAPSHOT = ROOT / "model_snapshot.json"
FACTS = {
    "pilot": ROOT / "data/curated/pilot-reviewed-20260926.jsonl",
    "heldout": ROOT / "data/curated/aya-independent-test-v1.jsonl",
    "external_google_re": ROOT / "data/curated/aya-google-re-external-reviewed-20260927.jsonl",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validated_grid(tokenizer, fact, row, e6_row, templates):
    template = template_for(templates, fact["relation"], row["language"], "t1")
    a, d = paired_prompts(tokenizer, template, fact["pairs"][row["language"]], row["language"])
    e6_grid = conditions(tokenizer, fact, row["language"], template, e6_row)
    b_ids = e6_grid["B"][0]
    mark_positions = set(e6_row["mark_token_positions"])
    retained = [index for index in range(len(d.input_ids)) if index not in mark_positions]
    b_subject_positions = [new for new, old in enumerate(retained) if old in d.subject_indices]
    if not b_subject_positions or len(b_subject_positions) != row["b_subject_tokens"]:
        raise ValueError("Frozen B subject span differs")
    first, last = b_subject_positions[0], b_subject_positions[-1]
    targets = {"mid": row["mid_subject_tokens"],
               "early": row["d_subject_tokens"], "late": row["d_subject_tokens"]}
    answer_ids = e6_grid["B"][2]
    if (answer_ids != row["gold_answer_ids"] or
            row["gold_answer"] != e6_row["gold_answer"] or
            sha_text(a.text) != row["unmarked_prompt_sha256"]):
        raise ValueError("Frozen prompt or answer differs")
    result = {}
    for label, target in targets.items():
        ids = row[f"{label}_input_ids"]
        if (ids is None or ids_sha(ids) != row[f"{label}_ids_sha256"] or
                len(ids) != len(b_ids) + target - row["b_subject_tokens"] or
                ids[:first] != b_ids[:first] or
                ids[len(ids) - (len(b_ids) - last - 1):] != b_ids[last + 1:] or
                tokenizer.decode(ids, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False) != a.text):
            raise ValueError("Same-count candidate violates frozen text or token grid")
        result[label] = (ids, None, answer_ids)
    if result["early"][0] == result["late"][0]:
        raise ValueError("Early/late controls must differ in token IDs")
    return result


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True, choices=[0, 1])
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards != 2:
        raise ValueError("Frozen E7 uses two shards")
    expected_output = ROOT / f"results/e7-same-count-20260928-shard{args.shard}"
    output = (ROOT / args.output).resolve()
    if not args.dry_run and output != expected_output:
        raise ValueError("Unexpected E7 result path")
    protocol = read_json(PROTOCOL)
    e6 = read_json(E6)
    if (sha(E6) != protocol["e6_protocol_sha256"] or
            sha(FEASIBILITY) != protocol["feasibility_sha256"] or
            sha(ROOT / "data/raw/aya_23_8b/tokenizer.json") != protocol["tokenizer_json_sha256"] or
            len(protocol["rows"]) != 246 or
            protocol["model_id"] != "CohereLabs/aya-23-8B" or
            protocol["precision"] != "nf4"):
        raise ValueError("E7 frozen protocol identity changed")
    for relative, expected in protocol["file_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("E7 input differs: " + relative)
    from transformers import AutoTokenizer
    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(
            str(ROOT / "data/raw/aya_23_8b"), use_fast=True,
            local_files_only=True, trust_remote_code=False)
        runner = None
    else:
        runner = load_runner(SNAPSHOT, read_json(CONFIG))
        tokenizer = runner.tokenizer
        if (runner.identity["revision"] != protocol["revision"] or
                runner.identity["precision"] != protocol["precision"] or
                runner.identity["compute_dtype"] != protocol["compute_dtype"]):
            raise ValueError("Loaded Aya differs from frozen E7 setup")
    facts = {cohort: {fact["fact_id"]: fact for fact in read_jsonl(path)}
             for cohort, path in FACTS.items()}
    templates = read_json(TEMPLATES)
    e6_rows = {(r["cohort"], r["fact_id"], r["language"]): r for r in e6["rows"]}
    selected = [row for row in protocol["rows"]
                if int(digest(row["subject_qid"]), 16) % 2 == args.shard]
    if args.dry_run:
        for row in selected:
            fact = facts[row["cohort"]][row["fact_id"]]
            frozen = e6_rows[(row["cohort"], row["fact_id"], row["language"])]
            validated_grid(tokenizer, fact, row, frozen, templates)
        print(json.dumps({"status": "dry_run_valid", "shard": args.shard,
                          "rows": len(selected)}), flush=True)
        return
    manifest = {"schema_version": 1, "experiment": protocol["experiment"],
                "method_status": protocol["method_status"], "model": runner.identity,
                "protocol_sha256": sha(PROTOCOL), "shard": args.shard,
                "num_shards": 2, "data_kind": "research"}
    expected_keys = []
    completed = resumed = 0
    with ResultStore(output, manifest) as store:
        for row in selected:
            key = {"experiment": protocol["experiment"], "cohort": row["cohort"],
                   "fact_id": row["fact_id"], "language": row["language"],
                   "template": "t1"}
            expected_keys.append(digest(key))
            if store.get(key):
                resumed += 1
                continue
            fact = facts[row["cohort"]][row["fact_id"]]
            frozen = e6_rows[(row["cohort"], row["fact_id"], row["language"])]
            grid = validated_grid(tokenizer, fact, row, frozen, templates)
            scores = {label: score(runner, *grid[label]) for label in ("mid", "early", "late")}
            store.put(key, {"fact": {field: fact[field] for field in
                                     ("fact_id", "subject_qid", "object_qid", "relation", "split")},
                            "condition_scores": scores,
                            "sum_logprob": {label: value["sum_logprob"] for label, value in scores.items()},
                            "subject_tokens": {label: row[f"{label}_subject_tokens"]
                                               if label == "mid" else row["d_subject_tokens"]
                                               for label in scores},
                            "runtime": runner.runtime()})
            completed += 1
        summary = {"status": "complete", "experiment": protocol["experiment"],
                   "run_id": store.run_id, "run_path": str(store.path),
                   "completed_now": completed, "resumed_items": resumed,
                   "total_items": len(store.rows()), "expected_count": len(expected_keys),
                   "expected_keys": expected_keys}
        if completed + resumed != len(expected_keys) or len(store.rows()) != len(expected_keys):
            raise ValueError("Incomplete E7 grid")
        write_json(store.path / "completion.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "expected_keys"}), flush=True)


if __name__ == "__main__":
    main()

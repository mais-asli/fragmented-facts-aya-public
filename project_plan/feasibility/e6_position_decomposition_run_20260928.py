"""Run frozen post-hoc A/B/C/D likelihood decomposition with Aya-23-8B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragmented_facts.io import ResultStore, digest, read_json, read_jsonl, write_json  # noqa: E402
from fragmented_facts.model import load_runner  # noqa: E402
from fragmented_facts.prompts import paired_prompts, template_for  # noqa: E402
from fragmented_facts.scoring import continuation_ids, sequence_logprobs  # noqa: E402
from fragmented_facts.unicode import MARKS  # noqa: E402

PROTOCOL = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
FACTS = {
    "pilot": ROOT / "data/curated/pilot-reviewed-20260926.jsonl",
    "heldout": ROOT / "data/curated/aya-independent-test-v1.jsonl",
    "external_google_re": ROOT / "data/curated/aya-google-re-external-reviewed-20260927.jsonl",
}
TEMPLATES = ROOT / "data/review/imported_ReviewerA_20260926/templates.json"
CONFIG = ROOT / "configs/study_independent_test_v1.json"
SNAPSHOT = ROOT / "model_snapshot.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def conditions(tokenizer, fact: dict, language: str, template: str, frozen: dict) -> dict:
    a, d = paired_prompts(tokenizer, template, fact["pairs"][language], language)
    mark_set = set(frozen["mark_token_positions"])
    if not mark_set or not mark_set.issubset(set(d.subject_indices)):
        raise ValueError("Frozen mark IDs are missing or outside the subject")
    for index in mark_set:
        start, end = d.offsets[index]
        if not d.text[start:end] or not all(char in MARKS[language] for char in d.text[start:end]):
            raise ValueError("Frozen mark ID now overlaps a nonmark character")
    retained = [i for i in range(len(d.input_ids)) if i not in mark_set]
    b_ids = [d.input_ids[i] for i in retained]
    if (tokenizer.decode(b_ids, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False) != a.text or
            ids_sha(retained) != frozen["retained_d_positions_sha256"] or
            {"A": ids_sha(a.input_ids), "B": ids_sha(b_ids),
             "D": ids_sha(d.input_ids)} != frozen["input_ids_sha256"]):
        raise ValueError("Frozen A/B/C/D token sequences changed")
    if ([len(a.subject_indices), len(d.subject_indices) - len(mark_set),
         len(d.subject_indices)] !=
            [frozen["a_subject_tokens"], frozen["b_subject_tokens"],
             frozen["d_subject_tokens"]]):
        raise ValueError("Frozen subject token counts changed")
    answer = frozen["gold_answer"]
    a_answer = continuation_ids(tokenizer, a, answer)
    d_answer = continuation_ids(tokenizer, d, answer)
    if a_answer != d_answer or a_answer != frozen["gold_answer_ids"]:
        raise ValueError("Canonical continuation changed")
    if tokenizer.decode(b_ids + a_answer, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != a.text + answer:
        raise ValueError("B/C continuation does not reconstruct A visible text")
    c_positions = retained + list(range(len(d.input_ids), len(d.input_ids) + len(a_answer)))
    return {"A": (a.input_ids, None, a_answer),
            "B": (b_ids, None, a_answer),
            "C": (b_ids, c_positions, a_answer),
            "D": (d.input_ids, None, d_answer)}


def score(runner, prompt_ids: list[int], positions: list[int] | None,
          answer_ids: list[int]) -> dict:
    import torch

    full = prompt_ids + answer_ids
    if positions is not None:
        if len(positions) != len(full) or positions[-1] >= runner.context_limit:
            raise ValueError("Counterfactual position sequence is invalid")
        inputs = runner.inputs(full)
        position_ids = torch.tensor([positions], device=runner.device, dtype=torch.long)
        with torch.inference_mode():
            logits = runner.model(**inputs, position_ids=position_ids).logits
    else:
        logits = runner.logits(full)
    value = sequence_logprobs(logits, len(prompt_ids), answer_ids)
    del logits
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True, choices=[0, 1])
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards != 2:
        raise ValueError("Frozen E6 protocol uses two shards")
    output = (ROOT / args.output).resolve()
    expected_output = ROOT / f"results/e6-position-decomposition-20260928-shard{args.shard}"
    if not args.dry_run and output != expected_output:
        raise ValueError("Unexpected E6 output path")
    protocol = read_json(PROTOCOL)
    for relative, expected in protocol["file_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen E6 input differs: " + relative)
    if sha(ROOT / "data/raw/aya_23_8b/tokenizer.json") != protocol["tokenizer_json_sha256"]:
        raise ValueError("Frozen Aya tokenizer differs")
    if (protocol["model_id"] != "CohereLabs/aya-23-8B" or
            protocol["precision"] != "nf4" or
            len(protocol["rows"]) != 246):
        raise ValueError("E6 protocol identity changed")
    facts = {cohort: {f["fact_id"]: f for f in read_jsonl(path)}
             for cohort, path in FACTS.items()}
    templates = read_json(TEMPLATES)
    config = read_json(CONFIG)
    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(ROOT / "data/raw/aya_23_8b"), local_files_only=True,
            use_fast=True, trust_remote_code=False)
        count = 0
        for row in protocol["rows"]:
            if int(digest(row["subject_qid"]), 16) % 2 != args.shard:
                continue
            fact = facts[row["cohort"]][row["fact_id"]]
            template = template_for(templates, fact["relation"], row["language"], "t1")
            conditions(tokenizer, fact, row["language"], template, row)
            count += 1
        print(json.dumps({"status": "dry_run_valid", "shard": args.shard,
                          "rows": count}), flush=True)
        return
    runner = load_runner(SNAPSHOT, config)
    if (runner.identity["revision"] != protocol["revision"] or
            runner.identity["precision"] != protocol["precision"] or
            runner.identity["compute_dtype"] != protocol["compute_dtype"]):
        raise ValueError("Loaded Aya configuration differs from frozen protocol")
    manifest = {"schema_version": 1, "experiment": protocol["experiment"],
                "method_status": protocol["method_status"], "model": runner.identity,
                "protocol_sha256": sha(PROTOCOL), "shard": args.shard,
                "num_shards": 2, "data_kind": "research"}
    expected_keys = []
    completed = resumed = 0
    with ResultStore(output, manifest) as store:
        for row in protocol["rows"]:
            if int(digest(row["subject_qid"]), 16) % 2 != args.shard:
                continue
            key = {"experiment": protocol["experiment"], "cohort": row["cohort"],
                   "fact_id": row["fact_id"], "language": row["language"],
                   "template": "t1"}
            expected_keys.append(digest(key))
            if store.get(key):
                resumed += 1
                continue
            fact = facts[row["cohort"]][row["fact_id"]]
            if (fact["subject_qid"] != row["subject_qid"] or
                    fact["relation"] != row["relation"]):
                raise ValueError("E6 fact identity changed")
            template = template_for(templates, fact["relation"], row["language"], "t1")
            grid = conditions(runner.tokenizer, fact, row["language"], template, row)
            scores = {label: score(runner, *grid[label]) for label in ("A", "B", "C", "D")}
            for label in ("A", "D"):
                deviation = abs(scores[label]["sum_logprob"] - row["e2_gold_sum"][label])
                if deviation > protocol["baseline_reproducibility_tolerance_nats"]:
                    raise ValueError(f"Aya E2 {label} baseline mismatch {row['fact_id']} "
                                     f"{row['language']}: {deviation:.5f} nats")
            sums = {label: scores[label]["sum_logprob"] for label in scores}
            store.put(key, {"fact": {field: fact[field] for field in
                                     ("fact_id", "subject_qid", "object_qid", "relation", "split")},
                            "condition_scores": scores, "sum_logprob": sums,
                            "components": {"A_to_B": sums["B"] - sums["A"],
                                           "B_to_C": sums["C"] - sums["B"],
                                           "C_to_D": sums["D"] - sums["C"],
                                           "A_to_D": sums["D"] - sums["A"]},
                            "e2_baseline": row["e2_gold_sum"],
                            "subject_tokens": {label: row[f"{label.lower()}_subject_tokens"]
                                               for label in ("A", "B", "D")},
                            "mark_token_positions": row["mark_token_positions"],
                            "runtime": runner.runtime()})
            completed += 1
        summary = {"status": "complete", "experiment": protocol["experiment"],
                   "run_id": store.run_id, "run_path": str(store.path),
                   "completed_now": completed, "resumed_items": resumed,
                   "total_items": len(store.rows()), "expected_count": len(expected_keys),
                   "expected_keys": expected_keys}
        if completed + resumed != len(expected_keys) or len(store.rows()) != len(expected_keys):
            raise ValueError("Incomplete E6 result grid")
        write_json(store.path / "completion.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "expected_keys"}), flush=True)


if __name__ == "__main__":
    main()

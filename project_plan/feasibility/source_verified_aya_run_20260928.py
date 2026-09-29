"""Aya-only prospective U/D runs on source-checked external name/fact cohorts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragmented_facts.io import ResultStore, digest, read_json, read_jsonl, write_json
from fragmented_facts.model import load_runner
from fragmented_facts.prompts import encode_prompt, paired_prompts
from fragmented_facts.scoring import continuation_ids

PROTOCOL = ROOT / "configs/aya_source_verified_external_protocol_20260928.json"
SNAPSHOT = ROOT / "model_snapshot.json"
CONFIG = ROOT / "configs/study_independent_test_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def setup(tokenizer, cohort: str, fact: dict, frozen: dict, settings: dict) -> dict:
    language = settings["language"]
    subject = {"U": fact[f"subject_{language}_U"],
               "D": fact[f"subject_{language}_D"]}
    gold, bad = fact[f"answer_{language}"], fact[f"distractor_answer_{language}"]
    prompts = {}
    for template_id, template in settings["templates"].items():
        u, d = paired_prompts(tokenizer, template, subject, language)
        prompts[template_id] = {"U": u, "D": d}
        for variant, prompt in prompts[template_id].items():
            key = f"{template_id}_{variant}"
            if ids_sha(prompt.input_ids) != frozen["prompt_ids_sha256"][key]:
                raise ValueError("Frozen Aya prompt IDs changed: " + fact["fact_id"] + " " + key)
            for answer_name, answer in (("gold", gold), ("distractor", bad)):
                ids = continuation_ids(tokenizer, prompt, answer)
                if ids_sha(ids) != frozen["answer_ids_sha256"][answer_name]:
                    raise ValueError("Frozen answer token IDs changed: " + fact["fact_id"])
    english = None
    if cohort == "arabic_geonames":
        english = encode_prompt(tokenizer, settings["english_control_template"],
                                fact["subject_en"], "en")
        if ids_sha(english.input_ids) != frozen["prompt_ids_sha256"]["english_control"]:
            raise ValueError("Frozen English control prompt changed")
    return {"prompts": prompts, "english": english}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("arabic_geonames", "hebrew_cbs"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards != 2:
        raise ValueError("Frozen protocol has exactly two shards")
    protocol = read_json(PROTOCOL)
    for relative, expected in protocol["file_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen source/code hash changed: " + relative)
    if sha(ROOT / "data/raw/aya_23_8b/tokenizer.json") != protocol["tokenizer_json_sha256"]:
        raise ValueError("Frozen tokenizer differs")
    settings = protocol["cohorts"][args.cohort]
    facts = {fact["fact_id"]: fact for fact in read_jsonl(ROOT / settings["data_path"])}
    if len(facts) != settings["source_rows"]:
        raise ValueError("Cohort source has duplicate or missing facts")
    rows = [row for row in settings["eligible_rows"]
            if int(digest(row["subject_qid"]), 16) % 2 == args.shard]
    if args.dry_run:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "data/raw/aya_23_8b"),
                                                  use_fast=True, local_files_only=True,
                                                  trust_remote_code=False)
        for row in rows:
            setup(tokenizer, args.cohort, facts[row["fact_id"]], row, settings)
        print(json.dumps({"status": "dry_run_valid", "cohort": args.cohort,
                          "shard": args.shard, "rows": len(rows)}), flush=True)
        return
    expected_output = ROOT / f"results/aya-source-verified-20260928-{args.cohort}-shard{args.shard}"
    if (ROOT / args.output).resolve() != expected_output.resolve():
        raise ValueError("Unexpected Aya output path")
    runner = load_runner(SNAPSHOT, read_json(CONFIG))
    if (runner.identity["revision"] != protocol["revision"] or
            runner.identity["precision"] != protocol["precision"] or
            runner.identity["compute_dtype"] != protocol["compute_dtype"]):
        raise ValueError("Aya runtime configuration differs from frozen protocol")
    manifest = {"schema_version": 1, "experiment": "source_verified_external_UD_20260928",
                "protocol_sha256": sha(PROTOCOL), "cohort": args.cohort,
                "shard": args.shard, "num_shards": 2,
                "model": runner.identity, "data_kind": "research"}
    expected = []
    completed = resumed = 0
    with ResultStore(expected_output, manifest) as store:
        for row in rows:
            key = {"experiment": manifest["experiment"], "cohort": args.cohort,
                   "fact_id": row["fact_id"], "shard": args.shard}
            expected.append(digest(key))
            if store.get(key):
                resumed += 1
                continue
            fact = facts[row["fact_id"]]
            if fact["subject_qid"] != row["subject_qid"]:
                raise ValueError("Frozen subject identity differs")
            pack = setup(runner.tokenizer, args.cohort, fact, row, settings)
            gold, bad = (fact[f"answer_{settings['language']}"],
                         fact[f"distractor_answer_{settings['language']}"])
            scores = {}
            for template_id, pair in pack["prompts"].items():
                scores[template_id] = {}
                for variant, prompt in pair.items():
                    gold_score = runner.score(prompt, gold)
                    bad_score = runner.score(prompt, bad)
                    scores[template_id][variant] = {
                        "gold": gold_score, "distractor": bad_score,
                        "gold_minus_distractor": (gold_score["sum_logprob"] -
                                                  bad_score["sum_logprob"]),
                        "subject_tokens": len(prompt.subject_indices),
                        "full_prompt_tokens": len(prompt.input_ids),
                        "subject_boundary_crossing": prompt.boundary_crossing,
                    }
            generations = {variant: runner.generate(pack["prompts"]["t1"][variant])
                           for variant in ("U", "D")}
            english = runner.generate(pack["english"]) if pack["english"] else None
            store.put(key, {
                "fact": fact, "scores": scores, "generations_t1": generations,
                "english_control_generation": english, "runtime": runner.runtime(),
            })
            completed += 1
            if completed % 10 == 0:
                print(json.dumps({"event": "progress", "cohort": args.cohort,
                                  "shard": args.shard, "completed_now": completed,
                                  "total_in_shard": len(rows)}), flush=True)
        if len(store.rows()) != len(rows) or completed + resumed != len(rows):
            raise ValueError("Incomplete Aya source-verified result grid")
        summary = {"status": "complete", "experiment": manifest["experiment"],
                   "cohort": args.cohort, "shard": args.shard,
                   "run_id": store.run_id, "run_path": str(store.path),
                   "completed_now": completed, "resumed_items": resumed,
                   "total_items": len(rows), "expected_count": len(rows),
                   "expected_keys": expected}
        write_json(store.path / "completion.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "expected_keys"}), flush=True)


if __name__ == "__main__":
    main()

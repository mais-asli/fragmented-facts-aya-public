"""Aya source-disjoint E5 likelihood-only residual-patching replication."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fragmented_facts.hooks import Patch
from fragmented_facts.io import ResultStore, digest, read_json, read_jsonl, write_json
from fragmented_facts.model import load_runner
from fragmented_facts.prompts import encode_prompt, paired_prompts
from fragmented_facts.scoring import continuation_ids

PROTOCOL = ROOT / "configs/aya_source_verified_e5_protocol_20260928.json"
SOURCE = ROOT / "configs/aya_source_verified_external_protocol_20260928.json"
SNAPSHOT = ROOT / "model_snapshot.json"
CONFIG = ROOT / "configs/study_independent_test_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def prepare(tokenizer, row: dict, facts: dict, setting: dict, layers: list[int]) -> dict:
    language = setting["language"]
    target, donor = facts[row["fact_id"]], facts[row["donor_fact_id"]]
    if (target["subject_qid"] != row["subject_qid"] or
            donor["subject_qid"] != row["donor_subject_qid"] or
            row["answer_stratum"] == row["donor_answer_stratum"]):
        raise ValueError("Frozen E5 target/donor identity differs")
    template = setting["template_t1"]
    u, d = paired_prompts(tokenizer, template,
                          {"U": target[f"subject_{language}_U"],
                           "D": target[f"subject_{language}_D"]}, language)
    donor_u = encode_prompt(tokenizer, template, donor[f"subject_{language}_U"], language)
    if ({"U": ids_sha(u.input_ids), "D": ids_sha(d.input_ids)} !=
            row["target_prompt_ids_sha256"] or
            ids_sha(donor_u.input_ids) != row["donor_prompt_ids_sha256"]):
        raise ValueError("Frozen E5 prompt token IDs differ")
    gold, bad = target[f"answer_{language}"], target[f"distractor_answer_{language}"]
    if (ids_sha(continuation_ids(tokenizer, d, gold)) != row["gold_answer_ids_sha256"] or
            ids_sha(continuation_ids(tokenizer, d, bad)) != row["distractor_answer_ids_sha256"] or
            continuation_ids(tokenizer, u, gold) != continuation_ids(tokenizer, d, gold) or
            continuation_ids(tokenizer, u, bad) != continuation_ids(tokenizer, d, bad)):
        raise ValueError("Frozen E5 answer token IDs differ")
    if layers != [12, 13]:
        raise ValueError("Unexpected E5 layer window")
    return {"U": u, "D": d, "donor_U": donor_u, "gold": gold, "bad": bad,
            "fact": target, "donor": donor}


def score_pair(runner, prompt, gold: str, bad: str, patches=()) -> dict:
    gold_score = runner.score(prompt, gold, patches)
    bad_score = runner.score(prompt, bad, patches)
    return {"gold": gold_score, "distractor": bad_score,
            "gold_minus_distractor": gold_score["sum_logprob"] - bad_score["sum_logprob"]}


def patches(vectors: dict, source_prompt, target_prompt, site: str, layers: list[int]) -> list[Patch]:
    return [Patch(layer, "residual", target_prompt.position(site),
                  vectors[(layer, "residual", source_prompt.position(site))])
            for layer in layers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=("arabic_geonames", "hebrew_cbs"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    source = read_json(SOURCE)
    if sha(SOURCE) != protocol["source_protocol_sha256"] or \
            sha(ROOT / "data/raw/aya_23_8b/tokenizer.json") != protocol["tokenizer_json_sha256"]:
        raise ValueError("Frozen source or tokenizer differs")
    if protocol["layers"] != [12, 13] or protocol["site"] != "last_subject":
        raise ValueError("Aya E5 localization changed")
    setting = protocol["cohorts"][args.cohort]
    if sha(ROOT / setting["data_path"]) != setting["source_data_sha256"]:
        raise ValueError("Frozen E5 data changed")
    if setting["selected_size"] != 96 or len(setting["rows"]) != 96:
        raise ValueError("Unexpected E5 sample size")
    facts = {fact["fact_id"]: fact for fact in read_jsonl(ROOT / setting["data_path"])}
    if len(facts) != source["cohorts"][args.cohort]["source_rows"]:
        raise ValueError("E5 source data incomplete")
    if args.dry_run:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "data/raw/aya_23_8b"),
                                                  use_fast=True, local_files_only=True,
                                                  trust_remote_code=False)
        for row in setting["rows"]:
            prepare(tokenizer, row, facts, setting, protocol["layers"])
        print(json.dumps({"status": "dry_run_valid", "cohort": args.cohort,
                          "rows": len(setting["rows"])}), flush=True)
        return
    output = (ROOT / args.output).resolve()
    expected_output = ROOT / f"results/aya-source-verified-e5-20260928-{args.cohort}"
    if output != expected_output.resolve():
        raise ValueError("Unexpected E5 output path")
    runner = load_runner(SNAPSHOT, read_json(CONFIG))
    if (runner.identity["revision"] != protocol["revision"] or
            runner.identity["precision"] != protocol["precision"] or
            runner.identity["compute_dtype"] != protocol["compute_dtype"]):
        raise ValueError("Aya runtime differs from frozen E5 identity")
    manifest = {"schema_version": 1, "experiment": protocol["experiment"],
                "method_status": protocol["method_status"], "cohort": args.cohort,
                "model": runner.identity, "protocol_sha256": sha(PROTOCOL),
                "runner_sha256": sha(Path(__file__)), "data_kind": "research"}
    expected_keys = []
    completed = resumed = 0
    with ResultStore(output, manifest) as store:
        for row in setting["rows"]:
            key = {"experiment": protocol["experiment"], "cohort": args.cohort,
                   "fact_id": row["fact_id"]}
            expected_keys.append(digest(key))
            if store.get(key):
                resumed += 1
                continue
            pack = prepare(runner.tokenizer, row, facts, setting, protocol["layers"])
            u, d, donor_u = pack["U"], pack["D"], pack["donor_U"]
            gold, bad = pack["gold"], pack["bad"]
            request_u = sorted({(layer, "residual", u.position(site))
                                for layer in protocol["layers"]
                                for site in ("last_subject", "first_subject")})
            vectors_u = runner.capture(u, request_u)
            vectors_d = runner.capture(d, [(layer, "residual", d.position("last_subject"))
                                           for layer in protocol["layers"]])
            vectors_unrelated = runner.capture(donor_u, [
                (layer, "residual", donor_u.position("last_subject"))
                for layer in protocol["layers"]])
            scores = {
                "baseline_U": score_pair(runner, u, gold, bad),
                "baseline_D": score_pair(runner, d, gold, bad),
                "same_entity_U_to_D": score_pair(runner, d, gold, bad,
                    patches(vectors_u, u, d, "last_subject", protocol["layers"])),
                "unrelated_U_to_D": score_pair(runner, d, gold, bad,
                    patches(vectors_unrelated, donor_u, d, "last_subject", protocol["layers"])),
                "identity_D_to_D": score_pair(runner, d, gold, bad,
                    patches(vectors_d, d, d, "last_subject", protocol["layers"])),
                "first_subject_U_to_D": score_pair(runner, d, gold, bad,
                    patches(vectors_u, u, d, "first_subject", protocol["layers"])),
            }
            for answer in ("gold", "distractor"):
                deviation = abs(scores["identity_D_to_D"][answer]["sum_logprob"] -
                                scores["baseline_D"][answer]["sum_logprob"])
                if deviation > 0.1:
                    raise ValueError(f"Aya E5 identity-patch check failed: {row['fact_id']} {deviation:.3f}")
            store.put(key, {"fact": {"fact_id": row["fact_id"],
                                     "subject_qid": row["subject_qid"],
                                     "answer_stratum": row["answer_stratum"],
                                     "donor_fact_id": row["donor_fact_id"],
                                     "donor_answer_stratum": row["donor_answer_stratum"]},
                            "scores": scores,
                            "subject_tokens": {"U": len(u.subject_indices),
                                               "D": len(d.subject_indices)},
                            "runtime": runner.runtime()})
            completed += 1
            if completed % 10 == 0:
                print(json.dumps({"event": "progress", "cohort": args.cohort,
                                  "completed_now": completed, "of": len(setting["rows"])}), flush=True)
        if len(store.rows()) != len(setting["rows"]) or completed + resumed != len(setting["rows"]):
            raise ValueError("Incomplete source E5 result grid")
        summary = {"status": "complete", "experiment": protocol["experiment"],
                   "cohort": args.cohort, "total_items": len(store.rows()),
                   "expected_count": len(setting["rows"]), "expected_keys": expected_keys,
                   "completed_now": completed, "resumed_items": resumed,
                   "run_id": store.run_id, "run_path": str(store.path)}
        write_json(store.path / "completion.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "expected_keys"}), flush=True)


if __name__ == "__main__":
    main()

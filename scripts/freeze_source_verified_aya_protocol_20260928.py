"""Freeze tokenizer-eligible Aya prompts for the two new sourced cohorts."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fragmented_facts.io import read_jsonl
from fragmented_facts.prompts import encode_prompt, paired_prompts
from fragmented_facts.scoring import continuation_ids

OUT = ROOT / "configs/aya_source_verified_external_protocol_20260928.json"
MODEL_PATH = ROOT / "data/raw/aya_23_8b"

COHORTS = {
    "arabic_geonames": {
        "data_path": "data/curated/aya-external-geonames-arabic-20260928.jsonl",
        "manifest_path": "configs/aya_external_geonames_arabic_cohort_20260928.json",
        "language": "ar",
        "templates": {
            "t1": "في أي دولة تقع {subject}؟ أجب باسم الدولة فقط.",
            "t2": "ما الدولة التي تقع فيها {subject}؟ اكتب اسم الدولة فقط.",
        },
        "english_control_template": "Which country is {subject} located in? Give only the country's name.",
    },
    "hebrew_cbs": {
        "data_path": "data/curated/aya-external-cbs-hebrew-20260928.jsonl",
        "manifest_path": "configs/aya_external_cbs_hebrew_cohort_20260928.json",
        "language": "he",
        "templates": {
            "t1": "באיזו נפה נמצא היישוב {subject}? השב בשם הנפה בלבד.",
            "t2": "מה שם הנפה שבה נמצא {subject}? כתוב רק את שם הנפה.",
        },
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True,
                                              local_files_only=True, trust_remote_code=False)
    cohorts = {}
    for name, source in COHORTS.items():
        facts = read_jsonl(ROOT / source["data_path"])
        manifest = json.loads((ROOT / source["manifest_path"]).read_text(encoding="utf-8"))
        if len(facts) != manifest["size"] or sha(ROOT / source["data_path"]) != manifest["cohort_sha256"]:
            raise ValueError("Source cohort manifest differs: " + name)
        frozen = {**source, "source_rows": len(facts), "eligible_rows": [],
                  "excluded_rows": []}
        exclusions = Counter()
        for fact in facts:
            language = source["language"]
            pair = {"U": fact[f"subject_{language}_U"],
                    "D": fact[f"subject_{language}_D"]}
            gold = fact[f"answer_{language}"]
            bad = fact[f"distractor_answer_{language}"]
            if gold == bad:
                raise ValueError("Distractor equals gold")
            prompt_hashes = {}
            answer_hashes = {}
            token_counts = {}
            try:
                for template_id, template in source["templates"].items():
                    u, d = paired_prompts(tokenizer, template, pair, language)
                    for variant, prompt in (("U", u), ("D", d)):
                        prompt_hashes[f"{template_id}_{variant}"] = ids_sha(prompt.input_ids)
                        token_counts[f"{template_id}_{variant}"] = len(prompt.subject_indices)
                        for label, answer in (("gold", gold), ("distractor", bad)):
                            encoded = continuation_ids(tokenizer, prompt, answer)
                            value = ids_sha(encoded)
                            if label in answer_hashes and answer_hashes[label] != value:
                                raise ValueError("answer_ids_differ_across_conditions")
                            answer_hashes[label] = value
                if name == "arabic_geonames":
                    english = encode_prompt(tokenizer, source["english_control_template"],
                                            fact["subject_en"], "en")
                    prompt_hashes["english_control"] = ids_sha(english.input_ids)
            except (ValueError, TypeError) as error:
                exclusions[type(error).__name__ + ":" + str(error)[:80]] += 1
                frozen["excluded_rows"].append({"fact_id": fact["fact_id"],
                                                 "reason": str(error)})
                continue
            frozen["eligible_rows"].append({
                "fact_id": fact["fact_id"], "subject_qid": fact["subject_qid"],
                "prompt_ids_sha256": prompt_hashes,
                "answer_ids_sha256": answer_hashes,
                "subject_token_counts": token_counts,
            })
        frozen["tokenizer_exclusions"] = dict(exclusions)
        frozen["eligible_rows"].sort(key=lambda row: row["fact_id"])
        cohorts[name] = frozen
    paths = [
        *(ROOT / c["data_path"] for c in COHORTS.values()),
        *(ROOT / c["manifest_path"] for c in COHORTS.values()),
        ROOT / "configs/study_independent_test_v1.json",
        ROOT / "project_plan/feasibility/source_verified_aya_run_20260928.py",
        ROOT / "src/fragmented_facts/model.py",
        ROOT / "src/fragmented_facts/prompts.py",
        ROOT / "src/fragmented_facts/scoring.py",
        ROOT / "src/fragmented_facts/unicode.py",
        ROOT / "src/fragmented_facts/io.py",
    ]
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Aya_prospective_source_verified_external_UD",
        "model_id": "CohereLabs/aya-23-8B",
        "revision": "89da1a0ed02d6130f93ae0ffdbedb63b760c0471",
        "precision": "nf4", "compute_dtype": "fp16",
        "method_status": "prospective_for_new_cohorts_exploratory_relative_to_prior_study",
        "primary_outcome": "t1 paired change in gold-minus-frozen-distractor total log likelihood, marked minus unmarked, analyzed by subject with answer-stratum macro weighting",
        "secondary_outcomes": ["t2 replication of paired margin change",
                               "gold-answer total log-likelihood change",
                               "greedy t1 exact and source-alias answer accuracy with prefix-rematch sensitivity",
                               "English unmarked recall control for Arabic cities"],
        "interpretation_limits": ["The manipulation still changes marks, segmentation, token identities and length together.",
                                  "This extension has different relation/domain distributions in Arabic and Hebrew.",
                                  "Source cross-checks are not independent human pronunciation review."],
        "tokenizer_json_sha256": sha(MODEL_PATH / "tokenizer.json"),
        "file_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths},
        "cohorts": cohorts,
    }
    OUT.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"source": c["source_rows"], "eligible": len(c["eligible_rows"]),
                             "tokenizer_exclusions": c["tokenizer_exclusions"]}
                      for name, c in cohorts.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

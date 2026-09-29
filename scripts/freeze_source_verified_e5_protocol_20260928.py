"""Freeze a source-disjoint Aya activation-patching replication before its outputs."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = ROOT / "configs/aya_source_verified_external_protocol_20260928.json"
OUT = ROOT / "configs/aya_source_verified_e5_protocol_20260928.json"
SALT = "aya-source-verified-e5-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(*parts: str) -> str:
    return hashlib.sha256("|".join((SALT, *parts)).encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Source E5 protocol is already frozen; do not overwrite it")
    source = read(SOURCE_PROTOCOL)
    result = {
        "experiment": "Aya_source_disjoint_E5_activation_patch_replication",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_status": "prospective_for_this_new_source_cohort_after_prior_pilot_and_heldout_E5",
        "model_id": source["model_id"], "revision": source["revision"],
        "precision": source["precision"], "compute_dtype": source["compute_dtype"],
        "source_protocol_sha256": sha(SOURCE_PROTOCOL),
        "tokenizer_json_sha256": source["tokenizer_json_sha256"],
        "site": "last_subject", "component": "residual", "layers": [12, 13],
        "site_selection": "Fixed from earlier pilot-selected and 40-subject held-out Aya E5; not reselected on the new cohorts.",
        "conditions": ["baseline_D", "same_entity_U_to_D", "unrelated_U_to_D", "identity_D_to_D", "first_subject_U_to_D"],
        "primary_outcome": "Within each cohort, source-stratum-macro paired difference in gold-minus-fixed-distractor log-likelihood margin, same-entity U-to-D patch minus unpatched D, t1 prompt.",
        "primary_controls": ["unrelated_U_to_D minus baseline_D", "identity_D_to_D minus baseline_D"],
        "secondary_controls": ["first_subject_U_to_D minus baseline_D", "gold-answer-only log-likelihood gain"],
        "inference_unit": "subject; answer stratum is country ISO2 for Arabic and official subdistrict for Hebrew; hierarchical answer-stratum bootstrap and sign flips, Holm over the two language/domain primaries",
        "no_outcome_selection": "Every row is chosen by a deterministic source-stratified hash rule before new E5 outputs, independent of Aya source-cohort E2 outcomes.",
        "claim_limits": ["This is a new relation and source distribution, not a matched Arabic/Hebrew comparison.",
                         "Whole-residual patching does not identify a unique circuit or isolate token count.",
                         "The layer was selected using earlier E5 data and tested once on the prior held-out cohort."],
        "cohorts": {},
    }
    for cohort, setting in source["cohorts"].items():
        eligible = setting["eligible_rows"]
        if cohort == "arabic_geonames":
            stratum = lambda fact: fact["country_iso2"]
        elif cohort == "hebrew_cbs":
            stratum = lambda fact: fact["answer_he"]
        else:
            raise ValueError("Unexpected source cohort")
        facts = {json.loads(line)["fact_id"]: json.loads(line)
                 for line in (ROOT / setting["data_path"]).read_text(encoding="utf-8").splitlines()
                 if line.strip()}
        by_stratum = defaultdict(list)
        for row in eligible:
            by_stratum[stratum(facts[row["fact_id"]])].append(row)
        for group in by_stratum.values():
            group.sort(key=lambda row: rank("select", row["subject_qid"]))
        selected = []
        while len(selected) < min(96, len(eligible)):
            progress = False
            for name in sorted(by_stratum):
                if by_stratum[name] and len(selected) < 96:
                    selected.append(by_stratum[name].pop(0))
                    progress = True
            if not progress:
                break
        if len(selected) != min(96, len(eligible)):
            raise ValueError("Source E5 hash selection incomplete")
        mapped = []
        for row in selected:
            fact = facts[row["fact_id"]]
            other = [candidate for candidate in eligible
                     if candidate["subject_qid"] != row["subject_qid"] and
                     stratum(facts[candidate["fact_id"]]) != stratum(fact)]
            if not other:
                raise ValueError("No unrelated different-answer donor")
            donor = min(other, key=lambda candidate: (
                abs(candidate["subject_token_counts"]["t1_U"] -
                    row["subject_token_counts"]["t1_U"]),
                rank("donor", row["subject_qid"], candidate["subject_qid"])))
            mapped.append({
                "fact_id": row["fact_id"], "subject_qid": row["subject_qid"],
                "answer_stratum": stratum(fact),
                "donor_fact_id": donor["fact_id"], "donor_subject_qid": donor["subject_qid"],
                "donor_answer_stratum": stratum(facts[donor["fact_id"]]),
                "target_prompt_ids_sha256": {variant: row["prompt_ids_sha256"][f"t1_{variant}"]
                                             for variant in ("U", "D")},
                "donor_prompt_ids_sha256": donor["prompt_ids_sha256"]["t1_U"],
                "gold_answer_ids_sha256": row["answer_ids_sha256"]["gold"],
                "distractor_answer_ids_sha256": row["answer_ids_sha256"]["distractor"],
            })
        mapped.sort(key=lambda row: row["fact_id"])
        if len({row["subject_qid"] for row in mapped}) != len(mapped):
            raise ValueError("Repeated source E5 subject")
        result["cohorts"][cohort] = {
            "data_path": setting["data_path"], "language": setting["language"],
            "template_t1": setting["templates"]["t1"],
            "source_data_sha256": sha(ROOT / setting["data_path"]),
            "eligible_population_size": len(eligible),
            "selected_size": len(mapped),
            "distinct_answer_strata": len({row["answer_stratum"] for row in mapped}),
            "selection": "Deterministic SHA-256 round-robin by answer stratum; at most 96 subjects",
            "donor": "Different answer stratum in same cohort; closest frozen unmarked t1 subject-token count, SHA-256 tie break",
            "rows": mapped,
        }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"subjects": value["selected_size"],
                             "answer_strata": value["distinct_answer_strata"]}
                      for name, value in result["cohorts"].items()}), flush=True)


if __name__ == "__main__":
    main()

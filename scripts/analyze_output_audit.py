"""Compare ReviewerA's confirmed Aya-output judgments to the frozen scorer."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs/01a09b51/aya_output_audit_20260926"
SHEET = FOLDER / "ReviewerA_output_review_confirmed_20260926.csv"
MAP = FOLDER / "private_sampling_map.json"
OUT = ROOT / "results/ReviewerA-output-audit-analysis-20260926.json"
SEMANTIC = FOLDER / "semantic_completion_adjudication_20260926.json"


def kappa(confusion):
    n = sum(confusion.values())
    if n == 0:
        return None
    agreement = (confusion["yes/yes"] + confusion["no/no"]) / n
    human_yes = (confusion["yes/yes"] + confusion["yes/no"]) / n
    machine_yes = (confusion["yes/yes"] + confusion["no/yes"]) / n
    expected = human_yes * machine_yes + (1 - human_yes) * (1 - machine_yes)
    return None if expected == 1 else (agreement - expected) / (1 - expected)


def main():
    mapping = json.loads(MAP.read_text(encoding="utf-8"))
    index = {row["audit_id"]: row for row in mapping}
    if len(index) != 80:
        raise ValueError("Expected 80 distinct sampled outputs")
    with SHEET.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    semantic_overrides = json.loads(SEMANTIC.read_text(encoding="utf-8"))["rows"]
    if len(rows) != 80 or len({row["Audit ID"] for row in rows}) != 80:
        raise ValueError("Review sheet IDs or row count changed")
    report = {"scope": "Fixed stratified Aya output sample; human labels are not used to change primary scoring",
              "sampled": len(rows),
              "distinct_facts_sampled": len({item["key"]["fact_id"] for item in mapping}),
              "reviewed": 0, "by_language": [], "disagreements": [],
              "entity_agreement_count": 0,
              "joint_entity_language_agreement_count": 0,
              "conditional_language_agreement_count": 0,
              "conditional_language_denominator": 0,
              "human_standalone_language_yes_count": 0,
              "mechanical_cap_vs_human_semantic_truncation": {}}
    reviewed_fact_ids = set()
    cap_semantic = Counter()
    for language in ("en", "he", "ar"):
        subset = [row for row in rows if row["Language"] == language]
        expected_size = {"en": 20, "he": 30, "ar": 30}[language]
        if len(subset) != expected_size:
            raise ValueError("Language sample count changed")
        confusion = Counter()
        joint_confusion = Counter()
        conditional_language = Counter()
        category_counts = Counter()
        stratum_n = Counter()
        stratum_disagreement = Counter()
        stratum_population = {}
        complete = 0
        distinct_facts = set()
        for row in subset:
            audit_id = row["Audit ID"]
            if audit_id not in index or index[audit_id]["key"]["language"] != language:
                raise ValueError("Review row does not match the private sample map")
            human_entity = row["ReviewerA: entity correct (yes/no)"].strip().lower()
            human_language = row["ReviewerA: requested language correct (yes/no)"].strip().lower()
            reviewer = row["Reviewer"].strip()
            if not human_entity and not human_language and not reviewer:
                continue
            if (human_entity not in ("yes", "no") or human_language not in ("yes", "no")
                    or reviewer != "Reviewer A" or not row["ReviewerA: error type"].strip()):
                raise ValueError("A partly completed row has invalid ReviewerA review fields: " + audit_id)
            complete += 1
            info = index[audit_id]
            distinct_facts.add(info["key"]["fact_id"])
            reviewed_fact_ids.add(info["key"]["fact_id"])
            machine = info["machine_primary"]
            machine_entity = "yes" if machine["entity_correct"] else "no"
            machine_joint = "yes" if machine["requested_language_correct"] else "no"
            human_joint = "yes" if human_entity == "yes" and human_language == "yes" else "no"
            confusion[f"{human_entity}/{machine_entity}"] += 1
            joint_confusion[f"{human_joint}/{machine_joint}"] += 1
            if human_entity == "yes" and machine_entity == "yes":
                # Only here does the machine's joint metric reduce to a
                # standalone output-language decision. Never compare its
                # false value directly with language on a wrong-entity row.
                conditional_language[f"{human_language}/{machine_joint}"] += 1
            category_counts[row["ReviewerA: error type"].strip()] += 1
            cap = machine["category"] == "malformed_truncated"
            semantic = (semantic_overrides[audit_id]["semantic_incomplete"]
                        if audit_id in semantic_overrides
                        else row["ReviewerA: error type"].strip() == "truncated")
            cap_semantic[f"cap_{int(cap)}/semantic_incomplete_{int(semantic)}"] += 1
            stratum = info["sampling_stratum"]
            stratum_n[stratum] += 1
            stratum_population[stratum] = info["stratum_size"]
            if human_entity != machine_entity:
                stratum_disagreement[stratum] += 1
                report["disagreements"].append({
                    "audit_id": audit_id, "language": language,
                    "relation": row["Relation"], "gold": row["Gold answer"],
                    "output": row["Aya output"], "ReviewerA_entity": human_entity,
                    "machine_entity": machine_entity,
                    "ReviewerA_notes": row["ReviewerA: notes"],
                })
        report["reviewed"] += complete
        report["entity_agreement_count"] += confusion["yes/yes"] + confusion["no/no"]
        report["joint_entity_language_agreement_count"] += joint_confusion["yes/yes"] + joint_confusion["no/no"]
        report["conditional_language_agreement_count"] += conditional_language["yes/yes"] + conditional_language["no/no"]
        report["conditional_language_denominator"] += sum(conditional_language.values())
        report["human_standalone_language_yes_count"] += sum(
            row["ReviewerA: requested language correct (yes/no)"].strip().lower() == "yes"
            for row in subset if row["Reviewer"].strip() == "Reviewer A")
        population = sum(stratum_population.values())
        weighted = (sum(stratum_population[s] * stratum_disagreement[s] / stratum_n[s]
                        for s in stratum_n) / population) if population and complete == expected_size else None
        report["by_language"].append({
            "language": language, "sampled": expected_size, "reviewed": complete,
            "distinct_facts_reviewed": len(distinct_facts),
            "entity_confusion_human/machine": dict(confusion),
            "entity_agreement": ((confusion["yes/yes"] + confusion["no/no"]) / complete) if complete else None,
            "entity_cohen_kappa_sample": kappa(confusion),
            "entity_and_language_confusion_human/machine": dict(joint_confusion),
            "human_standalone_language_yes": sum(row["ReviewerA: requested language correct (yes/no)"].strip().lower() == "yes"
                                                   for row in subset if row["Reviewer"].strip() == "Reviewer A"),
            "conditional_language_comparison_on_both_entity_correct": dict(conditional_language),
            "conditional_language_denominator": sum(conditional_language.values()),
            "human_error_types": dict(category_counts),
            "weighted_entity_disagreement_estimate": weighted,
            "weighting_note": "Stratum population/sampled counts; only defined after complete review. The sample was outcome-stratified, so unweighted prevalence is not population prevalence. Sparse strata and repeated facts make a reliable confidence interval unavailable; no row-independent CI is reported.",
        })
    report["distinct_facts_reviewed"] = len(reviewed_fact_ids)
    report["mechanical_cap_vs_human_semantic_truncation"] = dict(cap_semantic)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sampled": report["sampled"], "reviewed": report["reviewed"],
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

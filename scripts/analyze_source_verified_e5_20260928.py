"""Audit source-disjoint Aya E5 results against frozen data and earlier E2 baselines."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.io import digest
from scripts.analyze_source_verified_aya_20260928 import infer

ARCHIVE = ROOT / "results/aya-source-verified-e5-20260928.tar.gz"
E2_ARCHIVE = ROOT / "results/aya-source-verified-external-20260928.tar.gz"
PROTOCOL = ROOT / "configs/aya_source_verified_e5_protocol_20260928.json"
PLAN = ROOT / "configs/aya_source_verified_e5_analysis_plan_20260928.json"
RUNNER = ROOT / "project_plan/feasibility/source_verified_e5_run_20260928.py"
OUT = ROOT / "results/aya-source-verified-e5-analysis-20260928.json"
MD = ROOT / "results/AYA_SOURCE_VERIFIED_E5_RESULTS_20260928.md"
CONDITIONS = ("same_entity_U_to_D", "unrelated_U_to_D", "identity_D_to_D",
              "first_subject_U_to_D")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items(archive_path: Path, prefix: str) -> tuple[dict, dict, dict]:
    items = defaultdict(list)
    completions = defaultdict(list)
    manifests = defaultdict(list)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            cohort = next((cohort for cohort in ("arabic_geonames", "hebrew_cbs")
                           if name.startswith(prefix + cohort)), None)
            if cohort is None:
                continue
            with archive.extractfile(member) as stream:
                record = json.load(stream)
            if "/items/" in name and name.endswith(".json"):
                items[cohort].append(record)
            elif name.endswith("/completion.json"):
                completions[cohort].append(record)
            elif name.endswith("/manifest.json"):
                manifests[cohort].append(record)
    return items, completions, manifests


def verified_rows(protocol: dict) -> dict[str, list[dict]]:
    items, completions, manifests = load_items(
        ARCHIVE, "results/aya-source-verified-e5-20260928-")
    for cohort, setting in protocol["cohorts"].items():
        frozen = {row["fact_id"]: row for row in setting["rows"]}
        rows = items[cohort]
        if (len(rows) != len(frozen) or len(completions[cohort]) != 1 or
                len(manifests[cohort]) != 1 or
                {row["key"]["fact_id"] for row in rows} != set(frozen)):
            raise ValueError("Source E5 result grid is incomplete")
        manifest = manifests[cohort][0]
        completion = completions[cohort][0]
        if (manifest["protocol_sha256"] != sha(PROTOCOL) or
                manifest["runner_sha256"] != sha(RUNNER) or
                manifest["cohort"] != cohort or
                completion["total_items"] != len(frozen) or
                completion["status"] != "complete"):
            raise ValueError("Source E5 frozen identity or completion differs")
        if set(completion["expected_keys"]) != {digest(row["key"]) for row in rows}:
            raise ValueError("Source E5 item keys differ from completion")
        for row in rows:
            reference = frozen[row["key"]["fact_id"]]
            if (row["status"] != "complete" or
                    row["fact"]["subject_qid"] != reference["subject_qid"] or
                    row["fact"]["answer_stratum"] != reference["answer_stratum"] or
                    row["fact"]["donor_fact_id"] != reference["donor_fact_id"]):
                raise ValueError("Source E5 fact or donor differs from frozen protocol")
            for condition in ("baseline_U", "baseline_D", *CONDITIONS):
                scores = row["scores"][condition]
                arithmetic = (scores["gold"]["sum_logprob"] -
                              scores["distractor"]["sum_logprob"])
                if abs(scores["gold_minus_distractor"] - arithmetic) > 1e-6:
                    raise ValueError("E5 recorded margin arithmetic differs")
    return items


def verify_e2_baselines(items: dict) -> dict:
    e2, completions, manifests = load_items(
        E2_ARCHIVE, "results/aya-source-verified-20260928-")
    checks = {}
    for cohort, rows in items.items():
        old = {row["fact"]["fact_id"]: row for row in e2[cohort]}
        if len(completions[cohort]) != 2 or len(manifests[cohort]) != 2:
            raise ValueError("Source E2 archive is incomplete")
        deviations = []
        for row in rows:
            fact_id = row["fact"]["fact_id"]
            if fact_id not in old:
                raise ValueError("E5 subject missing from earlier E2 run")
            for variant in ("U", "D"):
                for answer in ("gold", "distractor"):
                    e5 = row["scores"][f"baseline_{variant}"][answer]["sum_logprob"]
                    e2_score = old[fact_id]["scores"]["t1"][variant][answer]["sum_logprob"]
                    deviations.append(abs(e5 - e2_score))
        maximum = max(deviations)
        if maximum > 0.1:
            raise ValueError(f"Source E5/E2 baseline mismatch in {cohort}: {maximum:.4f}")
        checks[cohort] = {"scores_compared": len(deviations),
                          "max_absolute_deviation_nats": maximum,
                          "tolerance_nats": 0.1}
    return checks


def descriptive(values, seed: int) -> dict:
    value = infer(values, seed)
    value.pop("two_sided_p")
    value.pop("test")
    return value


def main() -> None:
    if not ARCHIVE.is_file() or not E2_ARCHIVE.is_file():
        raise FileNotFoundError("Both Aya E5 and source-cohort E2 archives are required")
    protocol = read(PROTOCOL)
    items = verified_rows(protocol)
    baseline_check = verify_e2_baselines(items)
    result = {"method_status": protocol["method_status"],
              "protocol_sha256": sha(PROTOCOL),
              "analysis_plan_sha256": sha(PLAN),
              "e5_archive_sha256": sha(ARCHIVE), "e2_archive_sha256": sha(E2_ARCHIVE),
              "baseline_reproduction": baseline_check,
              "model_id": protocol["model_id"], "revision": protocol["revision"],
              "layers": protocol["layers"], "site": protocol["site"], "cohorts": {}}
    pvals = {}
    for index, (cohort, rows) in enumerate(sorted(items.items())):
        def contrast(condition: str, metric: str = "gold_minus_distractor"):
            return [(row["fact"]["answer_stratum"],
                     (row["scores"][condition][metric] if metric == "gold_minus_distractor"
                      else row["scores"][condition][metric]["sum_logprob"]) -
                     (row["scores"]["baseline_D"][metric] if metric == "gold_minus_distractor"
                      else row["scores"]["baseline_D"][metric]["sum_logprob"]))
                    for row in rows]
        primary = infer(contrast("same_entity_U_to_D"), 1729 + index)
        controls = {condition: descriptive(contrast(condition), 1800 + 10 * index + offset)
                    for offset, condition in enumerate(CONDITIONS[1:])}
        same_minus_unrelated = descriptive([
            (row["fact"]["answer_stratum"],
             row["scores"]["same_entity_U_to_D"]["gold_minus_distractor"] -
             row["scores"]["unrelated_U_to_D"]["gold_minus_distractor"])
            for row in rows], 1900 + index)
        gold_only = descriptive(contrast("same_entity_U_to_D", "gold"), 2000 + index)
        identity_deviations = [abs(row["scores"]["identity_D_to_D"]["gold_minus_distractor"] -
                                   row["scores"]["baseline_D"]["gold_minus_distractor"])
                               for row in rows]
        result["cohorts"][cohort] = {
            "subjects": len(rows), "distinct_answer_strata": primary["answer_strata"],
            "primary_same_entity_margin_gain": primary,
            "controls_vs_baseline_D": controls,
            "same_entity_minus_unrelated_margin": same_minus_unrelated,
            "same_entity_gold_only_gain": gold_only,
            "identity_max_absolute_margin_deviation_nats": max(identity_deviations),
            "per_subject": [{"fact_id": row["fact"]["fact_id"],
                             "answer_stratum": row["fact"]["answer_stratum"],
                             "donor_fact_id": row["fact"]["donor_fact_id"],
                             "margin_gains": {condition: row["scores"][condition]["gold_minus_distractor"] -
                                              row["scores"]["baseline_D"]["gold_minus_distractor"]
                                              for condition in CONDITIONS}}
                            for row in sorted(rows, key=lambda item: item["fact"]["fact_id"])],
        }
        pvals[cohort] = primary["two_sided_p"]
    ordered = sorted(pvals, key=pvals.get)
    adjusted = {}
    previous = 0.0
    for rank, cohort in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * pvals[cohort]))
        adjusted[cohort] = previous
    result["holm_adjusted_primary_p"] = adjusted
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Aya source-disjoint activation-patching replication", "",
             "The patch site (layers 12–13, last subject token) was fixed from the earlier Aya work. This run uses new source-disjoint place subjects and frozen, different-answer donors.", ""]
    for cohort, value in result["cohorts"].items():
        primary = value["primary_same_entity_margin_gain"]
        specific = value["same_entity_minus_unrelated_margin"]
        ci = primary["hierarchical_bootstrap_ci95"]
        sci = specific["hierarchical_bootstrap_ci95"]
        lines += [f"## {cohort}", "",
                  f"- {value['subjects']} subjects, {value['distinct_answer_strata']} answer strata.",
                  f"- Same-entity U→D patch minus unpatched D, gold/distractor margin: {primary['answer_macro_mean']:.3f} nats; hierarchical 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]; two-sided stratum sign-flip p={primary['two_sided_p']:.5g}; Holm p={adjusted[cohort]:.5g}.",
                  f"- Same-entity minus unrelated-donor margin: {specific['answer_macro_mean']:.3f} nats; 95% CI [{sci[0]:.3f}, {sci[1]:.3f}] (control contrast, descriptive).",
                  f"- Identity-patch maximum absolute margin deviation: {value['identity_max_absolute_margin_deviation_nats']:.4f} nats. E2 baseline maximum score deviation: {baseline_check[cohort]['max_absolute_deviation_nats']:.4f} nats.",
                  ""]
    lines += ["All controls were fixed before these E5 outputs. The site came from prior Aya experiments, and the new relations differ by language; the result tests transfer of the patch effect to new sources rather than independent discovery of a circuit. Whole-residual intervention does not separate token count from mark identity or spelling familiarity.", ""]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({cohort: {"subjects": value["subjects"],
                               "primary_gain": value["primary_same_entity_margin_gain"]["answer_macro_mean"],
                               "holm_p": adjusted[cohort]}
                      for cohort, value in result["cohorts"].items()}), flush=True)


if __name__ == "__main__":
    main()

"""Analyze frozen Aya Arabic/Hebrew external cohorts after their archive returns."""

from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.unicode import normalize_answer
from fragmented_facts.io import digest

ARCHIVE = ROOT / "results/aya-source-verified-external-20260928.tar.gz"
PROTOCOL = ROOT / "configs/aya_source_verified_external_protocol_20260928.json"
PLAN = ROOT / "configs/aya_source_verified_analysis_plan_20260928.json"
AMENDMENT = ROOT / "configs/aya_source_verified_analysis_amendment_20260928.json"
OUT = ROOT / "results/aya-source-verified-external-analysis-20260928.json"
MD = ROOT / "results/AYA_SOURCE_VERIFIED_EXTERNAL_RESULTS_20260928.md"
SEED = 1729


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_rows() -> dict[str, list[dict]]:
    protocol = read(PROTOCOL)
    cohorts = {name: [] for name in protocol["cohorts"]}
    completions = defaultdict(list)
    manifests = defaultdict(list)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            if not name.startswith("results/aya-source-verified-20260928-"):
                continue
            with archive.extractfile(member) as stream:
                data = json.load(stream)
            cohort = next((name_key for name_key in cohorts if
                           name.startswith(f"results/aya-source-verified-20260928-{name_key}-shard")), None)
            if cohort is None:
                raise ValueError("Unexpected Aya result cohort")
            if "/items/" in name and name.endswith(".json"):
                cohorts[cohort].append(data)
            elif name.endswith("/completion.json"):
                completions[cohort].append(data)
            elif name.endswith("/manifest.json"):
                manifests[cohort].append(data)
    for cohort, rows in cohorts.items():
        frozen = {row["fact_id"]: row for row in protocol["cohorts"][cohort]["eligible_rows"]}
        if len(rows) != len(frozen) or len(completions[cohort]) != 2 or len(manifests[cohort]) != 2:
            raise ValueError(f"Incomplete Aya {cohort} results: {len(rows)} / {len(frozen)}")
        if {row["fact"]["fact_id"] for row in rows} != set(frozen):
            raise ValueError("Aya results differ from frozen fact identities")
        if len({row["fact"]["subject_qid"] for row in rows}) != len(rows):
            raise ValueError("Repeated Aya subject")
        if {record["protocol_sha256"] for record in manifests[cohort]} != {sha(PROTOCOL)}:
            raise ValueError("Remote Aya protocol hash differs")
        if {record["shard"] for record in manifests[cohort]} != {0, 1} or \
                {record["shard"] for record in completions[cohort]} != {0, 1}:
            raise ValueError("Missing or repeated Aya shard")
        if sum(record["total_items"] for record in completions[cohort]) != len(frozen):
            raise ValueError("Aya shard completion counts differ")
        completed_keys = {key for record in completions[cohort] for key in record["expected_keys"]}
        actual_keys = {digest(row["key"]) for row in rows}
        if len(completed_keys) != len(frozen) or completed_keys != actual_keys:
            raise ValueError("Aya completion keys differ from item records")
        for row in rows:
            fact = row["fact"]
            frozen_row = frozen[fact["fact_id"]]
            if fact["subject_qid"] != frozen_row["subject_qid"] or row["status"] != "complete":
                raise ValueError("Aya item differs from frozen subject identity")
    return cohorts


def infer(values: list[tuple[str, float]], seed: int) -> dict:
    grouped = defaultdict(list)
    for stratum, value in values:
        grouped[stratum].append(float(value))
    groups = [np.asarray(grouped[name], dtype=float) for name in sorted(grouped)]
    means = np.asarray([group.mean() for group in groups])
    macro = float(means.mean())
    subject = float(np.concatenate(groups).mean())
    rng = np.random.default_rng(seed)
    K = len(groups)
    boots = np.empty(10000, dtype=float)
    for index in range(10000):
        sampled = rng.integers(0, K, size=K)
        boots[index] = np.mean([
            rng.choice(groups[category], size=len(groups[category]), replace=True).mean()
            for category in sampled
        ])
    ci = [float(x) for x in np.quantile(boots, [0.025, 0.975])]
    if K <= 18:
        absolute = abs(macro)
        extreme = sum(abs(float(np.mean([mean if bit else -mean for mean, bit
                                         in zip(means, bits)]))) >= absolute - 1e-12
                      for bits in itertools.product((False, True), repeat=K))
        p = extreme / (2**K)
        test = f"exact_answer_stratum_sign_flip_{2**K}_assignments"
    else:
        extreme = 0
        total = 200000
        absolute = abs(macro)
        for size in [5000] * (total // 5000):
            signs = rng.choice(np.array([-1.0, 1.0]), size=(size, K), replace=True)
            values_batch = (signs * means).mean(axis=1)
            extreme += int(np.count_nonzero(np.abs(values_batch) >= absolute - 1e-12))
        p = (extreme + 1) / (total + 1)
        test = "monte_carlo_answer_stratum_sign_flip_200000_plus_one"
    return {"subjects": len(values), "answer_strata": K,
            "answer_macro_mean": macro, "subject_weighted_mean": subject,
            "hierarchical_bootstrap_ci95": ci, "two_sided_p": p,
            "test": test, "bootstrap_replicates": 10000}


def answer_forms(fact: dict, cohort: str) -> list[str]:
    if cohort == "arabic_geonames":
        return fact["answer_ar_aliases_cldr"]
    return [fact["answer_he"]]


def prefix_candidate(text: str, cohort: str) -> str:
    if cohort == "arabic_geonames":
        pattern = r"^\s*(?:الإجابة|الجواب|الدولة|البلد)\s*[:：]\s*"
    else:
        pattern = r"^\s*(?:התשובה|הנפה)\s*[:：]\s*"
    changed = re.sub(pattern, "", text, count=1)
    if cohort == "hebrew_cbs" and changed == text:
        changed = re.sub(r"^\s*נפת\s+", "", text, count=1)
    return changed


def evaluate_output(text: str, fact: dict, cohort: str) -> dict:
    forms = {normalize_answer(value) for value in answer_forms(fact, cohort)}
    raw = normalize_answer(text)
    rematched = normalize_answer(prefix_candidate(text, cohort))
    strict = raw in forms
    prefix = rematched in forms
    return {"strict_exact": strict, "prefix_rematch": prefix,
            "raw_normalized": raw, "prefix_normalized": rematched}


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError("Aya result archive not returned yet")
    protocol = read(PROTOCOL)
    all_rows = archive_rows()
    report = {"protocol_sha256": sha(PROTOCOL), "archive_sha256": sha(ARCHIVE),
              "analysis_plan_sha256": sha(PLAN),
              "post_submission_exploratory_amendment_sha256": sha(AMENDMENT),
              "model_id": protocol["model_id"], "revision": protocol["revision"],
              "precision": protocol["precision"], "cohorts": {}}
    pvals = {}
    for cohort, rows in all_rows.items():
        ordered = sorted(rows, key=lambda row: row["fact"]["fact_id"])
        get_stratum = ((lambda f: f["country_iso2"]) if cohort == "arabic_geonames"
                       else (lambda f: f["answer_he"]))
        effects = {}
        for template_id in ("t1", "t2"):
            values = [(get_stratum(row["fact"]),
                       row["scores"][template_id]["D"]["gold_minus_distractor"] -
                       row["scores"][template_id]["U"]["gold_minus_distractor"])
                      for row in ordered]
            effects[template_id] = infer(values, SEED + (0 if template_id == "t1" else 101))
        gold_change = infer([
            (get_stratum(row["fact"]),
             row["scores"]["t1"]["D"]["gold"]["sum_logprob"] -
             row["scores"]["t1"]["U"]["gold"]["sum_logprob"])
            for row in ordered
        ], SEED + 201)
        generation = {}
        scored_by_variant = {}
        for variant in ("U", "D"):
            scored = [evaluate_output(row["generations_t1"][variant]["text"],
                                      row["fact"], cohort) for row in ordered]
            scored_by_variant[variant] = scored
            generation[variant] = {
                "strict_exact_count": sum(item["strict_exact"] for item in scored),
                "prefix_rematch_count": sum(item["prefix_rematch"] for item in scored),
                "prefix_only_rescues": sum(item["prefix_rematch"] and not item["strict_exact"]
                                           for item in scored),
                "denominator": len(scored),
                "generation_hit_cap_count": sum(row["generations_t1"][variant]["truncated"]
                                                for row in ordered),
            }
        paired_generation = {}
        for metric in ("strict_exact", "prefix_rematch"):
            paired_generation[metric] = {
                "both_correct": sum(u[metric] and d[metric] for u, d in
                                    zip(scored_by_variant["U"], scored_by_variant["D"])),
                "unmarked_only_correct": sum(u[metric] and not d[metric] for u, d in
                                             zip(scored_by_variant["U"], scored_by_variant["D"])),
                "marked_only_correct": sum(not u[metric] and d[metric] for u, d in
                                           zip(scored_by_variant["U"], scored_by_variant["D"])),
                "neither_correct": sum(not u[metric] and not d[metric] for u, d in
                                       zip(scored_by_variant["U"], scored_by_variant["D"])),
            }
        english = None
        if cohort == "arabic_geonames":
            english_recalled = [row for row in ordered if
                                normalize_answer(row["english_control_generation"]["text"]) ==
                                normalize_answer(row["fact"]["answer_en"])]
            by_answer = defaultdict(list)
            for row in english_recalled:
                by_answer[row["fact"]["country_iso2"]].append(
                    row["scores"]["t1"]["D"]["gold_minus_distractor"] -
                    row["scores"]["t1"]["U"]["gold_minus_distractor"])
            english = {"strict_exact_count": len(english_recalled),
                       "denominator": len(ordered),
                       "descriptive_exact_recall_subset": {
                           "subjects": len(english_recalled),
                           "answer_strata": len(by_answer),
                           "answer_macro_t1_margin_change": (
                               float(np.mean([np.mean(values) for values in by_answer.values()]))
                               if by_answer else None),
                           "subject_weighted_t1_margin_change": (
                               float(np.mean([value for values in by_answer.values() for value in values]))
                               if by_answer else None),
                           "confirmatory_test": False}}
        token_deltas = [row["scores"]["t1"]["D"]["subject_tokens"] -
                        row["scores"]["t1"]["U"]["subject_tokens"]
                        for row in ordered]
        report["cohorts"][cohort] = {
            "subjects": len(ordered),
            "primary_t1_paired_margin": effects["t1"],
            "secondary_t2_paired_margin": effects["t2"],
            "secondary_t1_gold_loglik_change": gold_change,
            "generation": generation, "paired_generation": paired_generation,
            "english_control": english,
            "subject_token_delta": {"mean": float(np.mean(token_deltas)),
                                    "median": float(np.median(token_deltas)),
                                    "negative": sum(value < 0 for value in token_deltas),
                                    "zero": sum(value == 0 for value in token_deltas),
                                    "positive": sum(value > 0 for value in token_deltas)},
            "per_fact": [{"fact_id": row["fact"]["fact_id"],
                          "subject_qid": row["fact"]["subject_qid"],
                          "answer_stratum": get_stratum(row["fact"]),
                          "t1_margin_change": row["scores"]["t1"]["D"]["gold_minus_distractor"] -
                              row["scores"]["t1"]["U"]["gold_minus_distractor"],
                          "t2_margin_change": row["scores"]["t2"]["D"]["gold_minus_distractor"] -
                              row["scores"]["t2"]["U"]["gold_minus_distractor"],
                          "generation_U": row["generations_t1"]["U"]["text"],
                          "generation_D": row["generations_t1"]["D"]["text"]}
                         for row in ordered],
        }
        pvals[cohort] = effects["t1"]["two_sided_p"]
    ranked = sorted(pvals, key=pvals.get)
    adjusted = {}
    previous = 0.0
    for rank, cohort in enumerate(ranked):
        previous = max(previous, min(1.0, (len(ranked) - rank) * pvals[cohort]))
        adjusted[cohort] = previous
    report["holm_adjusted_primary_p"] = adjusted
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Aya source-verified external cohort results", "",
             "The pilot, first held-out, and Google-RE results remain separate; this is a new domain/relation extension.", ""]
    for cohort, value in report["cohorts"].items():
        primary = value["primary_t1_paired_margin"]
        secondary = value["secondary_t2_paired_margin"]
        lines += [f"## {cohort}", "",
                  f"- Subjects: {value['subjects']}; answer strata: {primary['answer_strata']}.",
                  f"- Primary t1 marked-minus-unmarked gold/distractor margin: {primary['answer_macro_mean']:.3f} nats; hierarchical 95% CI [{primary['hierarchical_bootstrap_ci95'][0]:.3f}, {primary['hierarchical_bootstrap_ci95'][1]:.3f}]; two-sided p={primary['two_sided_p']:.5g}; Holm p={adjusted[cohort]:.5g}.",
                  f"- t2 replication: {secondary['answer_macro_mean']:.3f} nats; 95% CI [{secondary['hierarchical_bootstrap_ci95'][0]:.3f}, {secondary['hierarchical_bootstrap_ci95'][1]:.3f}].",
                  f"- Subject token delta median: {value['subject_token_delta']['median']:.1f}.",
                  f"- Greedy exact U/D: {value['generation']['U']['strict_exact_count']}/{value['subjects']} and {value['generation']['D']['strict_exact_count']}/{value['subjects']}; prefix-rematch U/D: {value['generation']['U']['prefix_rematch_count']}/{value['subjects']} and {value['generation']['D']['prefix_rematch_count']}/{value['subjects']}.",
                  f"- Paired exact correctness: U-only {value['paired_generation']['strict_exact']['unmarked_only_correct']}, D-only {value['paired_generation']['strict_exact']['marked_only_correct']}, both {value['paired_generation']['strict_exact']['both_correct']}, neither {value['paired_generation']['strict_exact']['neither_correct']}.",
                  ""]
        if value["english_control"] is not None:
            control = value["english_control"]
            subset = control["descriptive_exact_recall_subset"]
            estimate = (f"{subset['answer_macro_t1_margin_change']:.3f} nats"
                        if subset["answer_macro_t1_margin_change"] is not None else "unavailable")
            lines += [f"- English-control strict exact: {control['strict_exact_count']}/{control['denominator']}; among exact-recalled subjects, descriptive Arabic t1 margin change: {estimate} ({subset['subjects']} subjects, {subset['answer_strata']} country strata).", ""]
    lines += ["Methods: source and tokenization filters and the primary analysis were frozen before Aya outputs. The likelihood margin compares each gold answer to a fixed, prespecified distractor. Output rematching strips only enumerated answer prefixes; remaining unmatched text has no independent semantic audit.",
              "", "The paired generation cross-tab and the English-recallable subset summary were added after job submission, before inspecting returned outputs; they are exploratory and recorded in the separate analysis amendment.",
              "", "The new Arabic country and Hebrew subdistrict relations are not matched across languages. Name pointing is source attested, without independent speaker adjudication. None of these tests alone isolates token count as the causal mechanism.", ""]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({cohort: {"subjects": value["subjects"],
                               "primary": value["primary_t1_paired_margin"]["answer_macro_mean"],
                               "holm_p": adjusted[cohort]}
                      for cohort, value in report["cohorts"].items()}), flush=True)


if __name__ == "__main__":
    main()

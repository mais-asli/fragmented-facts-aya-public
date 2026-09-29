"""Verify the returned Aya E6 archive and summarize its exploratory components."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.io import digest

ARCHIVE = ROOT / "results/aya-e6-position-decomposition-20260928.tar.gz"
PROTOCOL = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
OUT = ROOT / "results/aya-e6-position-decomposition-analysis-20260928.json"
MD = ROOT / "results/AYA_E6_POSITION_DECOMPOSITION_RESULTS_20260928.md"
COMPONENTS = ("A_to_B", "B_to_C", "C_to_D", "A_to_D")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_rows(protocol: dict) -> list[dict]:
    rows, completions, manifests = [], [], []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            if not name.startswith("results/e6-position-decomposition-20260928-shard"):
                continue
            with archive.extractfile(member) as stream:
                record = json.load(stream)
            if "/items/" in name and name.endswith(".json"):
                rows.append(record)
            elif name.endswith("/completion.json"):
                completions.append(record)
            elif name.endswith("/manifest.json"):
                manifests.append(record)
    frozen = {(row["cohort"], row["fact_id"], row["language"], row["template"]): row
              for row in protocol["rows"]}
    actual = {(row["key"]["cohort"], row["key"]["fact_id"], row["key"]["language"],
               row["key"]["template"]): row for row in rows}
    if len(rows) != len(frozen) or len(actual) != len(frozen) or set(actual) != set(frozen):
        raise ValueError("E6 archive does not match the frozen 246-row grid")
    if len(completions) != 2 or len(manifests) != 2 or \
            {record["shard"] for record in manifests} != {0, 1}:
        raise ValueError("E6 archive is missing or repeating a shard")
    if {record["protocol_sha256"] for record in manifests} != {sha(PROTOCOL)}:
        raise ValueError("E6 remote protocol hash differs")
    if sum(record["total_items"] for record in completions) != len(frozen):
        raise ValueError("E6 completion counts differ")
    expected_keys = {key for record in completions for key in record["expected_keys"]}
    if len(expected_keys) != len(frozen) or expected_keys != {digest(row["key"]) for row in rows}:
        raise ValueError("E6 completion keys differ from item records")
    for key, row in actual.items():
        reference = frozen[key]
        if row["status"] != "complete" or row["key"]["experiment"] != protocol["experiment"]:
            raise ValueError("E6 item status or identity differs")
        if row["fact"]["subject_qid"] != reference["subject_qid"] or \
                row["fact"]["relation"] != reference["relation"]:
            raise ValueError("E6 fact differs from frozen protocol")
        sums = row["sum_logprob"]
        for label in ("A", "B", "C", "D"):
            if abs(float(sums[label]) - float(row["condition_scores"][label]["sum_logprob"])) > 1e-7:
                raise ValueError("E6 condition score differs from recorded log-likelihood")
        for label in ("A", "D"):
            if abs(float(sums[label]) - float(reference["e2_gold_sum"][label])) > \
                    protocol["baseline_reproducibility_tolerance_nats"]:
                raise ValueError("E6 original Aya E2 baseline was not reproduced")
        if abs(sum(float(row["components"][part]) for part in COMPONENTS[:3]) -
               float(row["components"]["A_to_D"])) > 1e-6:
            raise ValueError("E6 components do not telescope")
    return rows


def summarize(rows: list[dict], seed: int) -> dict:
    """Relation-macro mean; resample answers, then subjects, within each relation."""
    by_relation = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_relation[row["fact"]["relation"]][row["fact"]["object_qid"]].append(
            np.array([row["components"][part] for part in COMPONENTS], dtype=float))
    relation_means = []
    for answers in by_relation.values():
        relation_means.append(np.mean([value for group in answers.values() for value in group], axis=0))
    point = np.mean(relation_means, axis=0)
    subject_mean = np.mean([[row["components"][part] for part in COMPONENTS]
                            for row in rows], axis=0)
    rng = np.random.default_rng(seed)
    replicates = np.empty((10000, len(COMPONENTS)), dtype=float)
    relations = sorted(by_relation)
    for index in range(len(replicates)):
        means = []
        for relation in relations:
            groups = list(by_relation[relation].values())
            sampled = rng.integers(0, len(groups), size=len(groups))
            picked = []
            for answer_index in sampled:
                group = groups[answer_index]
                picked.extend(group[i] for i in rng.integers(0, len(group), size=len(group)))
            means.append(np.mean(picked, axis=0))
        replicates[index] = np.mean(means, axis=0)
    intervals = np.quantile(replicates, [0.025, 0.975], axis=0)
    values = np.array([[row["components"][part] for part in COMPONENTS] for row in rows])
    b_minus_a = [row["subject_tokens"]["B"] - row["subject_tokens"]["A"]
                 for row in rows]
    d_minus_b = [row["subject_tokens"]["D"] - row["subject_tokens"]["B"]
                 for row in rows]
    return {
        "subject_count": len(rows),
        "relation_count": len(relations),
        "distinct_answer_qids": len({row["fact"]["object_qid"] for row in rows}),
        "relation_counts": {relation: sum(map(len, by_relation[relation].values()))
                            for relation in relations},
        "components": {part: {
            "relation_macro_mean_nats": float(point[index]),
            "subject_weighted_mean_nats": float(subject_mean[index]),
            "answer_cluster_bootstrap_ci95_nats": [float(intervals[0, index]),
                                                    float(intervals[1, index])],
            "negative_subjects": int((values[:, index] < 0).sum()),
            "positive_subjects": int((values[:, index] > 0).sum()),
        } for index, part in enumerate(COMPONENTS)},
        "subject_token_delta_D_minus_A": {
            "median": float(np.median([row["subject_tokens"]["D"] -
                                        row["subject_tokens"]["A"] for row in rows])),
            "mean": float(np.mean([row["subject_tokens"]["D"] -
                                      row["subject_tokens"]["A"] for row in rows])),
        },
        "subject_token_steps": {
            "B_minus_A_median": float(np.median(b_minus_a)),
            "B_minus_A_range": [min(b_minus_a), max(b_minus_a)],
            "A_B_same_count_subjects": sum(value == 0 for value in b_minus_a),
            "D_minus_B_median": float(np.median(d_minus_b)),
            "D_minus_B_range": [min(d_minus_b), max(d_minus_b)],
        },
    }


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError("Aya E6 result archive has not returned yet")
    protocol = read(PROTOCOL)
    rows = archive_rows(protocol)
    cohorts = sorted({row["key"]["cohort"] for row in rows})
    report = {"method_status": protocol["method_status"],
              "protocol_sha256": sha(PROTOCOL), "archive_sha256": sha(ARCHIVE),
              "model_id": protocol["model_id"], "revision": protocol["revision"],
              "precision": protocol["precision"], "total_fact_language_pairs": len(rows),
              "cells": {}}
    lines = ["# Aya E6 exploratory position decomposition", "",
             "The 246 frozen fact-language rows were recovered and checked against the earlier E2 baselines.",
             "No pilot, held-out, and external estimates are pooled for a confirmatory claim.", ""]
    for cohort in cohorts:
        for language in ("ar", "he"):
            cell_rows = [row for row in rows if row["key"]["cohort"] == cohort and
                         row["key"]["language"] == language]
            if not cell_rows:
                continue
            label = f"{cohort}_{language}"
            value = summarize(cell_rows, 1729 + len(report["cells"]))
            report["cells"][label] = value
            lines += [f"## {cohort}, {language}", "",
                      f"{value['subject_count']} subjects; {value['distinct_answer_qids']} distinct answer QIDs; "
                      f"{value['relation_count']} relations. Relation-macro marked-minus-unmarked A→D: "
                      f"{value['components']['A_to_D']['relation_macro_mean_nats']:.3f} nats.",
                      f"Subject-token steps: B−A median {value['subject_token_steps']['B_minus_A_median']:.1f} "
                      f"(range {value['subject_token_steps']['B_minus_A_range'][0]}–{value['subject_token_steps']['B_minus_A_range'][1]}); "
                      f"D−B median {value['subject_token_steps']['D_minus_B_median']:.1f} "
                      f"(range {value['subject_token_steps']['D_minus_B_range'][0]}–{value['subject_token_steps']['D_minus_B_range'][1]}). "
                      f"A and B have the same token count for only {value['subject_token_steps']['A_B_same_count_subjects']} subjects.", "",
                      "| Contrast | Relation-macro mean (nats) | 95% answer-cluster bootstrap CI |",
                      "|---|---:|---:|"]
            for part in COMPONENTS:
                item = value["components"][part]
                lo, hi = item["answer_cluster_bootstrap_ci95_nats"]
                lines.append(f"| {part.replace('_to_', '→')} | {item['relation_macro_mean_nats']:.3f} | "
                             f"[{lo:.3f}, {hi:.3f}] |")
            lines.append("")
    lines += ["A→B keeps visible unmarked text fixed but changes token identities and usually token count; B→C changes retained-token RoPE positions using counterfactual gaps while keeping the token sequence fixed; C→D inserts mark tokens and changes attention context and length. These components sum to A→D by construction. They do not isolate a pure token-count or pure spelling effect.",
              "", "All estimates are exploratory because E6 was designed after viewing the earlier results. The bootstrap treats shared answer QIDs as clusters within relation; sparse answer strata can make its interval unstable.", ""]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "cells": list(report["cells"])}), flush=True)


if __name__ == "__main__":
    main()

"""Verify Aya E7 against E6 and summarize matched-count likelihood controls."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.io import digest
from scripts.analyze_e6_position_decomposition_20260928 import archive_rows as e6_archive_rows

ARCHIVE = ROOT / "results/aya-e7-same-count-20260928.tar.gz"
E6_ARCHIVE = ROOT / "results/aya-e6-position-decomposition-20260928.tar.gz"
PROTOCOL = ROOT / "configs/e7_same_count_protocol_20260928.json"
E6_PROTOCOL = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
PLAN = ROOT / "configs/e7_same_count_analysis_plan_20260928.json"
OUT = ROOT / "results/aya-e7-same-count-analysis-20260928.json"
MD = ROOT / "results/AYA_E7_SAME_COUNT_RESULTS_20260928.md"
CONTRASTS = ("D_minus_full_mean", "early_minus_late", "mid_minus_B",
             "early_minus_B", "late_minus_B", "full_mean_minus_mid",
             "D_minus_B", "D_minus_A")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_rows(protocol):
    rows, completions, manifests = [], [], []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            if not name.startswith("results/e7-same-count-20260928-shard"):
                continue
            with archive.extractfile(member) as stream:
                record = json.load(stream)
            if "/items/" in name and name.endswith(".json"):
                rows.append(record)
            elif name.endswith("/completion.json"):
                completions.append(record)
            elif name.endswith("/manifest.json"):
                manifests.append(record)
    frozen = {(row["cohort"], row["fact_id"], row["language"], "t1"): row
              for row in protocol["rows"]}
    actual = {(row["key"]["cohort"], row["key"]["fact_id"],
               row["key"]["language"], row["key"]["template"]) for row in rows}
    if len(rows) != 246 or len(actual) != 246 or actual != set(frozen):
        raise ValueError("E7 result grid differs from the 246 frozen rows")
    if (len(completions) != 2 or len(manifests) != 2 or
            {m["shard"] for m in manifests} != {0, 1} or
            {m["protocol_sha256"] for m in manifests} != {sha(PROTOCOL)} or
            {m["num_shards"] for m in manifests} != {2} or
            any(m["model"][field] != protocol[field]
                for m in manifests for field in
                ("model_id", "revision", "precision", "compute_dtype", "attention")) or
            any(c["status"] != "complete" or c["expected_count"] != c["total_items"]
                for c in completions) or
            sum(c["total_items"] for c in completions) != 246 or
            {key for c in completions for key in c["expected_keys"]} !=
            {digest(row["key"]) for row in rows}):
        raise ValueError("E7 manifests or completions differ")
    for row in rows:
        if row["status"] != "complete" or row["key"]["experiment"] != protocol["experiment"]:
            raise ValueError("E7 item incomplete")
        reference = frozen[(row["key"]["cohort"], row["key"]["fact_id"],
                            row["key"]["language"], row["key"]["template"])]
        if (row["fact"]["subject_qid"] != reference["subject_qid"] or
                row["subject_tokens"] != {"mid": reference["mid_subject_tokens"],
                                          "early": reference["d_subject_tokens"],
                                          "late": reference["d_subject_tokens"]}):
            raise ValueError("E7 fact or token count differs from frozen protocol")
        for label in ("mid", "early", "late"):
            if abs(row["sum_logprob"][label] -
                   row["condition_scores"][label]["sum_logprob"]) > 1e-7:
                raise ValueError("E7 likelihood summary disagrees")
    return rows


def summarize(rows, seed):
    by_relation = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_relation[row["relation"]][row["answer_qid"]].append(
            np.array([row["contrasts"][name] for name in CONTRASTS], dtype=float))
    relations = sorted(by_relation)
    points = np.mean([
        np.mean([value for group in by_relation[relation].values() for value in group], axis=0)
        for relation in relations], axis=0)
    rng = np.random.default_rng(seed)
    replicates = np.empty((10000, len(CONTRASTS)))
    for index in range(len(replicates)):
        means = []
        for relation in relations:
            groups = list(by_relation[relation].values())
            picked = []
            for group_index in rng.integers(0, len(groups), size=len(groups)):
                group = groups[group_index]
                picked.extend(group[j] for j in rng.integers(0, len(group), size=len(group)))
            means.append(np.mean(picked, axis=0))
        replicates[index] = np.mean(means, axis=0)
    intervals = np.quantile(replicates, [0.025, 0.975], axis=0)
    return {"subjects": len(rows), "relations": len(relations),
            "answer_qids": len({row["answer_qid"] for row in rows}),
            "contrasts": {name: {
                "relation_macro_mean_nats": float(points[i]),
                "answer_cluster_bootstrap_ci95_nats":
                    [float(intervals[0, i]), float(intervals[1, i])]
            } for i, name in enumerate(CONTRASTS)}}


def main():
    if not ARCHIVE.is_file() or not E6_ARCHIVE.is_file():
        raise FileNotFoundError("Both E6 and E7 Aya archives are required")
    protocol = read(PROTOCOL)
    if (protocol["e6_protocol_sha256"] != sha(E6_PROTOCOL) or
            protocol["e6_archive_sha256"] != sha(E6_ARCHIVE)):
        raise ValueError("E7 references different E6 baseline")
    e6 = {(row["key"]["cohort"], row["key"]["fact_id"], row["key"]["language"]): row
          for row in e6_archive_rows(read(E6_PROTOCOL))}
    e7 = archive_rows(protocol)
    joined = []
    for row in e7:
        key = (row["key"]["cohort"], row["key"]["fact_id"], row["key"]["language"])
        baseline = e6[key]
        if row["fact"] != baseline["fact"]:
            raise ValueError("E7 fact identity differs from E6")
        score = baseline["sum_logprob"] | row["sum_logprob"]
        full_mean = (score["early"] + score["late"]) / 2
        contrasts = {
            "D_minus_full_mean": score["D"] - full_mean,
            "early_minus_late": score["early"] - score["late"],
            "mid_minus_B": score["mid"] - score["B"],
            "early_minus_B": score["early"] - score["B"],
            "late_minus_B": score["late"] - score["B"],
            "full_mean_minus_mid": full_mean - score["mid"],
            "D_minus_B": score["D"] - score["B"],
            "D_minus_A": score["D"] - score["A"],
        }
        joined.append({"cohort": key[0], "fact_id": key[1], "language": key[2],
                       "relation": row["fact"]["relation"],
                       "answer_qid": row["fact"]["object_qid"],
                       "scores": score, "contrasts": contrasts})
    cells = {}
    lines = ["# Aya E7 same-visible-text, matched-count controls", "",
             "Exploratory follow-up designed after viewing E6. All 246 E6 fact-language pairs were included. The new forced-split sequences were frozen before the E7 Aya output. Every candidate decodes to the same visible unmarked prompt; the early and late versions have exactly the marked prompt's subject-token count.", "",
             "Outcome: total log likelihood of the same canonical gold answer, in nats. B and D come from the previously verified E6 archive. Each cell is summarized separately, with answer-QID cluster bootstrap within relation and relation-macro aggregation. No confirmatory p-values are claimed.", ""]
    for cohort in ("external_google_re", "heldout", "pilot"):
        for language in ("ar", "he"):
            values = [row for row in joined if row["cohort"] == cohort and
                      row["language"] == language]
            if not values:
                continue
            label = f"{cohort}_{language}"
            summary = summarize(values, 1729 + len(cells))
            cells[label] = summary
            lines += [f"## {cohort}, {language}", "",
                      f"{summary['subjects']} subjects; {summary['answer_qids']} answer QIDs; {summary['relations']} relations.", "",
                      "| Contrast | Relation-macro mean (nats) | 95% answer-cluster CI |",
                      "|---|---:|---:|"]
            for name in CONTRASTS:
                value = summary["contrasts"][name]
                lo, hi = value["answer_cluster_bootstrap_ci95_nats"]
                lines.append(f"| {name.replace('_', ' ')} | {value['relation_macro_mean_nats']:.3f} | [{lo:.3f}, {hi:.3f}] |")
            lines.append("")
    lines += ["B→mid→full compares the same visible unmarked prompt at increasing subject-token counts, but forced splitting necessarily changes token identities. Early and late are different same-count split patterns. D versus their mean matches the subject-token count but changes marked spelling, token identities, and segmentation. Therefore E7 can test robustness of a simple length explanation; it cannot identify a pure causal count effect. All forced splits are outside ordinary tokenizer output and may themselves impair Aya.", "",
              "This likelihood-only diagnostic does not establish generated-answer accuracy, knowledge loss, or a unique neural mechanism.", ""]
    report = {"method_status": protocol["method_status"],
              "protocol_sha256": sha(PROTOCOL), "analysis_plan_sha256": sha(PLAN),
              "e6_archive_sha256": sha(E6_ARCHIVE), "e7_archive_sha256": sha(ARCHIVE),
              "pairs": len(joined), "cells": cells, "per_pair": joined}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"pairs": len(joined), "cells": list(cells)}))


if __name__ == "__main__":
    main()

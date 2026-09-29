"""Post-hoc E2 gold-versus-distractor sensitivity from saved Aya records.

This analysis was proposed after the E2 outcomes were inspected. It does not
replace the frozen primary accuracy or canonical-answer likelihood analyses.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_v2" / "scripts"))
from stats_util import cluster_effect  # noqa: E402

SOURCES = {
    "pilot": ROOT / "paper_v2/_work/e2.jsonl",
    "heldout": ROOT / "paper_v2/_work/ho_e2.jsonl",
    "external_google_re": ROOT / "paper_v2/_work/gre_e2.jsonl",
}
RAW_GLOBS = {
    "heldout": "results/heldout/results/e2-aya-independent-test-v1-shard*/*/items/*.json",
    "external_google_re": "results/gre27/results/e2-aya-google-re-external-20260927-shard*/*/items/*.json",
}
OUT = ROOT / "results/posthoc_gold_distractor_margin_20260928.json"
SEED = 1728
PERMUTATIONS = 100_000


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def subject_signflip(values: list[float], subjects: list[str], relations: list[str]) -> dict:
    """Two-sided Monte Carlo test for the relation-macro paired effect.

    A single sign is flipped per subject, preserving all of its templates.
    Subject sorting makes the Monte Carlo stream invariant to record order.
    """
    cells: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for value, subject, relation in zip(values, subjects, relations, strict=True):
        cells[subject, relation].append(value)
    counts = collections.Counter(relation for _, relation in cells)
    contributions: dict[str, float] = collections.defaultdict(float)
    for (subject, relation), cell in cells.items():
        contributions[subject] += float(np.mean(cell)) / counts[relation] / len(counts)
    ordered = np.asarray([contributions[s] for s in sorted(contributions)], dtype=np.float64)
    observed = abs(float(ordered.sum()))
    rng = np.random.default_rng(SEED)
    extreme = 0
    for _ in range(PERMUTATIONS // 1000):
        signs = rng.choice((-1.0, 1.0), size=(1000, len(ordered)))
        extreme += int(np.count_nonzero(np.abs(signs @ ordered) >= observed - 1e-12))
    p = (extreme + 1) / (PERMUTATIONS + 1)
    return {"two_sided_p": p, "permutations": PERMUTATIONS,
            "monte_carlo_se": float(np.sqrt(p * (1 - p) / PERMUTATIONS))}


def summarize(values: list[float], facts: list[str]) -> dict:
    subjects = [fact.split("-")[0] for fact in facts]
    relations = [fact.split("-")[1] for fact in facts]
    point, ci, valid, n_subjects = cluster_effect(values, subjects, relations)
    return {"relation_macro_mean": point, "subject_cluster_bootstrap_ci95": ci,
            "bootstrap_valid_replicates": valid, "subjects": n_subjects,
            "prompt_pairs": len(values)}


def verify_raw_distractors(cohort: str, pairs: dict) -> dict:
    """Check that every U/D pair scores the same answer and distractor."""
    if cohort == "pilot":
        for key, pair in pairs.items():
            if pair["U"]["distractor"] != pair["D"]["distractor"]:
                raise ValueError(f"Pilot distractor changed: {key}")
        return {"scope": "consolidated_pilot_records", "pairs_checked": len(pairs)}
    from glob import glob

    paths = sorted(Path(x) for x in glob(str(ROOT / RAW_GLOBS[cohort])))
    raw = {}
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        key = row["key"]
        k = (key["fact_id"], key["language"], key["template"], key["variant"])
        if k in raw:
            raise ValueError(f"Duplicate raw record: {k}")
        raw[k] = row
    if len(raw) != len(pairs) * 2:
        raise ValueError(f"{cohort}: raw count {len(raw)} != {len(pairs) * 2}")
    for (fact, language, template), pair in pairs.items():
        u = raw[fact, language, template, "U"]
        d = raw[fact, language, template, "D"]
        if (u["distractor_score"]["answer"] != d["distractor_score"]["answer"]
                or u["gold_score"]["answer"] != d["gold_score"]["answer"]):
            raise ValueError(f"Answer or distractor changed: {cohort}/{fact}/{language}/{template}")
        for variant, row in (("U", u), ("D", d)):
            saved = pair[variant]
            if abs(row["gold_score"]["sum_logprob"] - saved["gold_sum"]) > 1e-8:
                raise ValueError("Consolidated gold score differs from raw")
            if abs(row["likelihood_margin"] - saved["likelihood_margin"]) > 1e-8:
                raise ValueError("Consolidated margin differs from raw")
    return {"scope": "original_per_prompt_records", "pairs_checked": len(pairs),
            "raw_records": len(raw)}


def main() -> None:
    result = {
        "status": "exploratory_post_hoc_after_E2_outcomes_seen",
        "purpose": "Check whether marked spelling lowers gold answer evidence relative to a fixed distractor",
        "metric": "(gold_sum - distractor_sum)_D - (gold_sum - distractor_sum)_U",
        "aggregation": "templates within subject, subjects within relation, four or two relations equally weighted",
        "bootstrap": {"seed": 17, "requested_replicates": 10_000, "unit": "subject"},
        "permutation": {"seed": SEED, "draws": PERMUTATIONS, "unit": "subject", "two_sided": True},
        "sources": {}, "cohorts": {},
        "limitations": [
            "One same-relation distractor per fact; conclusions may depend on distractor choice.",
            "Gold and distractor answer lengths may differ, but each answer is fixed within its U/D pair.",
            "A relative likelihood margin does not by itself isolate knowledge from response-format effects.",
            "This test was devised after seeing the E2 results and is not a new confirmatory test.",
        ],
    }
    for cohort, path in SOURCES.items():
        blob = path.read_bytes()
        result["sources"][cohort] = {"path": path.relative_to(ROOT).as_posix(),
                                      "sha256": hashlib.sha256(blob).hexdigest()}
        pairs: dict[tuple[str, str, str], dict[str, dict]] = collections.defaultdict(dict)
        for row in read_rows(path):
            key = row["key"]
            if key["language"] not in ("he", "ar"):
                continue
            pair_key = (key["fact_id"], key["language"], key["template"])
            variant = key["variant"]
            if variant in pairs[pair_key]:
                raise ValueError(f"Duplicate {cohort}/{pair_key}/{variant}")
            pairs[pair_key][variant] = row
        expected = {"pilot": 342, "heldout": 240, "external_google_re": 156}[cohort]
        if len(pairs) != expected:
            raise ValueError(f"{cohort}: expected {expected} pairs, got {len(pairs)}")
        result["sources"][cohort]["distractor_identity_check"] = verify_raw_distractors(cohort, pairs)
        cohort_out = {}
        raw_ps = {}
        for language in ("he", "ar"):
            facts, gold_deltas, bad_deltas, margin_deltas = [], [], [], []
            for (fact, lang, template), pair in sorted(pairs.items()):
                if lang != language:
                    continue
                if set(pair) != {"U", "D"}:
                    raise ValueError(f"Missing U/D for {cohort}/{fact}/{lang}/{template}")
                u, d = pair["U"], pair["D"]
                if u["gold_answer"] != d["gold_answer"]:
                    raise ValueError(f"Gold answer changed for {fact}/{lang}/{template}")
                margin_key = "margin" if cohort == "pilot" else "likelihood_margin"
                for row in (u, d):
                    if not np.isfinite(row["gold_sum"]) or not np.isfinite(row[margin_key]):
                        raise ValueError("Nonfinite E2 score")
                gold = d["gold_sum"] - u["gold_sum"]
                margin = d[margin_key] - u[margin_key]
                bad = gold - margin
                facts.append(fact)
                gold_deltas.append(gold)
                bad_deltas.append(bad)
                margin_deltas.append(margin)
            info = {"gold_delta": summarize(gold_deltas, facts),
                    "distractor_delta": summarize(bad_deltas, facts),
                    "gold_minus_distractor_delta": summarize(margin_deltas, facts)}
            info["gold_minus_distractor_delta"].update(subject_signflip(
                margin_deltas, [f.split("-")[0] for f in facts],
                [f.split("-")[1] for f in facts]))
            raw_ps[language] = info["gold_minus_distractor_delta"]["two_sided_p"]
            cohort_out[language] = info
        small, large = sorted(raw_ps, key=raw_ps.get)
        cohort_out[small]["gold_minus_distractor_delta"]["holm_p_two_languages"] = min(1.0, 2 * raw_ps[small])
        cohort_out[large]["gold_minus_distractor_delta"]["holm_p_two_languages"] = min(
            1.0, max(2 * raw_ps[small], raw_ps[large]))
        result["cohorts"][cohort] = cohort_out
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({cohort: {lang: {"estimate": round(vals["gold_minus_distractor_delta"]["relation_macro_mean"], 3),
                                      "ci95": [round(x, 3) for x in vals["gold_minus_distractor_delta"]["subject_cluster_bootstrap_ci95"]],
                                      "holm_p": vals["gold_minus_distractor_delta"]["holm_p_two_languages"]}
                                for lang, vals in group.items()}
                      for cohort, group in result["cohorts"].items()}, indent=2))


if __name__ == "__main__":
    main()

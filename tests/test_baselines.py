import math
import numpy as np
from fragmented_facts.baselines import failure_predictors, majority_baseline, matched_comparison, risk_coverage, clustered_associations


def artificial_observations():
    """Constructed numerical fixtures for estimator behavior; not research data."""
    rng = np.random.default_rng(47)
    rows = []
    for split in ("dev", "test"):
        for i in range(60):
            for t in ("t1", "t2"):
                correct = bool(rng.random() < 0.55)
                rows.append({
                    "data_kind": "synthetic_fixture",
                    "key": {"experiment": "E1", "language": "en", "template": t, "variant": "U", "fact_id": f"{split}-{i}"},
                    "fact": {"fact_id": f"{split}-{i}", "subject_qid": f"{split}-{i}", "object_qid": f"O{i % 5}",
                             "split": split, "relation": f"P{i % 2}"},
                    "prompt": {"base_characters": 5 + i % 7, "subject_words": 1 + i % 3,
                               "subject_tokens": 2 + i % 5},
                    "gold_score": {"answer_tokens": 1 + i % 2},
                    "popularity": {"sitelinks": 10 + i, "pageviews": {"en": None}},
                    "generation": {"answer_mean_logprob": -float(rng.random() * 3)},
                    "evaluation": {"entity_correct": correct}})
    return rows


def test_dev_trained_predictors_and_clustered_loss_differences():
    result = failure_predictors(artificial_observations(), repeats=100)
    fits = [r for r in result if "model" in r]
    assert len(fits) == 3
    assert all(r["development_subjects"] == 60 and math.isfinite(r["brier"]) for r in fits)
    deltas = [r for r in result if "comparison" in r]
    assert len(deltas) == 2 and all(r["n_subjects"] == 60 for r in deltas)


def test_baselines_and_risk_coverage_do_not_require_calibration():
    rows = artificial_observations()
    baseline = majority_baseline(rows)[0]
    assert baseline["covered_items"] == 120
    curve = risk_coverage(rows)[0]
    assert curve["status"] == "uncalibrated_ranking_diagnostic"
    assert curve["points"][-1]["coverage"] == 1.0
    matches = matched_comparison(rows)
    assert matches and all(e["matched_pairs"] <= min(e["n_low"], e["n_high"]) for e in matches)


def test_clustered_glm_reports_population_and_no_causal_claim():
    entry = clustered_associations(artificial_observations())[0]
    assert entry["language"] == "en"
    assert entry["status"] in ("descriptive_association_not_causal", "not_identifiable_or_unstable")
    if entry["status"] == "descriptive_association_not_causal":
        assert entry["n_subjects"] == 60

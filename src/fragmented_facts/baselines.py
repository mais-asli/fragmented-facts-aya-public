"""Development-trained frequency/length predictors, matching, and risk coverage."""
import collections

import numpy as np

from .analysis import cluster_effect
from .io import digest


def feature_record(row):
    prompt = row["prompt"]
    views = row.get("popularity", {}).get("pageviews", {}).get(row["key"]["language"])
    links = row.get("popularity", {}).get("sitelinks")
    return {"log_sitelinks": np.log1p(links) if links is not None else None,
            "log_pageviews": np.log1p(views) if views is not None else None,
            "base_characters": prompt["base_characters"], "subject_words": prompt["subject_words"],
            "answer_tokens": row["gold_score"]["answer_tokens"], "subject_tokens": prompt["subject_tokens"],
            "relation": row["fact"]["relation"], "template": row["key"]["template"]}


def failure_predictors(rows, repeats=10000, seed=17):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import pandas as pd

    results = []
    for lang in sorted({r["key"]["language"] for r in rows}):
        candidates = [r for r in rows if r["key"]["experiment"] == "E1" and r["key"]["language"] == lang]
        dev = [r for r in candidates if r["fact"]["split"] == "dev"]
        test = [r for r in candidates if r["fact"]["split"] == "test"]
        if not dev or not test:
            results.append({"language": lang, "status": "requires_dev_and_test"})
            continue
        ydev = np.array([not r["evaluation"]["entity_correct"] for r in dev], dtype=int)
        ytest = np.array([not r["evaluation"]["entity_correct"] for r in test], dtype=int)
        if len(set(ydev)) < 2:
            results.append({"language": lang, "status": "only_one_development_class"})
            continue
        xdev, xtest = pd.DataFrame([feature_record(r) for r in dev]), pd.DataFrame([feature_record(r) for r in test])
        predictions = {}
        for name, numerical in (
            ("frequency_only", ["log_sitelinks", "log_pageviews"]),
            ("frequency_length", ["log_sitelinks", "log_pageviews", "base_characters", "subject_words", "answer_tokens"]),
            ("plus_fragmentation", ["log_sitelinks", "log_pageviews", "base_characters", "subject_words", "answer_tokens", "subject_tokens"])):
            transform = ColumnTransformer([
                ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler())]), numerical),
                ("category", OneHotEncoder(handle_unknown="ignore"), ["relation", "template"])])
            model = Pipeline([("features", transform), ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=seed))])
            subject_counts = collections.Counter(r["fact"]["subject_qid"] for r in dev)
            weights = np.array([1 / subject_counts[r["fact"]["subject_qid"]] for r in dev])
            model.fit(xdev, ydev, classifier__sample_weight=weights)
            p = model.predict_proba(xtest)[:, 1]
            predictions[name] = p
            results.append({"language": lang, "model": name, "status": "fit_on_dev_evaluated_on_test",
                "development_subjects": len(subject_counts), "test_subjects": len({r["fact"]["subject_qid"] for r in test}),
                "log_loss": float(log_loss(ytest, p, labels=[0, 1])), "brier": float(brier_score_loss(ytest, p)),
                "roc_auc": float(roc_auc_score(ytest, p)) if len(set(ytest)) == 2 else None})
        for metric in ("brier", "log_loss"):
            losses = {}
            for name in ("frequency_length", "plus_fragmentation"):
                p = np.clip(predictions[name], 1e-8, 1 - 1e-8)
                losses[name] = (p - ytest) ** 2 if metric == "brier" else -(ytest * np.log(p) + (1 - ytest) * np.log(1 - p))
            delta = (losses["plus_fragmentation"] - losses["frequency_length"]).tolist()
            results.append({"language": lang, "comparison": "plus fragmentation minus frequency/length", "metric": metric,
                **cluster_effect(delta, [r["fact"]["subject_qid"] for r in test], [r["fact"]["relation"] for r in test], repeats, seed)})
    return results


def majority_baseline(rows):
    dev = [r for r in rows if r["key"]["experiment"] == "E1" and r["fact"]["split"] == "dev"]
    objects, seen = collections.defaultdict(collections.Counter), set()
    for r in dev:
        f = r["fact"]
        if f["fact_id"] in seen:
            continue
        seen.add(f["fact_id"])
        objects[f["relation"]][f["object_qid"]] += 1
    chosen = {rel: sorted(counts, key=lambda q: (-counts[q], q))[0] for rel, counts in objects.items()}
    results = []
    for lang in sorted({r["key"]["language"] for r in rows}):
        test = [r for r in rows if r["key"]["experiment"] == "E1" and r["fact"]["split"] == "test" and r["key"]["language"] == lang]
        covered = [r for r in test if r["fact"]["relation"] in chosen]
        results.append({"language": lang, "chosen_objects_from_dev": chosen, "covered_items": len(covered),
            "uncovered_items": len(test) - len(covered), "micro_accuracy": float(np.mean(
                [chosen[r["fact"]["relation"]] == r["fact"]["object_qid"] for r in covered])) if covered else None})
    return results


def matched_comparison(rows, seed=17):
    """Development median defines high/low groups; greedy matching without replacement.

    Calipers: 0.2 development SD log-sitelinks, two base characters, equal word
    count, at most one answer token. Only t1, one fact per subject, is used.
    """
    result = []
    for lang in ("en", "he", "ar"):
        for relation in sorted({r["fact"]["relation"] for r in rows}):
            selected = [r for r in rows if r["key"]["experiment"] == "E1" and r["key"]["template"] == "t1"
                and r["key"]["language"] == lang and r["fact"]["relation"] == relation]
            dev = [r for r in selected if r["fact"]["split"] == "dev"]
            test = [r for r in selected if r["fact"]["split"] == "test"]
            if not dev or not test:
                continue
            threshold = float(np.median([r["prompt"]["subject_tokens"] for r in dev]))
            dev_pop = [feature_record(r)["log_sitelinks"] for r in dev if feature_record(r)["log_sitelinks"] is not None]
            if len(dev_pop) < 2:
                continue
            scale = float(np.std(dev_pop))
            high = [r for r in test if r["prompt"]["subject_tokens"] > threshold]
            low = [r for r in test if r["prompt"]["subject_tokens"] <= threshold]
            pairs = []
            high.sort(key=lambda r: digest([seed, r["key"]]))
            available = list(low)
            for hi in high:
                x = feature_record(hi)
                if x["log_sitelinks"] is None:
                    continue
                possible = []
                for lo in available:
                    y = feature_record(lo)
                    if y["log_sitelinks"] is None:
                        continue
                    dist = abs(x["log_sitelinks"] - y["log_sitelinks"])
                    if dist <= 0.2 * scale and abs(x["base_characters"] - y["base_characters"]) <= 2 and x["subject_words"] == y["subject_words"] and abs(x["answer_tokens"] - y["answer_tokens"]) <= 1:
                        possible.append((dist, digest(lo["key"]), lo))
                if possible:
                    chosen = min(possible, key=lambda v: (v[0], v[1]))[2]
                    pairs.append((hi, chosen))
                    available.remove(chosen)
            result.append({"language": lang, "relation": relation, "threshold_from_dev": threshold,
                "n_high": len(high), "n_low": len(low), "matched_pairs": len(pairs),
                "pairs": [{"high": h["fact"]["subject_qid"], "low": l["fact"]["subject_qid"],
                    "accuracy_delta": int(h["evaluation"]["entity_correct"]) - int(l["evaluation"]["entity_correct"]),
                    "log_sitelink_delta": feature_record(h)["log_sitelinks"] - feature_record(l)["log_sitelinks"],
                    "base_character_delta": h["prompt"]["base_characters"] - l["prompt"]["base_characters"]} for h, l in pairs]})
    return result


def risk_coverage(rows):
    results = []
    for lang in sorted({r["key"]["language"] for r in rows}):
        sample = [r for r in rows if r["key"]["experiment"] == "E1" and r["fact"]["split"] == "test"
                  and r["key"]["language"] == lang and r["generation"]["answer_mean_logprob"] is not None]
        sample.sort(key=lambda r: (-r["generation"]["answer_mean_logprob"], digest(r["key"])))
        points = []
        for coverage in (0.1, 0.25, 0.5, 0.75, 1.0):
            n = max(1, round(len(sample) * coverage))
            top = sample[:n]
            points.append({"coverage": len(top) / len(sample) if sample else None, "n": len(top),
                "risk": float(np.mean([not r["evaluation"]["entity_correct"] for r in top])) if top else None})
        results.append({"language": lang, "status": "uncalibrated_ranking_diagnostic", "points": points})
    return results


def clustered_associations(rows):
    """Prespecified test-population association; inference clusters repeated subjects."""
    import warnings
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    result = []
    for language in ("en", "he", "ar"):
        sample = [r for r in rows if r["key"]["experiment"] == "E1" and r["fact"]["split"] == "test"
                  and r["key"]["language"] == language]
        if len({r["fact"]["subject_qid"] for r in sample}) < 30:
            result.append({"language": language, "status": "fewer_than_30_subject_clusters"})
            continue
        frame = pd.DataFrame([{**feature_record(r), "failure": int(not r["evaluation"]["entity_correct"]),
                               "subject": r["fact"]["subject_qid"]} for r in sample])
        if frame.failure.nunique() < 2:
            result.append({"language": language, "status": "outcome_has_no_variation"})
            continue
        numerical = ["log_sitelinks", "log_pageviews", "base_characters", "subject_words", "answer_tokens", "subject_tokens"]
        included = []
        for name in numerical:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
            missing = frame[name].isna()
            if frame[name].notna().any():
                frame[name] = frame[name].fillna(frame[name].median())
                if frame[name].nunique() > 1:
                    included.append(name)
                if missing.any() and not missing.all():
                    frame[name + "_missing"] = missing.astype(int)
                    included.append(name + "_missing")
        if "subject_tokens" not in included:
            result.append({"language": language, "status": "fragmentation_has_no_variation"})
            continue
        formula = "failure ~ " + " + ".join(included + ["C(relation)", "C(template)"])
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": frame.subject}, maxiter=200)
            coefficient = float(fit.params["subject_tokens"])
            interval = fit.conf_int().loc["subject_tokens"].tolist()
            if not fit.converged or not np.isfinite([coefficient] + interval).all():
                raise ValueError("Unstable/nonconvergent clustered GLM")
            result.append({"language": language, "status": "descriptive_association_not_causal",
                "formula": formula, "n_subjects": frame.subject.nunique(), "fragmentation_log_odds": coefficient,
                "cluster_ci95": interval, "warnings": sorted(set(str(w.message) for w in caught)),
                "missing_policy": "Within analyzed test population median plus missing indicator; not a prediction model."})
        except (ValueError, np.linalg.LinAlgError) as exc:
            result.append({"language": language, "status": "not_identifiable_or_unstable", "reason": str(exc), "formula": formula})
    return result

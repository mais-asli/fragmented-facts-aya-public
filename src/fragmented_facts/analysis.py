"""Subject-clustered paired effects, relation macro averages, and honest coverage."""
import collections
from pathlib import Path

import numpy as np

from .io import digest, read_json, write_json


def load_runs(paths, allow_fixture=False):
    rows, manifests, key_hashes = [], [], set()
    for path in paths:
        path = Path(path)
        manifest = read_json(path / "manifest.json")
        completion = read_json(path / "completion.json")
        if completion.get("status") != "complete":
            raise ValueError("Cannot analyze an incomplete run")
        if manifest.get("data_kind") == "synthetic_fixture" and not allow_fixture:
            raise ValueError("Synthetic test outputs cannot enter research analysis")
        items = [read_json(p) for p in sorted((path / "items").glob("*.json"))]
        if "expected_keys" in completion and {digest(r["key"]) for r in items} != set(completion["expected_keys"]):
            raise ValueError("Missing or unexpected result items")
        for row in items:
            if row.get("status") != "complete":
                raise ValueError("Incomplete result record")
            key = digest({"model": manifest["model"], "protocol": manifest["protocol_hash"], "key": row["key"]})
            if key in key_hashes:
                raise ValueError("Duplicate observation across runs/shards")
            key_hashes.add(key)
            row["manifest"] = manifest
            rows.append(row)
        manifests.append(manifest)
    # Separate model/precision/population comparisons belong in explicit sensitivity analyses.
    for field in ("model", "data_hash", "templates_hash", "code_hash"):
        if len({digest(m[field]) for m in manifests}) != 1:
            raise ValueError(f"Mixed {field}; analyze separately")
    for field in ("population", "max_new_tokens", "seed", "attention"):
        if len({digest(m["config"].get(field)) for m in manifests}) != 1:
            raise ValueError(f"Mixed generation/population setting {field}; use an explicit sensitivity comparison")
    return rows, manifests


def relation_macro(values, subjects, relations):
    """Average repeated prompts within fact/subject-relation, then within relations."""
    cells = collections.defaultdict(list)
    for value, subject, relation in zip(values, subjects, relations):
        cells[subject, relation].append(value)
    by_relation = collections.defaultdict(list)
    for (_, relation), vs in cells.items():
        by_relation[relation].append(float(np.mean(vs)))
    return float(np.mean([np.mean(vs) for vs in by_relation.values()])) if by_relation else None


def cluster_effect(values, subjects, relations, repeats=10000, seed=17):
    """Resample complete subject clusters. Missing relations in a resample are rejected.

    Multiple relations per subject stay together. Repeated draws get distinct
    bootstrap identities, so a subject drawn twice receives twice the weight.
    """
    if not values:
        return {"n_subjects": 0, "estimate": None, "ci95": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite values in statistical analysis")
    ids = sorted(set(subjects))
    all_relations = sorted(set(relations))
    # Compress templates first, retaining all relation cells for each subject.
    cells = collections.defaultdict(list)
    for v, s, r in zip(values, subjects, relations):
        cells[s, r].append(v)
    matrix = np.full((len(ids), len(all_relations)), np.nan)
    for i, s in enumerate(ids):
        for j, r in enumerate(all_relations):
            if (s, r) in cells:
                matrix[i, j] = np.mean(cells[s, r])
    point = float(np.nanmean(np.nanmean(matrix, axis=0)))
    if len(ids) < 2:
        return {"n_subjects": len(ids), "n_observations": len(values), "estimate": point, "ci95": None,
                "limitation": "fewer than two independent subjects"}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        sample = matrix[rng.integers(0, len(ids), size=len(ids))]
        counts = np.isfinite(sample).sum(axis=0)
        if np.all(counts > 0):
            boot.append(float(np.mean(np.nansum(sample, axis=0) / counts)))
    if len(boot) < repeats * 0.8:
        return {"n_subjects": len(ids), "estimate": point, "ci95": None,
                "limitation": "insufficient relation coverage in bootstrap resamples"}
    interval = np.quantile(boot, [0.025, 0.975]).tolist()
    return {"n_subjects": len(ids), "n_observations": len(values), "estimate": point, "ci95": interval,
            "bootstrap_replicates": len(boot), "aggregation": "subject mean, then relation macro"}


def sign_flip_p(values, subjects, relations, repeats=10000, seed=17):
    if not values:
        return None
    # Linear weights implement the same subject/relation-macro estimand.
    cells = collections.defaultdict(list)
    for v, s, r in zip(values, subjects, relations):
        cells[s, r].append(v)
    relation_n = collections.Counter(r for s, r in cells)
    weighted = collections.defaultdict(float)
    for (s, r), vs in cells.items():
        weighted[s] += float(np.mean(vs)) / relation_n[r] / len(relation_n)
    v = np.array(list(weighted.values()))
    observed = abs(float(v.sum()))
    rng = np.random.default_rng(seed)
    exceed = sum(abs(float(np.dot(v, rng.choice([-1, 1], size=len(v))))) >= observed - 1e-12 for _ in range(repeats))
    return (exceed + 1) / (repeats + 1)


def holm(pvalues):
    valid = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    result = [None] * len(pvalues)
    previous = 0.0
    for rank, (p, i) in enumerate(valid):
        previous = max(previous, min(1.0, p * (len(valid) - rank)))
        result[i] = previous
    return result


def effect_for_rows(rows, values, repeats, seed):
    subjects = [r["fact"]["subject_qid"] for r in rows]
    relations = [r["fact"]["relation"] for r in rows]
    report = cluster_effect(values, subjects, relations, repeats, seed)
    report["paired_sign_flip_p"] = sign_flip_p(values, subjects, relations, repeats, seed)
    return report


def paired_rows(rows, experiment, variant_a, variant_b, language, extra=None):
    selected = [r for r in rows if r["key"]["experiment"] == experiment and r["key"]["language"] == language]
    index = collections.defaultdict(dict)
    for r in selected:
        key = r["key"]
        if extra and any(key.get(k) != v for k, v in extra.items()) and key["variant"] not in ("baseline_U", "baseline_D"):
            continue
        variant = key["variant"]
        if variant in (variant_a, variant_b):
            pair = (key["fact_id"], key["template"])
            if variant in index[pair]:
                raise ValueError("Duplicate paired condition: narrow the window/component")
            index[pair][variant] = r
    pairs = [(v[variant_a], v[variant_b]) for v in index.values() if variant_a in v and variant_b in v]
    missing = sum(not (variant_a in v and variant_b in v) for v in index.values())
    return pairs, missing


def analyze_runs(paths, output, repeats=10000, seed=17, allow_fixture=False):
    rows, manifests = load_runs(paths, allow_fixture)
    if not rows:
        raise ValueError("No observations")
    report = {"schema_version": 1, "data_kind": manifests[0]["data_kind"], "run_paths": [str(p) for p in paths],
              "model": manifests[0]["model"], "bootstrap_seed": seed, "bootstrap_requested": repeats,
              "baseline": [], "orthography": [], "patch_specificity": [], "patch_controls": [], "readouts": [],
              "segmentation": [], "descriptive_recovery": [], "run_coverage": [],
              "limitations": ["Aliases matching multiple QIDs are ambiguous, counted incorrect, and require review.",
                              "Confidence scores are not automatically calibrated probabilities.",
                              "Orthographic effects do not isolate token count as the cause."]}
    for path in paths:
        completion = read_json(Path(path) / "completion.json")
        report["run_coverage"].append({"run_path": str(path), "experiment": completion["experiment"],
            "expected_count": completion.get("expected_count"),
            "ineligible_counts": dict(collections.Counter(
                f"{r.get('language', '')}:{r['reason']}" for r in completion.get("ineligible", [])))})
    baseline = [r for r in rows if r["key"]["experiment"] == "E1"]
    for split in sorted({r["fact"]["split"] for r in baseline}):
        for lang in sorted({r["key"]["language"] for r in baseline}):
            sample = [r for r in baseline if r["fact"]["split"] == split and r["key"]["language"] == lang]
            if not sample:
                continue
            entry = {"language": lang, "split": split, "category_counts": dict(collections.Counter(r["evaluation"]["category"] for r in sample)),
                     "relations": dict(collections.Counter(r["fact"]["relation"] for r in sample))}
            for field in ("entity_correct", "requested_language_correct"):
                entry[field] = cluster_effect([int(r["evaluation"][field]) for r in sample],
                    [r["fact"]["subject_qid"] for r in sample], [r["fact"]["relation"] for r in sample], repeats, seed)
                entry[field]["micro_accuracy"] = float(np.mean([r["evaluation"][field] for r in sample]))
            report["baseline"].append(entry)
    # Confirmatory pairs must be test-only, with development summaries reported separately.
    for split in ("dev", "test", "pilot"):
        split_rows = [r for r in rows if r["fact"]["split"] == split]
        for lang in ("he", "ar"):
            pairs, missing = paired_rows(split_rows, "E2", "U", "D", lang)
            if pairs:
                b = [d for u, d in pairs]
                entry = {"language": lang, "split": split, "missing_pairs": missing, "n_pairs": len(pairs),
                         "direction": "D minus U", "transitions": dict(collections.Counter(
                             f"{int(u['evaluation']['entity_correct'])}->{int(d['evaluation']['entity_correct'])}" for u, d in pairs))}
                entry["accuracy"] = effect_for_rows(b, [int(d["evaluation"]["entity_correct"]) - int(u["evaluation"]["entity_correct"]) for u, d in pairs], repeats, seed)
                entry["log_likelihood"] = effect_for_rows(b, [d["gold_score"]["sum_logprob"] - u["gold_score"]["sum_logprob"] for u, d in pairs], repeats, seed)
                entry["mean_token_log_likelihood"] = effect_for_rows(b, [d["gold_score"]["mean_logprob"] - u["gold_score"]["mean_logprob"] for u, d in pairs], repeats, seed)
                report["orthography"].append(entry)
            e3_rows = [r for r in split_rows if r["key"]["experiment"] == "E3"]
            for extra in sorted({r["key"]["extra_tokens"] for r in e3_rows if "extra_tokens" in r["key"]}):
                pairs, missing = paired_rows(split_rows, "E3", "outside_split", "subject_split", lang, {"extra_tokens": extra})
                if pairs:
                    effect = effect_for_rows([s for o, s in pairs],
                        [s["gold_score"]["sum_logprob"] - o["gold_score"]["sum_logprob"] for o, s in pairs], repeats, seed)
                    report["segmentation"].append({"language": lang, "split": split, "extra_tokens": extra,
                        "direction": "subject split minus matched outside-subject split", "effect": effect,
                        "coverage_gate_50_subjects_passed": effect["n_subjects"] >= 50, "missing_pairs": missing})
            windows = sorted({(r["key"]["window"], r["key"]["component"]) for r in split_rows
                if r["key"]["experiment"] == "E5" and "window" in r["key"]})
            for window, component in windows:
                pairs, missing = paired_rows(split_rows, "E5", "unrelated", "same_entity", lang, {"window": window, "component": component})
                if not pairs:
                    continue
                primary = any(m["config"].get("primary_window") == window for m in manifests) and component == "residual"
                entry = {"language": lang, "split": split, "window": window, "component": component,
                         "primary": primary, "missing_pairs": missing, "direction": "same entity minus unrelated donor"}
                entry["specificity"] = effect_for_rows([s for u, s in pairs],
                    [s["gold_score"]["sum_logprob"] - u["gold_score"]["sum_logprob"] for u, s in pairs], repeats, seed)
                report["patch_specificity"].append(entry)
                for condition in ("same_entity", "unrelated", "identity", "first_subject", "delimiter", "reverse"):
                    baseline_variant = "baseline_U" if condition == "reverse" else "baseline_D"
                    controls, incomplete = paired_rows(split_rows, "E5", baseline_variant, condition, lang, {"window": window, "component": component})
                    record = {"language": lang, "split": split, "window": window, "component": component,
                              "condition": condition, "missing_pairs": incomplete}
                    record["score_gain"] = effect_for_rows([p for b, p in controls], [p["gold_score"]["sum_logprob"] - b["gold_score"]["sum_logprob"] for b, p in controls], repeats, seed)
                    evaluated = [(b, p) for b, p in controls if "evaluation" in p]
                    if evaluated:
                        record["accuracy_gain"] = effect_for_rows([p for b, p in evaluated],
                            [int(p["evaluation"]["entity_correct"]) - int(b["evaluation"]["entity_correct"]) for b, p in evaluated], repeats, seed)
                        initially_correct = [(b, p) for b, p in evaluated if b["evaluation"]["entity_correct"]]
                        record["initially_correct_items"] = len(initially_correct)
                        record["harmed_items"] = sum(not p["evaluation"]["entity_correct"] for b, p in initially_correct)
                    report["patch_controls"].append(record)
                # Recovery on the selected U-correct/D-wrong subset is descriptive only.
                originals, _ = paired_rows(split_rows, "E5", "baseline_U", "baseline_D", lang)
                changed, _ = paired_rows(split_rows, "E5", "baseline_D", "same_entity", lang, {"window": window, "component": component})
                restored = {(p["key"]["fact_id"], p["key"]["template"]): p for b, p in changed}
                eligible_recovery = [(u, d) for u, d in originals
                    if u["evaluation"]["entity_correct"] and not d["evaluation"]["entity_correct"]]
                observed = [restored[(d["key"]["fact_id"], d["key"]["template"])] for u, d in eligible_recovery
                    if (d["key"]["fact_id"], d["key"]["template"]) in restored and
                       "evaluation" in restored[(d["key"]["fact_id"], d["key"]["template"])]]
                report["descriptive_recovery"].append({"language": lang, "split": split, "window": window,
                    "component": component, "selected_denominator": len(eligible_recovery),
                    "generation_observed": len(observed), "recovered": sum(r["evaluation"]["entity_correct"] for r in observed),
                    "warning": "Outcome-selected descriptive subset; the all-pair specificity analysis is primary."})
    for section, field in (("orthography", "accuracy"), ("patch_specificity", "specificity")):
        family = [e for e in report[section] if e["split"] == "test" and (section != "patch_specificity" or e["primary"])]
        adjusted = holm([e[field]["paired_sign_flip_p"] for e in family])
        for entry, p in zip(family, adjusted):
            entry[field]["holm_p"] = p
    # The currently reviewed cohort contains pilot subjects only. Its two
    # language effects are exploratory, but their declared families still
    # receive multiplicity correction before being reported.
    pilot = [e for e in report["orthography"] if e["split"] == "pilot"]
    for field in ("accuracy", "log_likelihood"):
        adjusted = holm([e[field]["paired_sign_flip_p"] for e in pilot])
        for entry, p in zip(pilot, adjusted):
            entry[field]["exploratory_holm_p"] = p
    for r in rows:
        if r["key"]["experiment"] == "E4":
            report["readouts"].append({"subject": r["fact"]["subject_qid"], "language": r["key"]["language"],
                "layer": r["key"]["layer"], "site": r["key"]["site"], "split": r["fact"]["split"],
                **r["readout"], "restoration_gain": r["restoration_gain"]})
    report["observation_counts"] = dict(collections.Counter(r["key"]["experiment"] for r in rows))
    write_json(output, report)
    return report

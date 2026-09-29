"""Blind human-output review exports and transparent agreement summaries."""
import collections

from .data import csv_rows, write_csv
from .io import digest, write_json


def export_audit(rows, output, seed=17, per_language=120):
    pool = collections.defaultdict(list)
    for r in rows:
        if "generation" not in r or r["fact"]["split"] != "test":
            continue
        category = r["evaluation"]["category"]
        pool[r["key"]["language"], category].append(r)
    selected, mapping = [], []
    for language in sorted({l for l, _ in pool}):
        buckets = {c: sorted(v, key=lambda r: digest([seed, r["run_id"], r["key"]]))
                   for (l, c), v in pool.items() if l == language}
        used = []
        while len(used) < per_language and any(buckets.values()):
            for category in sorted(buckets):
                if buckets[category] and len(used) < per_language:
                    used.append(buckets[category].pop())
        for i, r in enumerate(used):
            audit_id = digest([seed, r["run_id"], r["key"]])[:20]
            selected.append({"audit_id": audit_id, "language": language,
                "question": r["prompt"]["text"], "output": r["generation"]["text"], "gold_qid": r["fact"]["object_qid"],
                "gold_aliases": str(r["gold_aliases"]), "reviewer": "", "entity_correct": "",
                "requested_language_correct": "", "error_category": "", "notes": "", "double_review": i < 30})
            mapping.append({"audit_id": audit_id, "key": r["key"], "run_id": r["run_id"],
                "sampling_stratum": r["evaluation"]["category"],
                "population_stratum_size": len(pool[language, r["evaluation"]["category"]])})
    if selected:
        write_csv(output, selected, list(selected[0]))
    write_json(str(output) + ".private_map.json", mapping)
    return {"items": len(selected), "note": "Condition labels hidden; the written form itself cannot be blinded. Stratified sample is not an unweighted population error-rate estimate."}


def agreement(first, second):
    from sklearn.metrics import cohen_kappa_score
    a, b = {r["audit_id"]: r for r in csv_rows(first)}, {r["audit_id"]: r for r in csv_rows(second)}
    common = sorted(set(a) & set(b))
    common = [i for i in common if a[i]["entity_correct"] in ("true", "false") and b[i]["entity_correct"] in ("true", "false")]
    if not common:
        raise ValueError("No overlapping completed reviews")
    if any(not a[i]["reviewer"] or not b[i]["reviewer"] or a[i]["reviewer"] == b[i]["reviewer"] for i in common):
        raise ValueError("Agreement requires two distinct named reviewers")
    x, y = [a[i]["entity_correct"] for i in common], [b[i]["entity_correct"] for i in common]
    kappa = float(cohen_kappa_score(x, y)) if len(set(x + y)) > 1 else None
    return {"n_double_annotated": len(common), "raw_agreement": sum(u == v for u, v in zip(x, y)) / len(x),
            "cohen_kappa": kappa, "disagreements": [i for i, u, v in zip(common, x, y) if u != v]}

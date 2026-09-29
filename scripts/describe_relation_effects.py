"""Descriptive per-relation E2 effects from complete Aya pilot pairs."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.analysis import cluster_effect, paired_rows

BASE = ROOT / "results/tau_research_20260926"
SOURCE = BASE / "results/analysis-reviewed-pilot-20260926.json"
OUT = ROOT / "results/relation-effects-pilot-20260926.json"


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    if report["data_kind"] != "research":
        raise ValueError("Actual Aya data required")
    rows = []
    for relative in report["run_paths"]:
        for path in (BASE / relative / "items").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["key"]["experiment"] == "E2":
                rows.append(row)
    results = []
    for lang in ("he", "ar"):
        pairs, missing = paired_rows(rows, "E2", "U", "D", lang)
        if len(pairs) != 171 or missing:
            raise ValueError("Incomplete paired E2 data")
        for relation in ("P19", "P20", "P159", "P740"):
            subset = [(u, d) for u, d in pairs if d["fact"]["relation"] == relation]
            upper = [d for _, d in subset]
            subjects = [d["fact"]["subject_qid"] for d in upper]
            relations = [relation] * len(upper)
            accuracy = [int(d["evaluation"]["entity_correct"]) -
                        int(u["evaluation"]["entity_correct"]) for u, d in subset]
            likelihood = [d["gold_score"]["sum_logprob"] -
                          u["gold_score"]["sum_logprob"] for u, d in subset]
            results.append({
                "language": lang,
                "relation": relation,
                "subjects": len(set(subjects)),
                "prompt_pairs": len(subset),
                "transitions": dict(Counter(
                    f"{int(u['evaluation']['entity_correct'])}->"
                    f"{int(d['evaluation']['entity_correct'])}" for u, d in subset)),
                "accuracy": cluster_effect(accuracy, subjects, relations, 10000, 17),
                "log_likelihood": cluster_effect(likelihood, subjects, relations, 10000, 17),
            })
    OUT.write_text(json.dumps({"scope": "descriptive exploratory per-relation", "rows": results},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

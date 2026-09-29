"""Check whether pilot spelling effects repeat across the three fixed prompts."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.analysis import cluster_effect, paired_rows

BASE = ROOT / "results/tau_research_20260926"
SOURCE = BASE / "results/analysis-reviewed-pilot-20260926.json"
OUT = ROOT / "results/template-effects-pilot-20260926.json"


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    if report["data_kind"] != "research":
        raise ValueError("Actual Aya results required")
    rows = []
    for relative in report["run_paths"]:
        for path in (BASE / relative / "items").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["key"]["experiment"] == "E2":
                rows.append(row)
    results = []
    for language in ("he", "ar"):
        pairs, missing = paired_rows(rows, "E2", "U", "D", language)
        if len(pairs) != 171 or missing:
            raise ValueError("Incomplete E2 paired observations")
        for template in ("t1", "t2", "t3"):
            subset = [(u, d) for u, d in pairs if d["key"]["template"] == template]
            if len(subset) != 57:
                raise ValueError("Incomplete template")
            record = [d for _, d in subset]
            subjects = [d["fact"]["subject_qid"] for d in record]
            relations = [d["fact"]["relation"] for d in record]
            accuracy = [int(d["evaluation"]["entity_correct"]) -
                        int(u["evaluation"]["entity_correct"]) for u, d in subset]
            likelihood = [d["gold_score"]["sum_logprob"] -
                          u["gold_score"]["sum_logprob"] for u, d in subset]
            results.append({"language": language, "template": template,
                            "n_subjects": len(set(subjects)),
                            "accuracy": cluster_effect(accuracy, subjects, relations, 10000, 17),
                            "log_likelihood": cluster_effect(likelihood, subjects, relations, 10000, 17)})
    OUT.write_text(json.dumps({"scope": "exploratory prompt robustness", "rows": results},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

"""Summarize the frozen exact-name rule against imported-alias sensitivity."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.analysis import cluster_effect, paired_rows

BASE = ROOT / "results/tau_research_20260926"
SOURCE = BASE / "results/analysis-reviewed-pilot-20260926.json"
OUTPUT = ROOT / "results/alias-sensitivity-pilot-20260926.json"


def effect(rows, values):
    return cluster_effect(
        values,
        [row["fact"]["subject_qid"] for row in rows],
        [row["fact"]["relation"] for row in rows],
        repeats=10000,
        seed=17,
    )


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    if report["data_kind"] != "research" or report["model"]["model_id"] != "CohereLabs/aya-23-8B":
        raise ValueError("Actual Aya research analysis required")
    rows = []
    for relative in report["run_paths"]:
        for path in sorted((BASE / relative / "items").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["key"]["experiment"] in ("E1", "E2"):
                if "evaluation_all_aliases" not in row:
                    raise ValueError("A generation is missing the frozen alias evaluation")
                rows.append(row)
    result = {
        "scope": "Pilot-only sensitivity; primary rule was fixed before Aya outputs",
        "source": str(SOURCE.relative_to(ROOT)),
        "baseline": [],
        "orthography": [],
        "disagreement_counts": Counter(),
    }
    for row in rows:
        if row["evaluation"]["entity_correct"] != row["evaluation_all_aliases"]["entity_correct"]:
            result["disagreement_counts"][row["key"]["language"]] += 1
    for language in ("en", "he", "ar"):
        subset = [r for r in rows if r["key"]["experiment"] == "E1" and r["key"]["language"] == language]
        if len(subset) != 171:
            raise ValueError("Incomplete E1 language")
        result["baseline"].append({
            "language": language,
            "n_prompts": len(subset),
            "canonical": effect(subset, [int(r["evaluation"]["entity_correct"]) for r in subset]),
            "all_aliases": effect(subset, [int(r["evaluation_all_aliases"]["entity_correct"]) for r in subset]),
            "canonical_correct": sum(r["evaluation"]["entity_correct"] for r in subset),
            "all_aliases_correct": sum(r["evaluation_all_aliases"]["entity_correct"] for r in subset),
        })
    for language in ("he", "ar"):
        pairs, missing = paired_rows(rows, "E2", "U", "D", language)
        if len(pairs) != 171 or missing:
            raise ValueError("Incomplete E2 pairs")
        record = [d for _, d in pairs]
        result["orthography"].append({
            "language": language,
            "n_pairs": len(pairs),
            "canonical": effect(record, [int(d["evaluation"]["entity_correct"])
                                         - int(u["evaluation"]["entity_correct"]) for u, d in pairs]),
            "all_aliases": effect(record, [int(d["evaluation_all_aliases"]["entity_correct"])
                                           - int(u["evaluation_all_aliases"]["entity_correct"]) for u, d in pairs]),
            "canonical_transitions": dict(Counter(
                f"{int(u['evaluation']['entity_correct'])}->{int(d['evaluation']['entity_correct'])}"
                for u, d in pairs)),
            "all_aliases_transitions": dict(Counter(
                f"{int(u['evaluation_all_aliases']['entity_correct'])}->"
                f"{int(d['evaluation_all_aliases']['entity_correct'])}"
                for u, d in pairs)),
        })
    result["disagreement_counts"] = dict(result["disagreement_counts"])
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "disagreement_counts": result["disagreement_counts"]}, indent=2))


if __name__ == "__main__":
    main()

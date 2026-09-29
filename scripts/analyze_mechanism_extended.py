"""Postprocess the actual E5 pilot: eight-comparison Holm and late-site control.

The frozen E5 run and its combined primary analysis remain unchanged.
"""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.analysis import effect_for_rows, holm, paired_rows

BASE = ROOT / "results/tau_mechanism_20260926"
SOURCE = BASE / "results/analysis-reviewed-pilot-with-mechanism-20260926.json"
OUT = ROOT / "results/mechanism-extended-pilot-20260926.json"


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    if (report["data_kind"] != "research" or
            report["model"]["model_id"] != "CohereLabs/aya-23-8B"):
        raise ValueError("Actual Aya research analysis required")
    comparisons = [r for r in report["patch_specificity"]
                   if r["split"] == "pilot" and r["component"] == "residual"]
    if len(comparisons) != 8:
        raise ValueError("Expected all four windows in both languages")
    adjusted = holm([r["specificity"]["paired_sign_flip_p"] for r in comparisons])
    entries = [{
        "language": row["language"], "window": row["window"],
        "n_subjects": row["specificity"]["n_subjects"],
        "n_observations": row["specificity"]["n_observations"],
        "effect_nats": row["specificity"]["estimate"],
        "ci95": row["specificity"]["ci95"],
        "paired_sign_flip_p": row["specificity"]["paired_sign_flip_p"],
        "exploratory_holm_p_eight": p,
        "missing_pairs": row["missing_pairs"],
    } for row, p in zip(comparisons, adjusted)]
    paths = sorted(BASE.glob("results/e5-reviewed-pilot-20260926-shard*/*/items/*.json"))
    if not paths:
        raise ValueError("No extracted E5 item records")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row["key"]["experiment"] != "E5" or row["status"] != "complete" for row in rows):
        raise ValueError("Mixed or incomplete E5 records")
    late = []
    for lang in ("he", "ar"):
        pairs, missing = paired_rows(rows, "E5", "baseline_D", "late_prediction", lang)
        if len(pairs) != 57 or missing:
            raise ValueError("Late-prediction coverage is incomplete")
        patched = [p for _, p in pairs]
        likelihood = effect_for_rows(patched,
            [p["gold_score"]["sum_logprob"] - b["gold_score"]["sum_logprob"]
             for b, p in pairs], 10000, 17)
        evaluated = [(b, p) for b, p in pairs if "evaluation" in p]
        accuracy = (effect_for_rows([p for _, p in evaluated],
            [int(p["evaluation"]["entity_correct"]) - int(b["evaluation"]["entity_correct"])
             for b, p in evaluated], 10000, 17) if evaluated else None)
        late.append({"language": lang, "n_pairs": len(pairs),
                     "log_likelihood_gain": likelihood,
                     "accuracy_gain": accuracy,
                     "generation_coverage": len(evaluated),
                     "baseline_correct": sum(b["evaluation"]["entity_correct"] for b, _ in evaluated),
                     "patched_correct": sum(p["evaluation"]["entity_correct"] for _, p in evaluated),
                     "transitions": dict(Counter(
                         f"{int(b['evaluation']['entity_correct'])}->{int(p['evaluation']['entity_correct'])}"
                         for b, p in evaluated))})
    output = {
        "scope": "Exploratory pilot-only E5 extension; all four fixed windows in both languages",
        "family": "Eight language-by-window same-entity minus unrelated-donor likelihood contrasts",
        "specificity": entries,
        "late_prediction_position": late,
        "raw_e5_items": len(rows),
        "limitations": ["All 57 facts are from the original pilot, with no independent test cohort.",
                        "Restoration effects do not isolate MLP, attention, or a unique storage pathway.",
                        "The late-prediction-position control uses one residual layer, not a matched two-layer window."],
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "raw_e5_items": len(rows),
                      "specificity": len(entries), "late": len(late)}, indent=2))


if __name__ == "__main__":
    main()

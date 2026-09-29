"""Count paired generation transitions for all actual E5 controls."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.analysis import paired_rows

BASE = ROOT / "results/tau_mechanism_20260926/results"
OUT = ROOT / "results/e5-generation-transitions-20260926.json"
WINDOWS = ("early", "lower_middle", "upper_middle", "late")
CONDITIONS = ("same_entity", "unrelated", "identity", "first_subject", "delimiter", "reverse")


def main():
    paths = sorted(BASE.glob("e5-reviewed-pilot-20260926-shard*/*/items/*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(rows) != 3078 or any(row["key"]["experiment"] != "E5" for row in rows):
        raise ValueError("Complete actual E5 bundle required")
    output = {"scope": "Exploratory pilot E5 template-1 generated-answer transitions, not independent test",
              "total_raw_items": len(rows), "conditions": []}
    for language in ("he", "ar"):
        for window in WINDOWS:
            for condition in CONDITIONS:
                baseline = "baseline_U" if condition == "reverse" else "baseline_D"
                pairs, missing = paired_rows(rows, "E5", baseline, condition, language,
                                             {"window": window, "component": "residual"})
                if len(pairs) != 57 or missing:
                    raise ValueError("Incomplete E5 generation pairs")
                if any("evaluation" not in p for _, p in pairs):
                    raise ValueError("Missing patched generated answer")
                transitions = Counter(
                    f"{int(b['evaluation']['entity_correct'])}->{int(p['evaluation']['entity_correct'])}"
                    for b, p in pairs)
                output["conditions"].append({
                    "language": language, "window": window, "condition": condition,
                    "n_pairs": len(pairs),
                    "baseline_correct": sum(b["evaluation"]["entity_correct"] for b, _ in pairs),
                    "patched_correct": sum(p["evaluation"]["entity_correct"] for _, p in pairs),
                    "transitions": dict(transitions),
                })
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "conditions": len(output["conditions"])}, indent=2))


if __name__ == "__main__":
    main()

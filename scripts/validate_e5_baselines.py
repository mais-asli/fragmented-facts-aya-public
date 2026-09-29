"""Compare E5 U/D baselines with the earlier E2 template-1 Aya records."""

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR = ROOT / "results/tau_research_20260926/results"
MECHANISM = ROOT / "results/tau_mechanism_20260926/results"
OUT = ROOT / "results/e5-baseline-consistency-20260926.json"


def load(pattern, experiment, variants):
    index = {}
    for path in pattern:
        row = json.loads(path.read_text(encoding="utf-8"))
        key = row["key"]
        if key["experiment"] != experiment or key["template"] != "t1":
            continue
        variant = key["variant"]
        if variant not in variants:
            continue
        match_key = (key["fact_id"], key["language"], variants[variant])
        if match_key in index:
            raise ValueError("Duplicate baseline record")
        index[match_key] = row
    return index


def main():
    e2 = load(BEHAVIOR.glob("e2-reviewed-pilot-20260926-shard*/*/items/*.json"),
              "E2", {"U": "U", "D": "D"})
    e5 = load(MECHANISM.glob("e5-reviewed-pilot-20260926-shard*/*/items/*.json"),
              "E5", {"baseline_U": "U", "baseline_D": "D"})
    if set(e2) != set(e5) or len(e2) != 228:
        raise ValueError("E2/E5 baseline coverage differs from 57 x 2 x 2")
    deltas = []
    prompt_differences = []
    generation_differences = []
    evaluation_differences = []
    for key in sorted(e2):
        a, b = e2[key], e5[key]
        if a["prompt"]["text"] != b["prompt"]["text"]:
            prompt_differences.append(key)
        deltas.append(abs(a["gold_score"]["sum_logprob"] - b["gold_score"]["sum_logprob"]))
        if a["generation"]["text"] != b["generation"]["text"]:
            generation_differences.append(key)
        if a["evaluation"]["entity_correct"] != b["evaluation"]["entity_correct"]:
            evaluation_differences.append(key)
    report = {
        "scope": "Same checkpoint, prompt and decoding reproducibility diagnostic; E5 remains a separate run",
        "pairs": len(e2),
        "prompt_differences": prompt_differences,
        "max_abs_gold_log_likelihood_difference": max(deltas),
        "median_abs_gold_log_likelihood_difference": statistics.median(deltas),
        "gold_log_likelihood_differences_over_1e-3": sum(x > 1e-3 for x in deltas),
        "generation_differences": generation_differences,
        "entity_evaluation_differences": evaluation_differences,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "pairs": len(e2),
                      "max_abs_score_delta": max(deltas),
                      "generation_differences": len(generation_differences)}, indent=2))


if __name__ == "__main__":
    main()

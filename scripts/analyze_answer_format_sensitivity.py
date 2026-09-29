"""Post-hoc, conservative answer-wrapper sensitivity for actual Aya outputs.

This never edits frozen E1/E2 records or the predeclared primary scorer.
Arabic prefix removal is exact and anchored. Hebrew sentence extraction is a
separate, more interpretive diagnostic that requires human checking of rescues.
"""

from collections import Counter
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.analysis import cluster_effect, paired_rows
from fragmented_facts.data import alias_registry
from fragmented_facts.scoring import evaluate_answer

BASE = ROOT / "results/tau_research_20260926"
SOURCE = BASE / "results/analysis-reviewed-pilot-20260926.json"
FACTS = ROOT / "data/curated/pilot-reviewed-20260926.jsonl"
OUT = ROOT / "results/answer-format-sensitivity-pilot-20260926.json"
AR_PREFIX = re.compile(r"^\s*(?:الإجابة|الجواب)\s*[:：]\s*")
HE_COPULA = re.compile(r"^.+\s(?:הוא|היא)\s+(.+)$", flags=re.DOTALL)


def evaluate(row, registry, candidate):
    return evaluate_answer(candidate, row["fact"]["object_qid"],
                           row["key"]["language"], registry,
                           truncated=row["generation"]["truncated"])


def effect(rows, values):
    return cluster_effect(values,
                          [r["fact"]["subject_qid"] for r in rows],
                          [r["fact"]["relation"] for r in rows],
                          repeats=10000, seed=17)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--analysis", type=Path, default=SOURCE)
    parser.add_argument("--facts", type=Path, default=FACTS)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--scope", default="Post-hoc answer-format sensitivity; frozen complete-answer primary remains unchanged")
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if analysis["data_kind"] != "research" or analysis["model"]["model_id"] != "CohereLabs/aya-23-8B":
        raise ValueError("Actual Aya research outputs required")
    facts = [json.loads(line) for line in args.facts.read_text(encoding="utf-8").splitlines()]
    n_facts = len(facts)
    if n_facts < 1 or len({f["subject_qid"] for f in facts}) != n_facts:
        raise ValueError("Frozen facts missing or subject identities are duplicated")
    canonical = alias_registry(facts, mode="canonical")
    rows = []
    for relative in analysis["run_paths"]:
        for path in (args.base / relative / "items").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["key"]["experiment"] in ("E1", "E2"):
                rows.append(row)
    if len(rows) != 21 * n_facts:
        raise ValueError("Incomplete E1/E2 generation records")
    counts = Counter()
    rescues = []
    for row in rows:
        text = row["generation"]["text"]
        original = row["evaluation"]
        row["format_sensitivity"] = {
            "prefix_exact": original,
            "he_terminal_exact": original,
        }
        if row["key"]["language"] == "ar":
            match = AR_PREFIX.match(text)
            if match:
                counts["ar_prefix_records"] += 1
                stripped = text[match.end():]
                evaluated = evaluate(row, canonical, stripped)
                row["format_sensitivity"]["prefix_exact"] = evaluated
                row["format_sensitivity"]["he_terminal_exact"] = evaluated
                if evaluated["entity_correct"] and not original["entity_correct"]:
                    counts["ar_prefix_rescues"] += 1
                    rescues.append({"rule": "arabic_prefix_exact", "key": row["key"],
                                    "gold": row["gold_score"]["answer"], "output": text,
                                    "candidate": stripped})
        elif row["key"]["language"] == "he":
            match = HE_COPULA.match(text.strip())
            if match:
                counts["he_sentence_candidates"] += 1
                tail = match.group(1)
                evaluated = evaluate(row, canonical, tail)
                row["format_sensitivity"]["he_terminal_exact"] = evaluated
                if evaluated["entity_correct"] and not original["entity_correct"]:
                    counts["he_terminal_rescues"] += 1
                    rescues.append({"rule": "hebrew_terminal_exact", "key": row["key"],
                                    "gold": row["gold_score"]["answer"], "output": text,
                                    "candidate": tail})
        if row["generation"]["truncated"]:
            counts["cap_reached_records"] += 1
    output = {
        "scope": args.scope,
        "records": len(rows),
        "counts": dict(counts),
        "baseline": [],
        "orthography": [],
        "rescue_candidates_for_human_check": rescues,
        "rules": {
            "arabic_prefix_exact": "Remove one initial الإجابة: or الجواب: with optional spaces; then require the complete remainder to match a reviewed canonical answer. Retain the original cap rule.",
            "hebrew_terminal_exact": "For Hebrew sentences containing הוא or היא, take the final span after the last copula and require an exact canonical answer; interpretive and requires human audit. Retain the original cap rule.",
        },
    }
    for language in ("en", "he", "ar"):
        group = [r for r in rows if r["key"]["experiment"] == "E1"
                 and r["key"]["language"] == language]
        if len(group) != 3 * n_facts:
            raise ValueError("Missing E1 records")
        output["baseline"].append({
            "language": language,
            "primary": effect(group, [int(r["evaluation"]["entity_correct"]) for r in group]),
            "prefix_exact": effect(group, [int(r["format_sensitivity"]["prefix_exact"]["entity_correct"]) for r in group]),
            "with_he_terminal": effect(group, [int(r["format_sensitivity"]["he_terminal_exact"]["entity_correct"]) for r in group]),
        })
    for language in ("he", "ar"):
        pairs, missing = paired_rows(rows, "E2", "U", "D", language)
        if len(pairs) != 3 * n_facts or missing:
            raise ValueError("Missing E2 pairs")
        record = [d for _, d in pairs]
        entry = {"language": language, "prompt_pairs": len(pairs)}
        for name, field in (("primary", "evaluation"),
                            ("prefix_exact", "prefix_exact"),
                            ("with_he_terminal", "he_terminal_exact")):
            scores = [(u[field] if field == "evaluation" else u["format_sensitivity"][field],
                       d[field] if field == "evaluation" else d["format_sensitivity"][field])
                      for u, d in pairs]
            entry[name] = {
                "effect": effect(record, [int(d["entity_correct"]) - int(u["entity_correct"])
                                          for u, d in scores]),
                "U_correct_prompts": sum(u["entity_correct"] for u, _ in scores),
                "D_correct_prompts": sum(d["entity_correct"] for _, d in scores),
            }
        output["orthography"].append(entry)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": output["counts"],
                      "baseline": [{"language": x["language"],
                                    "primary": x["primary"]["estimate"],
                                    "prefix_exact": x["prefix_exact"]["estimate"],
                                    "with_he_terminal": x["with_he_terminal"]["estimate"]}
                                   for x in output["baseline"]],
                      "E2": [{"language": x["language"],
                              "primary": x["primary"]["effect"]["estimate"],
                              "prefix_exact": x["prefix_exact"]["effect"]["estimate"],
                              "with_he_terminal": x["with_he_terminal"]["effect"]["estimate"]}
                             for x in output["orthography"]]}, indent=2))


if __name__ == "__main__":
    main()

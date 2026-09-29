"""List every new Aya output beside its frozen answer and source evidence."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_source_verified_aya_20260928 import evaluate_output

ANALYSIS = ROOT / "results/aya-source-verified-external-analysis-20260928.json"
OUT = ROOT / "results/aya-source-verified-output-audit-20260928.csv"
DATA = {
    "arabic_geonames": ROOT / "data/curated/aya-external-geonames-arabic-20260928.jsonl",
    "hebrew_cbs": ROOT / "data/curated/aya-external-cbs-hebrew-20260928.jsonl",
}
FIELDS = ("cohort", "fact_id", "subject_qid", "subject_unmarked",
          "subject_marked", "answer", "answer_aliases", "distractor",
          "unmarked_output", "marked_output", "unmarked_exact",
          "marked_exact", "unmarked_prefix_rematch", "marked_prefix_rematch",
          "unmarked_normalized", "marked_normalized", "name_source",
          "independent_fact_source", "identity_source", "verification_status")


def main() -> None:
    report = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    rows = []
    for cohort, path in DATA.items():
        facts = {item["fact_id"]: item for line in path.read_text(encoding="utf-8").splitlines()
                 if (item := json.loads(line))}
        for result in report["cohorts"][cohort]["per_fact"]:
            fact = facts[result["fact_id"]]
            language = "ar" if cohort == "arabic_geonames" else "he"
            u = evaluate_output(result["generation_U"], fact, cohort)
            d = evaluate_output(result["generation_D"], fact, cohort)
            source = fact["source"]
            rows.append({
                "cohort": cohort, "fact_id": fact["fact_id"],
                "subject_qid": fact["subject_qid"],
                "subject_unmarked": fact[f"subject_{language}_U"],
                "subject_marked": fact[f"subject_{language}_D"],
                "answer": fact[f"answer_{language}"],
                "answer_aliases": " | ".join(fact.get("answer_ar_aliases_cldr", [fact.get("answer_he", "")])),
                "distractor": fact[f"distractor_answer_{language}"],
                "unmarked_output": result["generation_U"],
                "marked_output": result["generation_D"],
                "unmarked_exact": u["strict_exact"],
                "marked_exact": d["strict_exact"],
                "unmarked_prefix_rematch": u["prefix_rematch"],
                "marked_prefix_rematch": d["prefix_rematch"],
                "unmarked_normalized": u["raw_normalized"],
                "marked_normalized": d["raw_normalized"],
                "name_source": source.get("name_annotated", source.get("pointed_name", "")),
                "independent_fact_source": source.get("independent_country_fact", source.get("official_fact", "")),
                "identity_source": source.get("entity", source.get("wikidata_identity", "")),
                "verification_status": fact["verification_status"],
            })
    expected = sum(value["subjects"] for value in report["cohorts"].values())
    if len(rows) != expected or len({row["fact_id"] for row in rows}) != expected:
        raise ValueError("Source output audit is incomplete or duplicates facts")
    with OUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["cohort"], row["fact_id"])))
    print(json.dumps({"rows": len(rows), "path": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()

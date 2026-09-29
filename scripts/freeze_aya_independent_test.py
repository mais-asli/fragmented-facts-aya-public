"""Turn a fully completed, independently reviewed CSV into frozen test inputs.

Run only after ReviewerA saves the review workbook, import_aya_review_workbook.py
has extracted her entries, and check_aya_test_review.py passes. This script
does not create human decisions or silently approve AI suggestions.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.data import prepare, write_csv  # noqa: E402
from fragmented_facts.io import file_hash, write_json  # noqa: E402

BATCH = ROOT / "data" / "review" / "heldout_batch_v1"
DEFAULT_CSV = BATCH / "AYA_ONLY_TEST_REVIEW_FROM_WORKBOOK_20260927.csv"
DEFAULT_OUT = ROOT / "data" / "curated" / "aya-independent-test-v1.jsonl"


def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    review_csv = args.csv.resolve()
    output = args.output.resolve()
    if not review_csv.is_relative_to(ROOT.resolve()) or not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Use paths inside the project workspace")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_aya_test_review.py"),
                    "--csv", str(review_csv)], cwd=ROOT, check=True)

    stem = output.with_suffix("")
    fact_review = stem.with_name(stem.name + ".fact-review.csv")
    pair_review = stem.with_name(stem.name + ".pair-review.csv")
    freeze_manifest = stem.with_name(stem.name + ".freeze.json")
    all_targets = [output, fact_review, pair_review, freeze_manifest, output.with_suffix(".summary.json")]
    if any(path.exists() for path in all_targets):
        raise FileExistsError("Independent test freeze already exists; preserve it and choose a new version")

    base = {r["fact_id"]: r for r in read_csv(BATCH / "facts.csv")}
    reviews = read_csv(review_csv)
    facts_out, pairs_out = [], []
    exclusions = Counter()
    for row in reviews:
        fid = row["fact_id"]
        if row["decision_keep_or_reject"].strip().lower() != "keep":
            exclusions[row["relation"]] += 1
            continue
        original = base[fid]
        fact = dict(original)
        fact.update(decision="approved", reviewer="Reviewer A", pre_release_verified="true",
                    evidence_url=row["evidence_url"].strip(),
                    notes=(f"Reviewed {row['review_date']}; pinpoint: {row['evidence_quote_or_page'].strip()}. "
                           f"{row['exclusion_reason_or_notes'].strip()}"),
                    subject_en=row["subject_en"].strip(), object_en=row["answer_en"].strip(),
                    subject_he=row["subject_he_U_candidate"].strip(),
                    object_he=row["answer_he_candidate"].strip(),
                    subject_ar=row["subject_ar_U_candidate"].strip(),
                    object_ar=row["answer_ar_candidate"].strip())
        # Keep the primary reviewed canonical answer and no unreviewed aliases.
        for language in ("en", "he", "ar"):
            fact[f"aliases_{language}"] = json.dumps([fact[f"object_{language}"]], ensure_ascii=False)
        facts_out.append(fact)
        for language in ("he", "ar"):
            pairs_out.append({"fact_id": fid, "language": language,
                              "U": fact[f"subject_{language}"],
                              "D": row[f"subject_{language}_D_proposed"].strip(),
                              "P": "", "decision": "approved", "reviewer": "Reviewer A",
                              "notes": f"Personally reviewed {row['review_date']}."})
    if not facts_out:
        raise ValueError("No reviewed eligible facts; do not create an empty test")
    # The original CSV schema is retained so the established preparation code
    # can apply the original subject-disjoint split manifest and Unicode checks.
    write_csv(fact_review, facts_out, list(next(iter(base.values()))))
    write_csv(pair_review, pairs_out, ["fact_id", "language", "U", "D", "P", "decision", "reviewer", "notes"])
    result = prepare(ROOT / "data" / "raw" / "candidates_final.jsonl", fact_review,
                     pair_review, output, 17, ROOT / "data" / "review" / "subject_splits.json")
    if result["facts"] != len(facts_out) or result["by_split"] != {"test": len(facts_out)}:
        raise ValueError("Prepared facts differ from reviewed test decisions or original split")
    record = {
        "schema_version": 1, "cohort": "independent_outcome_unseen_test_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_candidates": len(reviews), "reviewed_kept": len(facts_out),
        "reviewed_rejected": len(reviews) - len(facts_out),
        "exclusions_by_relation": dict(exclusions),
        "kept_ids": [r["fact_id"] for r in reviews if r["decision_keep_or_reject"].strip().lower() == "keep"],
        "reviewer": "Reviewer A", "study_plan": "docs/AYA_ONLY_HELDOUT_EXTENSION_20260927.md",
        "sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
                   [review_csv, ROOT / "data" / "raw" / "candidates_final.jsonl",
                    ROOT / "data" / "review" / "subject_splits.json",
                    fact_review, pair_review, output]},
        "prepared_counts": result,
    }
    record["manifest_sha256"] = hashlib.sha256(json.dumps(record, ensure_ascii=False,
                            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    write_json(freeze_manifest, record)
    print(json.dumps({"output": str(output), "freeze": str(freeze_manifest),
                      "kept": len(facts_out), "rejected": len(reviews) - len(facts_out),
                      "by_relation": result["by_relation"]}, indent=2))


if __name__ == "__main__":
    main()

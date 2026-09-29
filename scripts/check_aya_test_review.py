"""Check completeness and provenance of the 60-row independent Aya review.

Read-only. A pass means the CSV is structurally ready to freeze, not that the
researcher's factual or pronunciation judgments are objectively true.
"""
import csv
import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.unicode import validate_pair  # noqa: E402

BATCH = ROOT / "data" / "review" / "heldout_batch_v1"
DEFAULT_QUEUE = BATCH / "AYA_ONLY_TEST_REVIEW_QUEUE_20260927.csv"
MANIFEST = BATCH / "manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_QUEUE)
    args = parser.parse_args()
    selected = json.loads(MANIFEST.read_text(encoding="utf-8"))["selected"]
    expected = [r for r in selected if r["split"] == "test"]
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8-sig", newline="")))
    errors = []
    if len(rows) != 60:
        errors.append(f"Expected 60 review rows, got {len(rows)}")
    expected_ids = [r["fact_id"] for r in expected]
    ids = [r["fact_id"] for r in rows]
    if ids != expected_ids:
        errors.append("Fact IDs/order differ from frozen test manifest")
    if len(set(ids)) != len(ids):
        errors.append("Duplicate fact IDs")

    pilot = json.loads((ROOT / "data" / "review" / "ReviewerA_confirmed_all_20260926" /
                        "reviewed_data.json").read_text(encoding="utf-8"))
    pilot_ids = {str(r.get("Fact ID", "")) for r in pilot["Pilot facts"]}
    overlap = sorted(set(ids) & pilot_ids)
    if overlap:
        errors.append(f"Test IDs overlap pilot: {overlap}")
    pilot_subjects = {json.loads(line)["subject_qid"] for line in
                      (ROOT / "data" / "curated" / "pilot-reviewed-20260926.jsonl")
                      .read_text(encoding="utf-8").splitlines() if line.strip()}
    test_subjects = {fact_id.split("-", 1)[0] for fact_id in ids}
    subject_overlap = sorted(test_subjects & pilot_subjects)
    if subject_overlap:
        errors.append(f"Test subjects overlap pilot: {subject_overlap}")

    decisions = Counter()
    keep_by_relation = Counter()
    for i, row in enumerate(rows, 1):
        prefix = f"row {i} {row['fact_id']}: "
        if i <= len(expected) and row["relation"] != expected[i - 1]["relation"]:
            errors.append(prefix + "relation differs from manifest")
        if row["review_order"] != str(i):
            errors.append(prefix + "review order changed")
        decision = row["decision_keep_or_reject"].strip().lower()
        decisions[decision or "blank"] += 1
        if decision not in {"keep", "reject"}:
            errors.append(prefix + "decision must be keep or reject")
            continue
        if row["reviewer"].strip() != "Reviewer A":
            errors.append(prefix + "reviewer must identify Reviewer A after her own check")
        try:
            reviewed = date.fromisoformat(row["review_date"].strip())
            if reviewed < date(2026, 9, 27):
                errors.append(prefix + "review date precedes preparation of this queue")
        except ValueError:
            errors.append(prefix + "review_date must be YYYY-MM-DD")

        if decision == "reject":
            if not row["exclusion_reason_or_notes"].strip():
                errors.append(prefix + "rejection needs a reason")
            continue

        keep_by_relation[row["relation"]] += 1
        required_yes = ["evidence_predates_may_2024", "entity_and_answer_level_verified",
                        "subject_he_U_approved", "subject_he_D_approved",
                        "subject_ar_U_approved", "subject_ar_D_approved"]
        for field in required_yes:
            if row[field].strip().lower() != "yes":
                errors.append(prefix + f"{field} must be yes for a kept fact")
        url = row["evidence_url"].strip()
        if urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc:
            errors.append(prefix + "a direct evidence URL is required")
        if not row["evidence_quote_or_page"].strip():
            errors.append(prefix + "a pinpoint quotation/section/page is required")
        for field in ["subject_en", "answer_en", "subject_he_U_candidate",
                      "subject_ar_U_candidate", "answer_he_candidate", "answer_ar_candidate"]:
            if not row[field].strip():
                errors.append(prefix + f"{field} is empty")
        for language in ("he", "ar"):
            try:
                validate_pair(row[f"subject_{language}_U_candidate"],
                              row[f"subject_{language}_D_proposed"], language)
            except ValueError as exc:
                errors.append(prefix + f"{language} U/D invalid: {exc}")

    print(f"Reviewed decisions: {dict(decisions)}")
    print(f"Kept by relation: {dict(keep_by_relation)}")
    print(f"Pilot fact/subject overlap: {len(overlap)}/{len(subject_overlap)}")
    if errors:
        print(f"REVIEW INCOMPLETE/INVALID: {len(errors)} issue(s)")
        for error in errors[:30]:
            print("- " + error)
        if len(errors) > 30:
            print(f"... {len(errors)-30} additional issues")
        return 1
    print("STRUCTURAL REVIEW CHECK PASSED; source and pronunciation content still require independent judgment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

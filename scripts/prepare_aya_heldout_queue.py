"""Combine the frozen test manifest and cached candidates for blind human review.

This prepares a queue, not approved facts or marked-name suggestions.
"""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "data" / "review" / "heldout_batch_v1"
facts = {r["fact_id"]: r for r in csv.DictReader((BATCH / "facts.csv").open(encoding="utf-8-sig", newline=""))}
selected = json.loads((BATCH / "manifest.json").read_text(encoding="utf-8"))["selected"]
test = [r for r in selected if r["split"] == "test"]
assert len(test) == 60
assert len({r["fact_id"] for r in test}) == 60

columns = [
    "review_order", "relation", "fact_id", "subject_en", "answer_en",
    "subject_he_U_candidate", "subject_ar_U_candidate",
    "answer_he_candidate", "answer_ar_candidate", "evidence_url",
    "evidence_quote_or_page", "evidence_predates_may_2024",
    "entity_and_answer_level_verified", "subject_he_U_approved",
    "subject_he_D_proposed", "subject_he_D_approved",
    "subject_ar_U_approved", "subject_ar_D_proposed",
    "subject_ar_D_approved", "decision_keep_or_reject", "reviewer",
    "review_date", "exclusion_reason_or_notes",
]
out = BATCH / "AYA_ONLY_TEST_REVIEW_QUEUE_20260927.csv"
with out.open("w", encoding="utf-8-sig", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=columns)
    writer.writeheader()
    for i, item in enumerate(test, 1):
        src = facts[item["fact_id"]]
        writer.writerow({
            "review_order": i, "relation": item["relation"],
            "fact_id": item["fact_id"], "subject_en": src["subject_en"],
            "answer_en": src["object_en"],
            "subject_he_U_candidate": src["subject_he"],
            "subject_ar_U_candidate": src["subject_ar"],
            "answer_he_candidate": src["object_he"],
            "answer_ar_candidate": src["object_ar"],
        })
print(out)

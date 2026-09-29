"""Join the frozen 60-row queue to clearly labeled AI research leads."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "data" / "review" / "heldout_batch_v1"
OUT = ROOT / "outputs" / "01a09b51" / "heldout_review_workbook_input.json"

def read(name):
    return list(csv.DictReader((BATCH / name).open(encoding="utf-8-sig", newline="")))

queue = read("AYA_ONLY_TEST_REVIEW_QUEUE_20260927.csv")
sources = {r["fact_id"]: r for r in read("AI_SOURCE_LEADS_NOT_APPROVED_20260927.csv")}
marks = {r["fact_id"]: r for r in read("AI_MARK_SUGGESTIONS_NOT_APPROVED_20260927.csv")}
assert len(queue) == 60 and len(sources) == 60 and len(marks) == 60

columns = [
    ("review_order", "No."), ("relation", "Relation"),
    ("subject_en", "Subject (English)"), ("answer_en", "Answer (English)"),
    ("fact_id", "Fact ID"),
    ("subject_he_U_candidate", "Hebrew name U"),
    ("subject_he_D_proposed", "Hebrew name D — ReviewerA entry"),
    ("subject_he_D_ai_suggestion", "Hebrew D — AI suggestion"),
    ("subject_he_U_approved", "Hebrew U checked?"),
    ("subject_he_D_approved", "Hebrew D checked?"),
    ("subject_ar_U_candidate", "Arabic name U"),
    ("subject_ar_D_proposed", "Arabic name D — ReviewerA entry"),
    ("subject_ar_D_ai_suggestion", "Arabic D — AI suggestion"),
    ("subject_ar_U_approved", "Arabic U checked?"),
    ("subject_ar_D_approved", "Arabic D checked?"),
    ("answer_he_candidate", "Answer (Hebrew)"),
    ("answer_ar_candidate", "Answer (Arabic)"),
    ("ai_source_lead_url", "Source lead — AI suggestion"),
    ("ai_note", "Source caveat — AI"),
    ("evidence_url", "Evidence URL — ReviewerA entry"),
    ("evidence_quote_or_page", "Evidence quote/page — ReviewerA entry"),
    ("evidence_predates_may_2024", "Fact true by May 2024?"),
    ("entity_and_answer_level_verified", "Entity/place level checked?"),
    ("decision_keep_or_reject", "Decision: keep/reject"),
    ("reviewer", "Reviewer"), ("review_date", "Review date YYYY-MM-DD"),
    ("exclusion_reason_or_notes", "Reason / notes"),
]

rows = []
for q in queue:
    fid = q["fact_id"]
    merged = dict(q)
    merged.update({"subject_he_D_ai_suggestion": marks[fid]["subject_he_D_ai_suggestion"],
                   "subject_ar_D_ai_suggestion": marks[fid]["subject_ar_D_ai_suggestion"],
                   "ai_source_lead_url": sources[fid]["ai_source_lead_url"],
                   "ai_note": sources[fid]["ai_note"]})
    rows.append([int(merged[key]) if key == "review_order" else merged.get(key, "")
                 for key, _ in columns])

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False), encoding="utf-8")
print(OUT)

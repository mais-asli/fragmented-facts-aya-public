"""Record ReviewerA's full review and the approved Toyota City disambiguation.

The six retained facts without a direct supporting source remain pending for
research inclusion despite the user's data review. Original files are preserved.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/review/ReviewerA_confirmed_20260926/reviewed_data.json"
ADJUDICATION = ROOT / "data/review/merged_20260925/fact_adjudication.json"
SUMMARY = ROOT / "data/review/merged_20260925/summary.json"
OUT = ROOT / "data/review/ReviewerA_confirmed_all_20260926"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_note(row, note):
    row["Review notes"] = (row.get("Review notes") or "").rstrip() + " " + note


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_verdicts = {r["fact_id"]: r["verdict"] for r in json.loads(ADJUDICATION.read_text(encoding="utf-8"))}
    missing_direct = {
        r["fact_id"] for r in json.loads(SUMMARY.read_text(encoding="utf-8"))["facts_missing_direct_evidence_url"]
    }
    fact_counts = {"approved": 0, "pending_direct_source": 0, "rejected": 0}
    for row in data["Pilot facts"]:
        fid = row["Fact ID"]
        if source_verdicts[fid] == "reject":
            row["Decision"] = "rejected"
            fact_counts["rejected"] += 1
        elif fid in missing_direct:
            row["Decision"] = "pending"
            append_note(row, "ReviewerA confirmed the data review on 2026-09-26; research inclusion awaits a direct supporting source beyond the current link.")
            fact_counts["pending_direct_source"] += 1
        else:
            row["Decision"] = "approved"
            append_note(row, "ReviewerA confirmed all retained fact rows and answer names in chat on 2026-09-26; prior source recheck retained this fact.")
            fact_counts["approved"] += 1
        row["Reviewer"] = "Reviewer A"

    toyota = next(row for row in data["Pilot facts"] if row["Subject in English"] == "Toyota" and row["Relation"] == "Headquarters location")
    if toyota["Fact ID"] != "Q53268-P159-Q201117":
        raise ValueError("Unexpected Toyota fact identity")
    toyota["Answer in English"] = "Toyota City"
    toyota["Accepted English names (JSON)"] = json.dumps(["Toyota City", "Toyota-shi"], ensure_ascii=False)
    toyota["Accepted Hebrew names (JSON)"] = json.dumps(["טויוטה (עיר)"], ensure_ascii=False)
    toyota["Accepted Arabic names (JSON)"] = json.dumps(["تويوتا، آيتشي"], ensure_ascii=False)
    append_note(toyota, "On 2026-09-26 ReviewerA approved using Toyota City and excluding the bare company-identical Toyota answer in all three languages. Official 2012 Toyota source identifies Toyota City, Aichi. The answer QID is unchanged.")

    for sheet in ("Hebrew", "Arabic"):
        if len(data[sheet]) != 70:
            raise ValueError(f"Expected 70 {sheet} pairs")
        for row in data[sheet]:
            previous_decision = row["Decision"]
            previous_reviewer = row.get("Reviewer") or "none"
            row["Decision"] = "approved"
            row["Reviewer"] = "Reviewer A"
            row["Pair review status"] = "Review recorded"
            if previous_decision != "approved" or previous_reviewer != "Reviewer A":
                append_note(row, f"ReviewerA confirmed all 70 Hebrew/Arabic C/D pairs in chat on 2026-09-26. Earlier decision/reviewer: {previous_decision}/{previous_reviewer}.")

    if len(data["Templates"]) != 44:
        raise ValueError("Expected 44 question templates")
    for row in data["Templates"]:
        previous_decision = row["Decision"]
        previous_reviewer = row.get("Reviewer") or "none"
        row["Decision"] = "approved"
        row["Reviewer"] = "Reviewer A"
        row["Template status"] = "Review recorded"
        if previous_decision != "approved" or previous_reviewer != "Reviewer A":
            append_note(row, f"ReviewerA confirmed all 44 question templates in chat on 2026-09-26. Earlier decision/reviewer: {previous_decision}/{previous_reviewer}.")

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "reviewed_data.json"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log = {
        "status": "full_user_review_recorded_with_source_gate",
        "date": "2026-09-26",
        "reviewer": "Reviewer A",
        "user_statement": "Yes, all of those",
        "statement_context": "All 70 Hebrew/Arabic name pairs, all 44 templates and all 62 retained fact rows including answer names in columns N-P",
        "toyota_user_decision": "Yes, use Toyota City",
        "toyota_source": "https://global.toyota/en/detail/151468",
        "fact_counts": fact_counts,
        "approved_pair_counts": {sheet: sum(r["Decision"] == "approved" for r in data[sheet]) for sheet in ("Hebrew", "Arabic")},
        "approved_templates": sum(r["Decision"] == "approved" for r in data["Templates"]),
        "six_direct_source_ids_pending": sorted(missing_direct),
        "source_sha256": sha256(SOURCE),
        "output_sha256": sha256(output),
    }
    (OUT / "review_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(log, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

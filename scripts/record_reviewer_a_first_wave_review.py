"""Record ReviewerA's 26 Sep confirmation of the first-wave name pairs.

This records only the Hebrew/Arabic U/D spelling review she explicitly confirmed
in chat. It does not approve facts, aliases, templates, or reserve subjects.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/review/merged_20260925/reviewed_clean_data.json"
WAVE = ROOT / "data/review/merged_20260925/first_wave_50.csv"
OUT = ROOT / "data/review/ReviewerA_confirmed_20260926"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    with WAVE.open(encoding="utf-8-sig", newline="") as stream:
        ids = {row["Fact ID"] for row in csv.DictReader(stream)}
    if len(ids) != 50:
        raise ValueError("Expected exactly 50 distinct first-wave subjects")
    changed = {}
    for sheet in ("Hebrew", "Arabic"):
        rows = [row for row in data[sheet] if row["Fact ID"] in ids]
        if len(rows) != 50:
            raise ValueError(f"Expected 50 {sheet} pairs")
        for row in rows:
            if not row["Unmarked spelling"] or not row["Marked spelling"]:
                raise ValueError(f"Missing spelling: {row['Fact ID']}")
            previous = row["Decision"]
            previous_reviewer = row.get("Reviewer") or "none"
            row["Decision"] = "approved"
            row["Reviewer"] = "Reviewer A"
            addition = (
                "ReviewerA confirmed in chat on 2026-09-26 that she checked the first-wave "
                "Hebrew and Arabic C/D name pairs and found them correct. "
                f"Previous decision: {previous}; previous reviewer: {previous_reviewer}."
            )
            row["Review notes"] = (row.get("Review notes") or "").rstrip() + " " + addition
            row["Pair review status"] = "Review recorded"
        changed[sheet] = len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "reviewed_data.json"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log = {
        "status": "first_wave_pairs_user_confirmed",
        "date": "2026-09-26",
        "reviewer": "Reviewer A",
        "scope": "50 first-wave subject-name U/D pairs in Hebrew and Arabic only",
        "user_statement": "i checked, they ARE RIGHT",
        "context": "Reply to request to check identity and pronunciation of Hebrew/Arabic C/D for the 50 names in First_Wave_50.csv",
        "changed": changed,
        "not_approved_by_this_statement": ["fact rows", "answer aliases", "question templates", "reserve subject names"],
        "source_sha256": sha256(SOURCE),
        "wave_sha256": sha256(WAVE),
        "output_sha256": sha256(output),
    }
    (OUT / "review_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(log, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

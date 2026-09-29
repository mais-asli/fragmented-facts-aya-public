"""Join pointed Wikidata locality names to the official 2023 CBS table.

This is a source-level candidate filter, not independent pronunciation review.
No Aya outcomes are read or used for selection.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
OUT = ROOT / "data/candidates/official_localities_20260928"
CBS = RAW / "israel_cbs_localities_2023.json"
WD = RAW / "wikidata_he_vocalized_country_admin.json"
PRIOR = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"


def without_marks(value: str) -> str:
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def hash_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    official = read(CBS)
    if not official.get("success") or official["result"]["total"] != len(official["result"]["records"]):
        raise ValueError("CBS retrieval is incomplete")
    by_name = defaultdict(list)
    for row in official["result"]["records"]:
        by_name[row["שם יישוב"]].append(row)
    prior_qids = {row["subject_qid"] for row in read(PRIOR)["rows"]}
    grouped = defaultdict(set)
    reasons = Counter()
    for binding in read(WD)["results"]["bindings"]:
        qid = binding["item"]["value"].rsplit("/", 1)[-1]
        if not qid.startswith("Q") or "labelHe" not in binding:
            reasons["no_item_or_hebrew_label"] += 1
            continue
        name = binding["labelHe"]["value"]
        marked = binding["vocalized"]["value"]
        if without_marks(marked) != name or marked == name:
            reasons["name_letters_differ_or_no_marks"] += 1
            continue
        if len(by_name[name]) != 1:
            reasons["not_unique_exact_official_name"] += 1
            continue
        grouped[(qid, name)].add(marked)
    candidates = []
    for (qid, name), marked_values in sorted(grouped.items()):
        if len(marked_values) != 1:
            reasons["multiple_marked_forms"] += 1
            continue
        if qid in prior_qids:
            reasons["prior_subject_overlap"] += 1
            continue
        locality = by_name[name][0]
        if locality["שם מחוז"] == "אזור יהודה ושומרון":
            reasons["disputed_administrative_category"] += 1
            continue
        if not locality["שם נפה"] or not locality["שם מחוז"]:
            reasons["missing_answer"] += 1
            continue
        candidates.append({
            "qid": qid,
            "hebrew_unmarked": name,
            "hebrew_marked": next(iter(marked_values)),
            "english_locality_name": locality.get("שם יישוב באנגלית"),
            "cbs_locality_code": str(locality["סמל יישוב"]).zfill(4),
            "cbs_row_id": locality["_id"],
            "district_he": locality["שם מחוז"],
            "subdistrict_he": locality["שם נפה"],
            "district_code": locality["סמל מחוז"],
            "subdistrict_code": locality["סמל נפה"],
            "name_source_url": f"https://www.wikidata.org/wiki/{qid}#P4239",
            "fact_source_url": "https://data.gov.il/he/datasets/lamas/localities-in-israel/d47a54ff-87f0-44b3-b33a-f284c0c38e5a",
            "verification_status": "candidate_pending_Wikidata_P3466_code_and_pointing_source_checks",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hebrew_cbs_candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates),
        encoding="utf-8",
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url_cbs": "https://data.gov.il/api/3/action/datastore_search?resource_id=d47a54ff-87f0-44b3-b33a-f284c0c38e5a&limit=10000",
        "source_url_marked_names": "https://www.wikidata.org/wiki/Property:P4239",
        "source_sha256": {str(path.relative_to(ROOT)): hash_of(path) for path in (CBS, WD, PRIOR)},
        "cbs_rows": len(official["result"]["records"]),
        "wikidata_bindings": len(read(WD)["results"]["bindings"]),
        "candidate_subjects": len(candidates),
        "unique_subdistrict_answers": len({row["subdistrict_he"] for row in candidates}),
        "unique_district_answers": len({row["district_he"] for row in candidates}),
        "excluded_binding_or_entity_reasons": dict(reasons),
        "caution": "CBS fact and undiacritized-name fields are official; P4239 marked strings are Wikidata contributions and are not independently pronunciation-verified. These rows are candidates, not final gold.",
    }
    (OUT / "filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("candidate_subjects", "unique_subdistrict_answers", "unique_district_answers", "excluded_binding_or_entity_reasons")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

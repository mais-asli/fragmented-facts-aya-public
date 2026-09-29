"""Find explicitly pointed locality names in Hebrew Wikipedia article leads.

CBS provides the facts and unpointed names. A separate code-ID check follows.
No model predictions are used in filtering.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import unicodedata
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
CBS = RAW / "israel_cbs_localities_2023.json"
BATCH_DIR = RAW / "all_official_locality_hewiki_lead_batches"
OUT = RAW / "all_official_locality_pointed_leads.json"
MARK_PATTERN = r"[\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]*"
HEBREW_LETTER = re.compile(r"[\u05D0-\u05EA]")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def get(url: str, destination: Path) -> dict:
    for attempt in range(6):
        process = subprocess.run(
            ["powershell.exe", "-NoProfile", "-File",
             str(ROOT / "scripts/fetch_public_json_20260928.ps1"),
             "-Url", url, "-OutFile", str(destination)],
            capture_output=True, timeout=80,
        )
        if process.returncode == 0:
            return read(destination)
        message = process.stderr.decode("utf-8", errors="replace")[-800:]
        if attempt == 5:
            raise RuntimeError(message)
        time.sleep(60 + 20 * attempt if "too many requests" in message.lower() else 2**attempt)
    raise AssertionError("unreachable")


def pointed_occurrence(unmarked: str, lead: str) -> tuple[str | None, int, int]:
    pattern = "".join(re.escape(char) + MARK_PATTERN if HEBREW_LETTER.fullmatch(char)
                      else re.escape(char) for char in unmarked)
    normalized = unicodedata.normalize("NFC", lead[:350])
    for match in re.finditer(pattern, normalized):
        before = normalized[match.start() - 1] if match.start() else ""
        after = normalized[match.end()] if match.end() < len(normalized) else ""
        if HEBREW_LETTER.fullmatch(before) or HEBREW_LETTER.fullmatch(after):
            continue
        value = match.group()
        marks = sum(unicodedata.category(char) == "Mn" for char in value)
        letters = sum(bool(HEBREW_LETTER.fullmatch(char)) for char in value)
        if marks >= (2 if letters >= 4 else 1):
            return value, marks, match.start()
    return None, 0, -1


def main() -> None:
    source = read(CBS)
    if not source["success"] or source["result"]["total"] != len(source["result"]["records"]):
        raise ValueError("Official CBS table is incomplete")
    by_name = defaultdict(list)
    for row in source["result"]["records"]:
        if row["שם מחוז"] != "אזור יהודה ושומרון" and row["שם נפה"]:
            by_name[row["שם יישוב"]].append(row)
    official = {name: rows[0] for name, rows in by_name.items() if len(rows) == 1}
    titles = sorted(official)
    batches = [titles[index:index + 20] for index in range(0, len(titles), 20)]
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    checked = []
    reasons = Counter()
    for index, batch in enumerate(batches):
        path = BATCH_DIR / f"{index:03d}.json"
        if path.exists():
            saved = read(path)
            if saved["requested_titles"] != batch:
                raise ValueError("Cached official-title batch differs")
        else:
            url = "https://he.wikipedia.org/w/api.php?" + urlencode({
                "action": "query", "format": "json", "formatversion": "2",
                "prop": "extracts|pageprops", "ppprop": "wikibase_item",
                "explaintext": "1", "exintro": "1", "exlimit": "max",
                "titles": "|".join(batch), "redirects": "1",
            })
            saved = {"requested_titles": batch, "source_url": url,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "response": get(url, BATCH_DIR / "current_api_response.json")}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            time.sleep(2)
        response = saved["response"]
        title_map = {name: name for name in batch}
        for transformation in [*response.get("query", {}).get("normalized", []),
                               *response.get("query", {}).get("redirects", [])]:
            for original, resolved in list(title_map.items()):
                if resolved == transformation["from"]:
                    title_map[original] = transformation["to"]
        pages = {page["title"]: page for page in response.get("query", {}).get("pages", [])}
        for name in batch:
            page = pages.get(title_map[name], {})
            qid = page.get("pageprops", {}).get("wikibase_item")
            lead = page.get("extract", "")
            if not qid or not lead:
                reasons["no_article_qid_or_intro"] += 1
                continue
            pointed, marks, start = pointed_occurrence(name, lead)
            if not pointed:
                reasons["no_exact_letter_pointed_form_in_lead"] += 1
                continue
            cbs = official[name]
            checked.append({
                "qid": qid, "cbs_locality_code": str(cbs["סמל יישוב"]).zfill(4),
                "cbs_row_id": cbs["_id"], "hebrew_unmarked": name,
                "hebrew_marked": pointed, "marked_name_count": marks,
                "marked_name_offset_in_lead": start,
                "article_title": page["title"],
                "article_url": "https://he.wikipedia.org/wiki/" + quote(page["title"]),
                "article_lead_excerpt": unicodedata.normalize("NFC", lead[:350]),
                "subdistrict_he": cbs["שם נפה"],
                "district_he": cbs["שם מחוז"],
                "subdistrict_code": cbs["סמל נפה"],
                "district_code": cbs["סמל מחוז"],
                "english_locality_name": cbs.get("שם יישוב באנגלית"),
                "fact_source_url": "https://data.gov.il/he/datasets/lamas/localities-in-israel/d47a54ff-87f0-44b3-b33a-f284c0c38e5a",
                "verification_status": "candidate_pending_P3466_code_match",
            })
        print(f"Hebrew official-name batch {index + 1}/{len(batches)} resolved", flush=True)
    result = {
        "source_sha256": hashlib.sha256(CBS.read_bytes()).hexdigest(),
        "source_official_rows": len(source["result"]["records"]),
        "unique_non_West_Bank_named_rows": len(official),
        "lead_pointing_candidate_rows": len(checked),
        "unique_qids": len({row["qid"] for row in checked}),
        "excluded_reasons": dict(reasons), "rows": checked,
        "caution": "Names are attested in Wikipedia article leads, not independently certified by a linguist. Fact gold comes from CBS; entity identity requires a P3466 locality-code match. Golan rows are still present at this precheck stage and are excluded downstream.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("unique_non_West_Bank_named_rows", "lead_pointing_candidate_rows", "unique_qids")}), flush=True)


if __name__ == "__main__":
    main()

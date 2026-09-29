"""Check each CAMeL name against Wikidata's Arabic entity label and sitelink.

Matching labels establish entity identity only, not fact truth or pronunciation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
LINKS = RAW / "camelprop_enwiki_qid_links.json"
BATCH_DIR = RAW / "wikidata_label_batches"
OUT = RAW / "camelprop_entity_label_checks.json"


def get_json(url: str, destination: Path) -> dict:
    for attempt in range(6):
        process = subprocess.run(
            ["powershell.exe", "-NoProfile", "-File",
             str(ROOT / "scripts/fetch_public_json_20260928.ps1"),
             "-Url", url, "-OutFile", str(destination)],
            capture_output=True, timeout=80)
        if process.returncode == 0:
            return json.loads(destination.read_text(encoding="utf-8-sig"))
        message = process.stderr.decode("utf-8", errors="replace")[-1000:]
        if attempt == 5:
            raise RuntimeError(message)
        time.sleep(60 + 20 * attempt if "too many requests" in message.lower() else 2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    source = json.loads(LINKS.read_text(encoding="utf-8"))
    qids = sorted({row["wikidata_qid"] for row in source["rows"]
                   if row["wikidata_qid"] and row["wikidata_qid"].startswith("Q")},
                  key=lambda q: int(q[1:]))
    batches = [qids[i:i + 50] for i in range(0, len(qids), 50)]
    entities = {}
    for index, batch in enumerate(batches):
        path = BATCH_DIR / f"{index:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["requested_qids"] != batch:
                raise ValueError("Cached QID batch differs from source mapping")
        else:
            params = urlencode({"action": "wbgetentities", "format": "json",
                                "ids": "|".join(batch), "props": "labels|sitelinks",
                                "languages": "ar|he|en", "sitefilter": "arwiki|hewiki|enwiki",
                                "formatversion": "2"})
            url = "https://www.wikidata.org/w/api.php?" + params
            saved = {"requested_qids": batch, "source_url": url,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "response": get_json(url, BATCH_DIR / "current_api_response.json")}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            time.sleep(8)
        response = saved["response"]
        if "error" in response:
            raise ValueError(f"Wikidata API error batch {index}: {response['error']}")
        for entity in response["entities"].values():
            if entity.get("id"):
                entities[entity["id"]] = entity
        print(f"label batch {index + 1}/{len(batches)} resolved", flush=True)
    checked = []
    for row in source["rows"]:
        qid = row["wikidata_qid"]
        entity = entities.get(qid, {})
        labels = {lang: entity.get("labels", {}).get(lang, {}).get("value")
                  for lang in ("ar", "he", "en")}
        sitelinks = {lang: entity.get("sitelinks", {}).get(lang + "wiki", {}).get("title")
                     for lang in ("ar", "he", "en")}
        exact_label = row["arabic_U"] == labels["ar"]
        exact_arwiki = row["arabic_U"] == sitelinks["ar"]
        checked.append({**row, "wikidata_labels": labels, "wikipedia_titles": sitelinks,
                        "arabic_identity_match": exact_label or exact_arwiki,
                        "match_basis": "arabic_label" if exact_label else
                                       "arwiki_title" if exact_arwiki else None})
    result = {"source_sha256": hashlib.sha256(LINKS.read_bytes()).hexdigest(),
              "method": "CAMeL U must exactly equal the Arabic Wikidata label or Arabic Wikipedia title of the enwiki-resolved QID; identity check only",
              "source_rows": len(checked), "unique_qids": len(qids),
              "identity_matched_rows": sum(r["arabic_identity_match"] for r in checked),
              "rows": checked}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("source_rows", "unique_qids", "identity_matched_rows")}), flush=True)


if __name__ == "__main__":
    main()

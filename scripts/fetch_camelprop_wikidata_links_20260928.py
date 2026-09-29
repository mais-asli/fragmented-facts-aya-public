"""Resolve the published CAMeL Arabic proper-name glosses to exact enwiki items.

This is source discovery, not fact or spelling approval. Every raw API batch is
cached so the mapping is reproducible and can resume after a network failure.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
CAMEL = RAW / "CamelPROPWIKID3K.tsv"
BATCH_DIR = RAW / "enwiki_api_batches"
OUT = RAW / "camelprop_enwiki_qid_links.json"
AR_MARKS = re.compile(r"[\u064b-\u065f\u0670]")
AR_LETTERS = re.compile(r"[\u0621-\u063a\u0641-\u064a]")
LATIN = re.compile(r"[A-Za-z]")


def get_json(url: str) -> dict:
    # This Windows host trusts the university network certificate through the
    # OS store; its bundled Python CA store does not. Keep TLS verification on.
    temporary = BATCH_DIR / "current_api_response.json"
    for attempt in range(6):
        try:
            process = subprocess.run(["powershell.exe", "-NoProfile", "-File",
                                      str(ROOT / "scripts/fetch_public_json_20260928.ps1"),
                                      "-Url", url, "-OutFile", str(temporary)],
                                     capture_output=True, timeout=60)
            if process.returncode:
                raise RuntimeError(process.stderr.decode("utf-8", errors="replace")[-1000:])
            return json.loads(temporary.read_text(encoding="utf-8-sig"))
        except Exception as error:
            if attempt == 5:
                raise
            if "too many requests" in str(error).lower():
                time.sleep(60 + 20 * attempt)
            else:
                time.sleep(2**attempt)
    raise AssertionError("unreachable")


def eligible_rows() -> list[dict]:
    rows = []
    with CAMEL.open(encoding="utf-8-sig", newline="") as source:
        for index, row in enumerate(csv.DictReader(source, delimiter="\t"), start=2):
            u = row["Arabic Word "].strip()
            d = row["Diacritized Reference"].strip()
            gloss = row["Gloss"].strip()
            if (AR_MARKS.sub("", d) == u and AR_MARKS.search(d)
                    and len(AR_LETTERS.findall(u)) >= 2 and not LATIN.search(u)
                    and gloss and len(gloss) < 120):
                rows.append({"source_line": index, "arabic_U": u, "arabic_D": d,
                             "english_gloss": gloss})
    return rows


def main() -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    rows = eligible_rows()
    titles = sorted({row["english_gloss"] for row in rows})
    batches = [titles[i:i + 40] for i in range(0, len(titles), 40)]
    all_pages = []
    for index, batch in enumerate(batches):
        path = BATCH_DIR / f"{index:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["requested_titles"] != batch:
                raise ValueError(f"Cached batch {index} does not match input")
        else:
            params = urlencode({"action": "query", "format": "json", "prop": "pageprops",
                                "ppprop": "wikibase_item", "redirects": "1",
                                "titles": "|".join(batch), "formatversion": "2"})
            url = "https://en.wikipedia.org/w/api.php?" + params
            saved = {"requested_titles": batch, "source_url": url,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "response": get_json(url)}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            time.sleep(8)
        response = saved["response"]
        if "error" in response or "query" not in response:
            raise ValueError(f"API error in batch {index}: {response.get('error')}")
        normalized = {item["from"]: item["to"]
                      for item in response["query"].get("normalized", [])}
        redirects = {item["from"]: item["to"]
                     for item in response["query"].get("redirects", [])}
        by_title = {page["title"]: page for page in response["query"]["pages"]}
        for title in batch:
            resolved = redirects.get(normalized.get(title, title), normalized.get(title, title))
            page = by_title.get(resolved, {})
            all_pages.append({"gloss": title, "resolved_title": resolved,
                              "wikidata_qid": page.get("pageprops", {}).get("wikibase_item"),
                              "missing": page.get("missing", False),
                              "batch": index})
        print(f"batch {index+1}/{len(batches)} resolved", flush=True)
    link = {item["gloss"]: item for item in all_pages}
    for row in rows:
        row.update(link[row["english_gloss"]])
    result = {"source": "https://github.com/CAMeL-Lab/CamelProp",
              "source_sha256": hashlib.sha256(CAMEL.read_bytes()).hexdigest(),
              "method": "Exact English gloss enwiki title lookup; redirects recorded; Arabic U/D must differ only by marks. No fact is approved by this mapping.",
              "eligible_source_rows": len(rows), "unique_glosses": len(titles),
              "linked_rows": sum(bool(row["wikidata_qid"]) for row in rows),
              "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("eligible_source_rows", "unique_glosses", "linked_rows")}), flush=True)


if __name__ == "__main__":
    main()

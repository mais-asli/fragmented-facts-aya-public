"""Cross-check Wikidata pointed localities against Hebrew Wikipedia article leads."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
SOURCE = RAW / "hebrew_locality_claim_checks.json"
OUT = RAW / "hebrew_locality_wikipedia_pointing_checks.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = [row for row in source["rows"] if row["cbs_code_exact_match"] and row["hewiki_title"]]
    titles = sorted({row["hewiki_title"] for row in rows})
    pages = {}
    for index in range(0, len(titles), 20):
        batch = titles[index:index + 20]
        url = "https://he.wikipedia.org/w/api.php?" + urlencode({
            "action": "query", "format": "json", "formatversion": "2",
            "prop": "extracts", "explaintext": "1", "exintro": "1",
            "exlimit": "max", "titles": "|".join(batch), "redirects": "1",
        })
        path = RAW / f"hewiki_locality_leads_batch_{index // 20:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8-sig"))
            if saved["requested_titles"] != batch:
                raise ValueError("Cached article-title batch differs")
        else:
            response_path = RAW / "hewiki_locality_current_response.json"
            process = subprocess.run([
                "powershell.exe", "-NoProfile", "-File",
                str(ROOT / "scripts/fetch_public_json_20260928.ps1"),
                "-Url", url, "-OutFile", str(response_path)],
                capture_output=True, timeout=100,
            )
            if process.returncode:
                raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
            saved = {"requested_titles": batch, "source_url": url,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "response": json.loads(response_path.read_text(encoding="utf-8-sig"))}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        response = saved["response"]
        redirects = {record["from"]: record["to"]
                     for record in response.get("query", {}).get("redirects", [])}
        for page in response.get("query", {}).get("pages", []):
            pages[page["title"]] = page
        for title in batch:
            if title in redirects:
                pages[title] = pages.get(redirects[title], {})
    checked = []
    for row in rows:
        page = pages.get(row["hewiki_title"], {})
        extract = unicodedata.normalize("NFC", page.get("extract", ""))
        marked = unicodedata.normalize("NFC", row["hebrew_marked"])
        match = marked in extract[:250]
        checked.append({"qid": row["qid"], "hebrew_unmarked": row["hebrew_unmarked"],
                        "hebrew_marked": row["hebrew_marked"],
                        "cbs_code_exact_match": row["cbs_code_exact_match"],
                        "article_title": row["hewiki_title"],
                        "article_url": "https://he.wikipedia.org/wiki/" + quote(row["hewiki_title"]),
                        "first_250_chars": extract[:250],
                        "pointed_form_in_article_lead": match})
    result = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "checked_rows": len(checked),
              "lead_exact_matches": sum(row["pointed_form_in_article_lead"] for row in checked),
              "caution": "Wikipedia and Wikidata belong to the same ecosystem; this is a second displayed attestation, not independent expert adjudication.",
              "rows": checked}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("checked_rows", "lead_exact_matches")}))


if __name__ == "__main__":
    main()

"""Resolve Wikidata country QIDs to ISO-3166 alpha-2 for GeoNames checks."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
SOURCE = ROOT / "data/candidates/camelprop_geonames_20260928/arabic_city_country_precheck.jsonl"
OUT = RAW / "wikidata_country_iso_crosswalk.json"


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    countries = sorted({row["wikidata_country_qid"] for row in rows}, key=lambda q: int(q[1:]))
    values = " ".join("wd:" + qid for qid in countries)
    query = "SELECT ?country ?iso WHERE { VALUES ?country { " + values + " } ?country wdt:P297 ?iso. }"
    url = "https://query.wikidata.org/sparql?" + urlencode({"format": "json", "query": query})
    response_path = RAW / "wikidata_country_iso_query_response.json"
    process = subprocess.run(
        ["powershell.exe", "-NoProfile", "-File",
         str(ROOT / "scripts/fetch_public_json_20260928.ps1"),
         "-Url", url, "-OutFile", str(response_path)],
        capture_output=True, timeout=100,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    response = json.loads(response_path.read_text(encoding="utf-8-sig"))
    codes = {}
    for binding in response["results"]["bindings"]:
        qid = binding["country"]["value"].rsplit("/", 1)[-1]
        codes.setdefault(qid, set()).add(binding["iso"]["value"])
    result = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "query_url": url, "requested_country_qids": countries,
        "iso_codes": {qid: sorted(values) for qid, values in codes.items()},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"requested_countries": len(countries), "resolved": len(codes)}), flush=True)


if __name__ == "__main__":
    main()

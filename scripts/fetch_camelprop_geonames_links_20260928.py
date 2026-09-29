"""Retrieve GeoNames IDs for entity-checked CAMeL proper-name candidates.

This fetches identifiers and optional Wikidata countries, not factual gold.
Cached queries and source hashes make exclusions auditable.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
IDENTITY = RAW / "camelprop_entity_label_checks.json"
PRIOR = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
BATCH_DIR = RAW / "camelprop_geonames_batches"
OUT = RAW / "camelprop_geonames_links.json"


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


def main() -> None:
    source = read(IDENTITY)
    prior_qids = {row["subject_qid"] for row in read(PRIOR)["rows"]}
    by_arabic = defaultdict(set)
    for row in source["rows"]:
        if row["match_basis"] == "arabic_label":
            by_arabic[row["arabic_U"]].add((row["wikidata_qid"], row["arabic_D"]))
    reasons = Counter()
    eligible = []
    for row in source["rows"]:
        if row["match_basis"] != "arabic_label":
            reasons["no_exact_arabic_label_match"] += 1
            continue
        if row["wikipedia_titles"]["en"] != row["english_gloss"]:
            reasons["english_title_differs_from_gloss"] += 1
            continue
        if len(row["arabic_U"].replace(" ", "")) < 5:
            reasons["very_short_name"] += 1
            continue
        if len(by_arabic[row["arabic_U"]]) != 1:
            reasons["ambiguous_arabic_name_or_vocalization"] += 1
            continue
        if row["wikidata_qid"] in prior_qids:
            reasons["prior_subject_overlap"] += 1
            continue
        eligible.append(row)
    qids = sorted({row["wikidata_qid"] for row in eligible}, key=lambda q: int(q[1:]))
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batches = [qids[index:index + 100] for index in range(0, len(qids), 100)]
    geonames = defaultdict(set)
    countries = defaultdict(set)
    for index, batch in enumerate(batches):
        path = BATCH_DIR / f"{index:03d}.json"
        if path.exists():
            saved = read(path)
            if saved["requested_qids"] != batch:
                raise ValueError("Cached QID batch differs from source")
        else:
            values = " ".join("wd:" + qid for qid in batch)
            query = ("SELECT ?item ?geonames ?country WHERE { VALUES ?item { " + values +
                     " } ?item wdt:P1566 ?geonames. OPTIONAL { ?item wdt:P17 ?country. } }")
            url = "https://query.wikidata.org/sparql?" + urlencode({"format": "json", "query": query})
            saved = {"requested_qids": batch, "source_url": url,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "response": get(url, BATCH_DIR / "current_query_response.json")}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            time.sleep(5)
        for binding in saved["response"]["results"]["bindings"]:
            qid = binding["item"]["value"].rsplit("/", 1)[-1]
            geonames[qid].add(binding["geonames"]["value"])
            if "country" in binding:
                countries[qid].add(binding["country"]["value"].rsplit("/", 1)[-1])
        print(f"GeoNames batch {index + 1}/{len(batches)} resolved", flush=True)
    rows = []
    for row in eligible:
        qid = row["wikidata_qid"]
        rows.append({**row, "geonames_ids": sorted(geonames[qid]),
                     "wikidata_country_qids": sorted(countries[qid])})
    result = {
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (IDENTITY, PRIOR)},
        "source_rows": len(source["rows"]), "eligible_identity_rows": len(rows),
        "unique_eligible_qids": len(qids),
        "rows_with_geonames_id": sum(bool(row["geonames_ids"]) for row in rows),
        "rows_with_single_geonames_id": sum(len(row["geonames_ids"]) == 1 for row in rows),
        "excluded_identity_reasons": dict(reasons), "rows": rows,
        "caution": "P1566 is only a cross-dataset ID link. Fact truth requires a matching GeoNames city record and country cross-check.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("eligible_identity_rows", "rows_with_geonames_id", "rows_with_single_geonames_id")}), flush=True)


if __name__ == "__main__":
    main()

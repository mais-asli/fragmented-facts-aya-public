"""Profile independently linked Arabic city-to-country fact candidates.

This is pre-outcome source filtering. It does not inspect Aya generations.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
OUT = ROOT / "data/candidates/camelprop_geonames_20260928"
LINKS = RAW / "camelprop_geonames_links.json"
GEONAMES = RAW / "geonames_cities500_20260928.zip"
COUNTRIES = RAW / "geonames_countryInfo_20260928.txt"
CLDR = RAW / "cldr_ar_territories_20260928.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def norm(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().strip()


def main() -> None:
    source = read(LINKS)
    wanted = {row["geonames_ids"][0] for row in source["rows"]
              if len(row["geonames_ids"]) == 1}
    gazetteer = defaultdict(list)
    with zipfile.ZipFile(GEONAMES) as archive:
        with archive.open("cities500.txt") as records:
            for line in records:
                cells = line.decode("utf-8").rstrip("\n").split("\t")
                if cells[0] in wanted:
                    gazetteer[cells[0]].append({
                        "geonames_id": cells[0], "name": cells[1],
                        "ascii_name": cells[2], "alternate_names": cells[3].split(","),
                        "feature_class": cells[6], "feature_code": cells[7],
                        "country_iso2": cells[8], "population": int(cells[14] or "0"),
                        "last_modified": cells[18],
                    })
    country_info = {}
    for line in COUNTRIES.read_text(encoding="utf-8-sig").splitlines():
        if line and not line.startswith("#"):
            cells = line.split("\t")
            country_info[cells[0]] = cells[4]
    territories = read(CLDR)["main"]["ar"]["localeDisplayNames"]["territories"]
    counts = Counter()
    rows = []
    for row in source["rows"]:
        if len(row["geonames_ids"]) != 1:
            counts["no_unique_geonames_id"] += 1
            continue
        matches = gazetteer[row["geonames_ids"][0]]
        if len(matches) != 1:
            counts["not_one_cities500_record"] += 1
            continue
        geo = matches[0]
        if geo["feature_class"] != "P":
            counts["not_populated_place"] += 1
            continue
        if norm(row["english_gloss"]) not in {norm(name) for name in
                                               [geo["name"], geo["ascii_name"], *geo["alternate_names"]]}:
            counts["english_name_not_in_geonames"] += 1
            continue
        if not geo["country_iso2"] or geo["country_iso2"] not in country_info:
            counts["no_geonames_country"] += 1
            continue
        if geo["country_iso2"] not in territories:
            counts["no_cldr_arabic_country_name"] += 1
            continue
        if geo["country_iso2"] in {"PS", "XK", "TW", "HK", "MO"}:
            counts["disputed_or_special_geographic_granularity"] += 1
            continue
        if len(row["wikidata_country_qids"]) != 1:
            counts["no_unique_wikidata_country"] += 1
            continue
        rows.append({
            "subject_qid": row["wikidata_qid"],
            "subject_en": row["english_gloss"],
            "subject_ar_U": row["arabic_U"],
            "subject_ar_D": row["arabic_D"],
            "geonames_id": geo["geonames_id"],
            "geonames_name": geo["name"],
            "geonames_feature_code": geo["feature_code"],
            "geonames_population": geo["population"],
            "geonames_last_modified": geo["last_modified"],
            "country_iso2": geo["country_iso2"],
            "country_en": country_info[geo["country_iso2"]],
            "country_ar": territories[geo["country_iso2"]],
            "wikidata_country_qid": row["wikidata_country_qids"][0],
            "name_source_line": row["source_line"],
            "name_source_url": "https://github.com/CAMeL-Lab/CamelProp/blob/main/CamelPROPWIKID3K.tsv",
            "fact_source_url": "https://www.geonames.org/" + geo["geonames_id"],
            "entity_source_url": "https://www.wikidata.org/wiki/" + row["wikidata_qid"],
            "verification_status": "candidate_pending_Wikidata_country_ISO_crosscheck",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arabic_city_country_precheck.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (LINKS, GEONAMES, COUNTRIES, CLDR)},
        "identity_checked_rows": source["eligible_identity_rows"],
        "precheck_fact_rows": len(rows),
        "unique_countries": len({row["country_iso2"] for row in rows}),
        "country_distribution": dict(Counter(row["country_iso2"] for row in rows).most_common()),
        "filter_exclusions": dict(counts),
        "caution": "Country truth and pronunciation have separate sources; country QID/ISO agreement is still pending. GeoNames is CC BY 4.0, and source attribution is required.",
    }
    (OUT / "precheck_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in
                      ("identity_checked_rows", "precheck_fact_rows", "unique_countries", "filter_exclusions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

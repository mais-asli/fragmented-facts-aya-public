"""Freeze an Aya-unseen Arabic city-to-country cohort from cross-checked sources."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
PRECHECK = ROOT / "data/candidates/camelprop_geonames_20260928/arabic_city_country_precheck.jsonl"
COUNTRY_CODES = RAW / "wikidata_country_iso_crosswalk.json"
GEONAMES = RAW / "geonames_cities500_20260928.zip"
CAMELPROP = RAW / "CamelPROPWIKID3K.tsv"
CLDR = RAW / "cldr_ar_territories_20260928.json"
OUT = ROOT / "data/curated/aya-external-geonames-arabic-20260928.jsonl"
MANIFEST = ROOT / "configs/aya_external_geonames_arabic_cohort_20260928.json"
EXCLUDED_COUNTRY_CODES = {"GB", "RU", "UA"}  # constituent-country and currently contested-country granularity


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def norm(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().strip()


def without_marks(value: str) -> str:
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in PRECHECK.read_text(encoding="utf-8").splitlines()]
    codes = read(COUNTRY_CODES)["iso_codes"]
    territories = read(CLDR)["main"]["ar"]["localeDisplayNames"]["territories"]
    reasons = Counter()
    verified = []
    for row in rows:
        if codes.get(row["wikidata_country_qid"]) != [row["country_iso2"]]:
            reasons["wikidata_geonames_country_code_disagree_or_missing"] += 1
            continue
        if row["country_iso2"] in EXCLUDED_COUNTRY_CODES:
            reasons["excluded_country_granularity_or_border_risk"] += 1
            continue
        if without_marks(row["subject_ar_D"]) != row["subject_ar_U"]:
            reasons["marked_unmarked_letters_differ"] += 1
            continue
        verified.append(row)
    selected_names = {norm(row["subject_en"]) for row in verified} | {
        norm(row["subject_ar_U"]) for row in verified}
    name_countries = defaultdict(set)
    with zipfile.ZipFile(GEONAMES) as archive:
        with archive.open("cities500.txt") as stream:
            for line in stream:
                cells = line.decode("utf-8").rstrip("\n").split("\t")
                country = cells[8]
                for name in (cells[1], cells[2], *cells[3].split(",")):
                    normalized = norm(name)
                    if normalized in selected_names:
                        name_countries[normalized].add(country)
    final = []
    for row in verified:
        if any(name_countries[norm(name)] != {row["country_iso2"]}
               for name in (row["subject_en"], row["subject_ar_U"])
               if name_countries[norm(name)]):
            reasons["toponym_matches_geonames_city_in_other_country"] += 1
            continue
        country = row["country_iso2"]
        aliases = [territories[country]]
        aliases += [value for key, value in territories.items()
                    if key.startswith(country + "-alt-") and value not in aliases]
        final.append({
            "fact_id": f"{row['subject_qid']}-GeoNames-country-{country}",
            "subject_qid": row["subject_qid"],
            "subject_en": row["subject_en"],
            "subject_ar_U": row["subject_ar_U"],
            "subject_ar_D": row["subject_ar_D"],
            "relation": "city_in_country",
            "country_iso2": country,
            "country_wikidata_qid": row["wikidata_country_qid"],
            "answer_ar": row["country_ar"],
            "answer_ar_aliases_cldr": aliases,
            "answer_en": row["country_en"],
            "geonames_id": row["geonames_id"],
            "geonames_population": row["geonames_population"],
            "source": {
                "name_annotated": row["name_source_url"],
                "name_source_line": row["name_source_line"],
                "entity": row["entity_source_url"],
                "independent_country_fact": row["fact_source_url"],
                "answer_label": "https://github.com/unicode-org/cldr-json/blob/main/cldr-json/cldr-localenames-full/main/ar/territories.json",
                "checks": ["Arabic U equals Wikidata Arabic label",
                           "English gloss equals Wikipedia article title",
                           "Wikidata P1566 equals GeoNames city ID",
                           "GeoNames ISO country equals Wikidata P17 country P297",
                           "English place name appears in GeoNames name or aliases",
                           "Arabic D differs from U only by combining marks",
                           "No exact GeoNames cross-country toponym collision"],
            },
            "verification_status": "cross_source_checked_no_independent_speaker_review",
            "split": "external_prospective_test",
        })
    final.sort(key=lambda row: row["subject_qid"])
    if len({row["subject_qid"] for row in final}) != len(final):
        raise ValueError("Repeated subject QID")
    country_info = {}
    for line in (RAW / "geonames_countryInfo_20260928.txt").read_text(encoding="utf-8-sig").splitlines():
        if line and not line.startswith("#"):
            cells = line.split("\t")
            country_info[cells[0]] = {"continent": cells[8],
                                      "neighbors": set(cells[17].split(","))}
    represented = {row["country_iso2"] for row in final}
    for row in final:
        code = row["country_iso2"]
        same_continent = {other for other in represented - {code}
                          if country_info[other]["continent"] == country_info[code]["continent"]}
        neighboring = same_continent & country_info[code]["neighbors"]
        choices = neighboring or same_continent or (represented - {code})
        distractor = min(choices, key=lambda other: (
            abs(len(territories[other]) - len(row["answer_ar"])), other))
        row["distractor_country_iso2"] = distractor
        row["distractor_answer_ar"] = territories[distractor]
        row["distractor_selection"] = ("GeoNames neighboring country in cohort" if neighboring
                                      else "same-continent country in cohort" if same_continent
                                      else "other country in cohort")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final),
                   encoding="utf-8")
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_before_aya_outcomes": True,
        "cohort": "Arabic marked proper place names and city-to-country facts",
        "size": len(final),
        "unique_country_answers": len({row["country_iso2"] for row in final}),
        "country_distribution": dict(Counter(row["country_iso2"] for row in final).most_common()),
        "distractor_rule": "Choose another represented country, preferring a GeoNames-listed neighbor, else same continent, then closest Arabic answer character length and ISO code. Frozen before Aya outputs.",
        "excluded_reasons_after_266_precheck": dict(reasons),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path)
                          for path in (PRECHECK, COUNTRY_CODES, GEONAMES, CAMELPROP, CLDR)},
        "cohort_sha256": sha(OUT),
        "source_note": "CAMeL CP-WIKI-D3K manually annotated name markings; GeoNames CC BY 4.0 for place-country facts; Wikidata entity and code cross-check; Unicode CLDR for Arabic country spellings.",
        "limitations": ["Source agreement does not guarantee all facts or names are error-free.",
                        "No independent speaker judged this new cohort.",
                        "One-word place names and the country relation differ from earlier cohorts.",
                        "Current GeoNames snapshot can postdate Aya's training cutoff."],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("size", "unique_country_answers", "excluded_reasons_after_266_precheck")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

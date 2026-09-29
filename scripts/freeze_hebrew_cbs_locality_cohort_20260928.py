"""Freeze source-checked Hebrew locality-to-subdistrict Aya test rows."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import unicodedata
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
SOURCE = RAW / "all_official_locality_pointed_leads.json"
PRIOR = ROOT / "configs/e6_position_decomposition_protocol_20260928.json"
BATCH_DIR = RAW / "all_official_locality_claim_batches"
OUT = ROOT / "data/curated/aya-external-cbs-hebrew-20260928.jsonl"
MANIFEST = ROOT / "configs/aya_external_cbs_hebrew_cohort_20260928.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_marks(value: str) -> str:
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def get(url: str, destination: Path) -> dict:
    for attempt in range(6):
        process = subprocess.run([
            "powershell.exe", "-NoProfile", "-File",
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
    source = read(SOURCE)
    prior = {row["subject_qid"] for row in read(PRIOR)["rows"]}
    qids = sorted({row["qid"] for row in source["rows"]}, key=lambda q: int(q[1:]))
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    entities = {}
    batches = [qids[index:index + 40] for index in range(0, len(qids), 40)]
    for index, batch in enumerate(batches):
        path = BATCH_DIR / f"{index:03d}.json"
        if path.exists():
            saved = read(path)
            if saved["requested_qids"] != batch:
                raise ValueError("Cached locality-claim batch differs")
        else:
            url = "https://www.wikidata.org/w/api.php?" + urlencode({
                "action": "wbgetentities", "format": "json", "formatversion": "2",
                "ids": "|".join(batch), "props": "claims",
            })
            saved = {"requested_qids": batch, "source_url": url,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "response": get(url, BATCH_DIR / "current_api_response.json")}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            time.sleep(2)
        entities.update(saved["response"]["entities"])
        print(f"Hebrew code batch {index + 1}/{len(batches)} resolved", flush=True)
    reasons = Counter()
    by_qid = defaultdict(list)
    by_name = defaultdict(list)
    for row in source["rows"]:
        by_qid[row["qid"]].append(row)
        by_name[row["hebrew_unmarked"]].append(row)
    selected = []
    for row in source["rows"]:
        if len(by_qid[row["qid"]]) != 1 or len(by_name[row["hebrew_unmarked"]]) != 1:
            reasons["duplicate_qid_or_unmarked_name"] += 1
            continue
        if row["subdistrict_he"] == "גולן":
            reasons["disputed_administrative_classification"] += 1
            continue
        if row["subdistrict_he"] in row["hebrew_unmarked"]:
            reasons["answer_name_cued_in_subject"] += 1
            continue
        if row["qid"] in prior:
            reasons["prior_subject_overlap"] += 1
            continue
        if without_marks(row["hebrew_marked"]) != row["hebrew_unmarked"]:
            reasons["marked_unmarked_letters_differ"] += 1
            continue
        entity = entities.get(row["qid"], {})
        codes = {str(statement["mainsnak"].get("datavalue", {}).get("value", "")).zfill(4)
                 for statement in entity.get("claims", {}).get("P3466", [])
                 if statement.get("mainsnak", {}).get("snaktype") == "value"}
        if row["cbs_locality_code"] not in codes:
            reasons["wikidata_official_locality_code_disagrees_or_missing"] += 1
            continue
        selected.append(row)
    pool_size = len(selected)
    pool_path = ROOT / "data/candidates/official_localities_20260928/hebrew_cbs_source_verified_pool.jsonl"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in sorted(selected, key=lambda item: item["qid"])),
                         encoding="utf-8")
    # The test sample is balanced over answer subdistricts and fixed before Aya output.
    by_subdistrict = defaultdict(list)
    for row in selected:
        by_subdistrict[row["subdistrict_he"]].append(row)
    for code, group in by_subdistrict.items():
        group.sort(key=lambda item: hashlib.sha256(
            ("CBS2023-UD-v1|" + item["qid"]).encode("utf-8")).hexdigest())
    selected = []
    while len(selected) < min(240, pool_size):
        progress = False
        for answer_name in sorted(by_subdistrict):
            if by_subdistrict[answer_name] and len(selected) < 240:
                selected.append(by_subdistrict[answer_name].pop(0))
                progress = True
        if not progress:
            break
    if len(selected) != min(240, pool_size):
        raise ValueError("Deterministic subdistrict sample incomplete")
    represented = defaultdict(set)
    for row in selected:
        represented[row["district_he"]].add((row["subdistrict_he"], row["subdistrict_code"]))
    all_answers = {(row["subdistrict_he"], row["subdistrict_code"]) for row in selected}
    final = []
    for row in selected:
        gold = (row["subdistrict_he"], row["subdistrict_code"])
        same_district = {item for item in represented[row["district_he"]]
                         if item[0] != gold[0]}
        choices = same_district or {item for item in all_answers if item[0] != gold[0]}
        if not choices:
            raise ValueError("Cannot select a different subdistrict distractor")
        distractor = min(choices, key=lambda item: (
            abs(len(item[0]) - len(gold[0])), str(item[1])))
        final.append({
            "fact_id": f"{row['qid']}-CBS-subdistrict-{row['subdistrict_code']}",
            "subject_qid": row["qid"],
            "subject_en": row["english_locality_name"],
            "subject_he_U": row["hebrew_unmarked"],
            "subject_he_D": row["hebrew_marked"],
            "marked_name_count": row["marked_name_count"],
            "relation": "locality_in_subdistrict",
            "answer_he": row["subdistrict_he"],
            "answer_code": row["subdistrict_code"],
            "distractor_answer_he": distractor[0],
            "distractor_answer_code": distractor[1],
            "distractor_selection": ("different represented subdistrict in same district"
                                     if same_district else "different represented subdistrict"),
            "cbs_locality_code": row["cbs_locality_code"],
            "cbs_district_he": row["district_he"],
            "source": {
                "pointed_name": row["article_url"],
                "pointed_name_excerpt": row["article_lead_excerpt"],
                "official_fact": row["fact_source_url"],
                "wikidata_identity": "https://www.wikidata.org/wiki/" + row["qid"],
                "checks": ["Marked form appears in article lead",
                           "Removing marks exactly yields unique official locality name",
                           "Wikidata P3466 code equals official locality code",
                           "Official 2023 CBS subdistrict is nonempty",
                           "Prior Aya subject QID absent"],
            },
            "verification_status": "cross_source_checked_no_independent_speaker_review",
            "split": "external_prospective_test",
        })
    final.sort(key=lambda row: row["subject_qid"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final),
                   encoding="utf-8")
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_before_aya_outcomes": True,
        "cohort": "Hebrew marked locality names and CBS subdistrict facts",
        "source_lead_candidates": len(source["rows"]),
        "source_verified_pool_size": pool_size,
        "source_verified_pool_sha256": sha(pool_path),
        "sampling_rule": "Deterministic SHA-256 ordering within official subdistrict answer name, round-robin across distinct answer names, up to 240 subjects; fixed before Aya output.",
        "size": len(final),
        "unique_subdistrict_answers": len({row["answer_he"] for row in final}),
        "subdistrict_distribution": dict(Counter(row["answer_he"] for row in final).most_common()),
        "excluded_reasons": dict(reasons),
        "distractor_rule": "Choose a different represented subdistrict answer name in the same district if possible, minimizing Hebrew character-length difference; otherwise a different represented subdistrict name. Frozen before Aya outputs.",
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (SOURCE, PRIOR)},
        "claims_batch_sha256": {path.name: sha(path) for path in sorted(BATCH_DIR.glob("[0-9][0-9][0-9].json"))},
        "cohort_sha256": sha(OUT),
        "limitations": ["Pointing is attested in Wikipedia leads, not independently certified by a linguist.",
                        "Wikidata and Wikipedia are related sources; CBS independently supplies the fact and locality code.",
                        "One-country locality panel differs from prior person and organization relations.",
                        "Current Wikipedia leads can postdate Aya's training cutoff."],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("source_lead_candidates", "source_verified_pool_size", "size", "unique_subdistrict_answers", "excluded_reasons")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

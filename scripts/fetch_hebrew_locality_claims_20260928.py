"""Check the Wikidata locality code and pointed-name claim provenance."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_names_20260928"
SOURCE = ROOT / "data/candidates/official_localities_20260928/hebrew_cbs_candidates.jsonl"
OUT = RAW / "hebrew_locality_claim_checks.json"


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    qids = sorted({row["qid"] for row in rows}, key=lambda q: int(q[1:]))
    entities = {}
    queries = []
    for index in range(0, len(qids), 40):
        batch = qids[index:index + 40]
        url = "https://www.wikidata.org/w/api.php?" + urlencode({
            "action": "wbgetentities", "format": "json", "formatversion": "2",
            "ids": "|".join(batch), "props": "claims|sitelinks",
            "sitefilter": "hewiki|enwiki",
        })
        path = RAW / f"hebrew_locality_claims_batch_{index // 40:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8-sig"))
            if saved["requested_qids"] != batch:
                raise ValueError("Cached Hebrew claim batch differs")
        else:
            response_path = RAW / "hebrew_claims_current_response.json"
            process = subprocess.run([
                "powershell.exe", "-NoProfile", "-File",
                str(ROOT / "scripts/fetch_public_json_20260928.ps1"),
                "-Url", url, "-OutFile", str(response_path)],
                capture_output=True, timeout=100,
            )
            if process.returncode:
                raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
            saved = {"requested_qids": batch, "source_url": url,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "response": json.loads(response_path.read_text(encoding="utf-8-sig"))}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        entities.update(saved["response"]["entities"])
    checked = []
    for row in rows:
        entity = entities.get(row["qid"], {})
        claims = entity.get("claims", {})
        codes = {str(statement["mainsnak"].get("datavalue", {}).get("value", "")).zfill(4)
                 for statement in claims.get("P3466", [])
                 if statement.get("mainsnak", {}).get("snaktype") == "value"}
        marked_claims = []
        for statement in claims.get("P4239", []):
            value = statement["mainsnak"].get("datavalue", {}).get("value")
            if isinstance(value, dict) and value.get("language") == "he":
                marked_claims.append({"text": value.get("text"),
                                      "reference_count": len(statement.get("references", [])),
                                      "reference_urls": [snak.get("datavalue", {}).get("value")
                                                         for reference in statement.get("references", [])
                                                         for snak in reference.get("snaks", {}).get("P854", [])]})
        checked.append({**row, "wikidata_cbs_codes": sorted(codes),
                        "cbs_code_exact_match": row["cbs_locality_code"] in codes,
                        "marked_claim_exact_match": any(claim["text"] == row["hebrew_marked"]
                                                        for claim in marked_claims),
                        "marked_claims": marked_claims,
                        "hewiki_title": entity.get("sitelinks", {}).get("hewiki", {}).get("title")})
    result = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "rows": checked, "candidate_rows": len(checked),
              "code_matched": sum(row["cbs_code_exact_match"] for row in checked),
              "marked_claim_matched": sum(row["marked_claim_exact_match"] for row in checked),
              "both_matched": sum(row["cbs_code_exact_match"] and row["marked_claim_exact_match"]
                                  for row in checked)}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("candidate_rows", "code_matched", "marked_claim_matched", "both_matched")}))


if __name__ == "__main__":
    main()

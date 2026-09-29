"""Fetch official ACL style and bibliography metadata, recording source hashes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import urllib.request
from fragmented_facts.io import file_hash, now, write_json
from fragmented_facts.sources import USER_AGENT, open_url

ROOT = Path(__file__).resolve().parents[1] / "paper"
PAPERS = ["2023.emnlp-main.751", "2025.findings-acl.827", "2026.findings-eacl.22",
          "2025.acl-long.253", "2021.eacl-main.284"]


def fetch(url, target):
    with open_url(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        data = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"url": url, "path": str(target), "sha256": file_hash(target), "retrieved_at": now()}


if __name__ == "__main__":
    records = []
    for name in ("acl.sty", "acl_natbib.bst", "acl_latex.tex", "acl_lualatex.tex", "README.md"):
        target = ROOT / ("official_" + name if name in ("README.md", "acl_latex.tex", "acl_lualatex.tex") else name)
        records.append(fetch("https://raw.githubusercontent.com/acl-org/acl-style-files/master/" + name, target))
    for paper in PAPERS:
        records.append(fetch(f"https://aclanthology.org/{paper}.bib", ROOT / "bib" / f"{paper}.bib"))
    combined = "\n".join((ROOT / "bib" / f"{paper}.bib").read_text(encoding="utf-8") for paper in PAPERS)
    (ROOT / "references.bib").write_text(combined, encoding="utf-8")
    write_json(ROOT / "source_manifest.json", records)
    print(f"Fetched {len(records)} official style/bibliography assets")

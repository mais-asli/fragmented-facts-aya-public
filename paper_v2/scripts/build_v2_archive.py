"""Build the reproducibility archive for the v2 paper (pilot + held-out + external cohorts).

Run from anywhere:  python "nlp final/paper_v2/scripts/build_v2_archive.py"
Writes nlp final/output/Fragmented_Facts_Reproducibility_20260927.zip and checks every member
against the SHA-256 manifest. Earlier drafts (paper/, paper_refined/ PDF), output/ work logs,
model weights, the 72 MB public LAMA download, duplicate run tarballs and cluster connection
receipts are excluded.
"""
from __future__ import annotations
import hashlib, json, os, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "output" / "Fragmented_Facts_Reproducibility_20260927.zip"
FINAL_PDF = ROOT / "paper_v2" / "Fragmented_Facts_Final_Paper_v2.pdf"
INCLUDE_DIRS = (
    "paper_v2", "src", "scripts", "slurm", "configs", "tests", "project_plan/feasibility",
    "data/curated", "data/review", "data/external/wikimedia_2023", "data/external/wikimedia_2023_google_re",
    "results/tau_research_20260926", "results/tau_mechanism_20260926", "results/heldout", "results/h27",
    "results/b27", "results/e3p27", "results/e427", "results/e4c27", "results/gre27",
    "outputs/01a09b51/aya_output_audit_20260926", "outputs/01a09b51/ReviewerA_review_20260926",
    "outputs/01a09b51/reviewed_tokenization_20260926",
)
INCLUDE_FILES = (
    "docs/ANNOTATION_GUIDE.md", "docs/PILOT_ANALYSIS_PROTOCOL_20260926.md", "docs/MECHANISM_PILOT_PROTOCOL_20260926.md",
    "docs/AYA_FEASIBILITY_RESULT.md", "docs/IMPLEMENTATION_NOTES.md", "docs/RUNBOOK.md", "docs/DOCKER.md",
    "docs/SLURM_AYA_CHECK.md", "docs/AYA_ONLY_HELDOUT_EXTENSION_20260927.md", "docs/AYA_HELDOUT_E3_E5_PLAN_20260927.md",
    "docs/AYA_HELDOUT_SOURCE_QA_20260927.md", "outputs/01a09b51/Aya_Independent_Test_Review_20260927.xlsx",
    "model_snapshot.json", "pyproject.toml", "requirements-common.txt", "requirements-cluster.txt",
    "requirements-cluster-validated.txt", "requirements-windows-tested.txt", "Dockerfile", "ff.py",
)
EXCLUDE_SUFFIXES = {".pyc", ".png", ".log", ".aux", ".bbl", ".blg", ".out", ".synctex.gz", ".tmp", ".tar.gz", ".gz"}
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".cache", "_to_delete", "node_modules"}
EXCLUDE_NAMES = {"main.pdf"}
def excluded_name(name: str) -> bool:
    return name.startswith(("tau-", "tau_")) and name.endswith(".json")   # cluster connection/transfer receipts

README = """# Fragmented Facts: reproducibility archive (27 September 2026)

Start with paper_v2/Fragmented_Facts_Final_Paper_v2.pdf and paper_v2/README.md.

- paper_v2/: LaTeX source, fonts, figures, exact PDF; scripts/consolidate_v2.py (raw records ->
  paper_v2/_work/*.jsonl), compute_v2.py and recompute_v2*.py (every new estimate from the records),
  make_figures_v2.py (all figures); _work/ holds the consolidated per-prompt records.
- results/tau_research_20260926, results/tau_mechanism_20260926: pilot per-prompt records (57 facts).
- results/heldout (E1/E2), results/h27 (E3/E5), results/b27 and results/e3p27 (E3 variants),
  results/e4c27 and results/e427 (English E4 and the failed development screen), results/gre27
  (Google-RE external E1/E2): per-prompt records with protocols, manifests and completion records.
- results/*.json, *.csv: analysis outputs, sensitivity analyses, audits and 2023 pageviews.
- data/curated, data/review: reviewed facts, templates, review sheets, reconciliations and freezes.
- data/external/wikimedia_2023*: raw pageview API responses. The public LAMA 2019 archive used to draw
  the Google-RE cohort is not included (see data/review/google_re_external_20260927 for its hash).
- src/, scripts/, configs/, slurm/, tests/, project_plan/feasibility/, docs/: code and protocols.

Model weights are not included (CohereLabs/aya-23-8B, revision 89da1a0ed02d6130f93ae0ffdbedb63b760c0471).
SUBMISSION_MANIFEST.json lists the SHA-256 of every other file in the archive.
"""
def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def selected_files() -> list[Path]:
    files: list[Path] = []
    for folder in INCLUDE_DIRS:
        base = ROOT / folder
        if not base.is_dir(): raise FileNotFoundError(base)
        for dirpath, dirnames, filenames in os.walk(base, onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_PARTS]
            for fn in filenames:
                path = Path(dirpath) / fn
                try:
                    is_file = path.is_file()
                except OSError:
                    continue
                if (is_file and not any(fn.lower().endswith(s) for s in EXCLUDE_SUFFIXES)
                        and fn not in EXCLUDE_NAMES and not excluded_name(fn)):
                    files.append(path)
    for name in INCLUDE_FILES:
        path = ROOT / name
        if not path.is_file(): raise FileNotFoundError(path)
        files.append(path)
    for pattern in ("*.json", "*.csv"):
        for path in (ROOT / "results").glob(pattern):
            if not excluded_name(path.name): files.append(path)
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())
def main() -> None:
    files = selected_files(); rel = [p.relative_to(ROOT).as_posix() for p in files]
    assert "paper_v2/Fragmented_Facts_Final_Paper_v2.pdf" in rel, "final PDF missing"
    assert not any(r.startswith(("paper/", "output/", "paper_refined/")) for r in rel), "earlier draft included"
    DEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    readme = README.encode("utf-8")
    manifest.append({"path": "README_ARCHIVE.md", "bytes": len(readme), "sha256": sha(readme)})
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        z.writestr("README_ARCHIVE.md", readme)
        for path, r in zip(files, rel):
            content = path.read_bytes(); z.writestr(r, content)
            manifest.append({"path": r, "bytes": len(content), "sha256": sha(content)})
        payload = {"title": "Fragmented Facts: Optional Diacritics, Subword Fragmentation, and Factual Recall in Hebrew and Arabic",
                   "created": "2026-09-27", "paper_pdf": "paper_v2/Fragmented_Facts_Final_Paper_v2.pdf",
                   "paper_pdf_sha256": sha(FINAL_PDF.read_bytes()), "model": "CohereLabs/aya-23-8B",
                   "revision": "89da1a0ed02d6130f93ae0ffdbedb63b760c0471",
                   "cohorts": {"pilot": 57, "heldout": 40, "external_google_re": 26},
                   "excluded": "earlier drafts, output/ work logs, model weights, LAMA 2019 download, run tarballs, connection receipts",
                   "files": manifest}
        z.writestr("SUBMISSION_MANIFEST.json", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    with zipfile.ZipFile(DEST) as z:
        if z.testzip() is not None: raise RuntimeError("corrupt member")
        for item in manifest:
            if sha(z.read(item["path"])) != item["sha256"]: raise RuntimeError("hash mismatch: " + item["path"])
        names = z.namelist()
    unlisted = sorted(set(names) - {i["path"] for i in manifest} - {"SUBMISSION_MANIFEST.json"})
    if unlisted: raise RuntimeError("members missing from manifest: " + ", ".join(unlisted[:10]))
    counts = {k: sum(n.startswith(k) for n in names) for k in ("paper_v2/", "results/", "data/", "src/")}
    print(json.dumps({"archive": str(DEST), "members": len(names), "bytes": DEST.stat().st_size,
                      "sha256": sha(DEST.read_bytes()), "paper_pdf_sha256": payload["paper_pdf_sha256"], "counts": counts}, indent=2))
if __name__ == "__main__":
    main()

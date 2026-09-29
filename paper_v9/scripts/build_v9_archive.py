"""Build the reproducibility archive for the final paper (version 9).

Run from anywhere:  python "nlp final/paper_v9/scripts/build_v9_archive.py"
Writes nlp final/output/Fragmented_Facts_Public_Reproducibility_v9.zip and checks every member
against the SHA-256 manifest. Earlier drafts (paper/, paper_refined/ PDF), output/ work logs,
model weights, the 72 MB public LAMA download, duplicate run tarballs and cluster connection
receipts are excluded.
"""
from __future__ import annotations
import hashlib, json, os, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "output" / "Fragmented_Facts_Public_Reproducibility_v9.zip"
FINAL_PDF = ROOT / "paper_v9" / "Fragmented_Facts_Final_Paper_v9.pdf"
INCLUDE_DIRS = (
    "paper_v9", "paper_v2/_work", "paper_v2/scripts", "src", "scripts", "slurm", "configs", "tests", "project_plan/feasibility", "data/curated", "data/review", "data/external/wikimedia_2023", "data/external/wikimedia_2023_google_re",
    "results/tau_research_20260926", "results/tau_mechanism_20260926", "results/heldout", "results/h27",
    "results/b27", "results/e3p27", "results/e427", "results/e4c27", "results/gre27", "results/figures",
    "outputs/01a09b51/aya_output_audit_20260926", "outputs/01a09b51/ReviewerA_review_20260926",
    "outputs/01a09b51/reviewed_tokenization_20260926",
)
INCLUDE_FILES = (
    "results/aya-e6-position-decomposition-20260928.tar.gz", "results/aya-e7-same-count-20260928.tar.gz",
    "results/aya-source-verified-external-20260928.tar.gz", "results/aya-source-verified-e5-20260928.tar.gz",
    "results/AYA_20260928_RESEARCH_HANDOFF.md", "results/AYA_20260928_RUN_STATUS.md", "results/AYA_RUNTIME_ERRATUM_20260928.md",
    "results/SOURCE_VERIFIED_COHORT_AUDIT_20260928.md", "results/AYA_E6_POSITION_DECOMPOSITION_RESULTS_20260928.md",
    "results/AYA_E7_SAME_COUNT_RESULTS_20260928.md", "results/AYA_SOURCE_VERIFIED_EXTERNAL_RESULTS_20260928.md",
    "results/AYA_SOURCE_VERIFIED_E5_RESULTS_20260928.md",
    "docs/ANNOTATION_GUIDE.md", "docs/PILOT_ANALYSIS_PROTOCOL_20260926.md", "docs/MECHANISM_PILOT_PROTOCOL_20260926.md",
    "docs/AYA_FEASIBILITY_RESULT.md", "docs/IMPLEMENTATION_NOTES.md", "docs/RUNBOOK.md", "docs/DOCKER.md",
    "docs/SLURM_AYA_CHECK.md", "docs/AYA_ONLY_HELDOUT_EXTENSION_20260927.md", "docs/AYA_HELDOUT_E3_E5_PLAN_20260927.md",
    "docs/AYA_HELDOUT_SOURCE_QA_20260927.md", "outputs/01a09b51/Aya_Independent_Test_Review_20260927.xlsx",
    "REPRODUCIBILITY.md", "pyproject.toml", "requirements-common.txt", "requirements-cluster.txt",
    "requirements-cluster-validated.txt", "requirements-windows-tested.txt", "Dockerfile", "ff.py",
)
EXCLUDE_SUFFIXES = {".pyc", ".png", ".log", ".aux", ".bbl", ".blg", ".out", ".synctex.gz", ".tmp", ".tar.gz", ".gz"}
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".cache", "_to_delete", "node_modules"}
EXCLUDE_NAMES = {"main.pdf"}
def excluded_name(name: str) -> bool:
    return name.startswith(("tau-", "tau_")) and name.endswith(".json")   # cluster connection/transfer receipts

README = """# Lost in Vocalization: reproducibility archive, final paper (version 9)

Start with paper_v9/Fragmented_Facts_Final_Paper_v9.pdf. REPRODUCIBILITY.md explains how to
verify these files, recompute every analysis, number and figure, and rerun Aya.
Repository: https://github.com/mais-asli/fragmented-facts-aya-public

- paper_v9/: LaTeX source, fonts, figures, exact PDF; scripts/ turn the per-prompt records into
  paper_v9/_work (consolidate_*.py), every reported number (compute_*.py) and every figure
  (make_figures_v3.py). paper_v2/_work and paper_v2/scripts are used by the post hoc margin analysis.
- results/tau_research_20260926, results/tau_mechanism_20260926: pilot per-prompt records (57 facts).
- results/heldout (E1/E2), results/h27 (E3/E5), results/b27 and results/e3p27 (E3 variants),
  results/e4c27 and results/e427 (English E4 and the failed development screen), results/gre27
  (Google-RE E1/E2): per-prompt records with protocols, manifests and completion records.
- results/aya-*-20260928.tar.gz: checksum-verified run archives of the source-checked cohorts (E1/E2),
  the source-disjoint E5, E6 and E7, with their frozen protocols, runners and completion records;
  results/*_20260928.md: result summaries, the cohort source audit and the runtime erratum.
- results/*.json, *.csv: analysis outputs, sensitivity analyses, audits, the 424-row generation audit
  of the source-checked cohorts and 2023 pageviews.
- data/curated, data/review: reviewed and source-checked facts with source links, templates, review
  sheets, reconciliations and freezes.
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
    assert "paper_v9/Fragmented_Facts_Final_Paper_v9.pdf" in rel, "final PDF missing"
    assert "REPRODUCIBILITY.md" in rel, "reproducibility guide missing"
    assert not any(r.startswith(("paper/", "output/", "paper_refined/", "paper_v3/", "paper_v4/", "paper_v5/", "paper_v6/", "paper_v7/", "paper_v8/")) for r in rel), "earlier draft included"
    assert all(r.startswith(("paper_v2/_work/", "paper_v2/scripts/")) for r in rel if r.startswith("paper_v2/")), "paper_v2 draft included"
    DEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    readme = README.encode("utf-8")
    manifest.append({"path": "README_ARCHIVE.md", "bytes": len(readme), "sha256": sha(readme)})
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        z.writestr("README_ARCHIVE.md", readme)
        for path, r in zip(files, rel):
            content = path.read_bytes(); z.writestr(r, content)
            manifest.append({"path": r, "bytes": len(content), "sha256": sha(content)})
        payload = {"title": "Lost in Vocalization: Optional Diacritics Disrupt Hebrew and Arabic Factual Recall in Aya-23-8B",
                   "created": "2026-09-29", "paper_pdf": "paper_v9/Fragmented_Facts_Final_Paper_v9.pdf",
                   "paper_pdf_sha256": sha(FINAL_PDF.read_bytes()), "model": "CohereLabs/aya-23-8B",
                   "revision": "89da1a0ed02d6130f93ae0ffdbedb63b760c0471",
                   "cohorts": {"pilot": 57, "heldout": 40, "external_google_re": 26, "arabic_cities_source_checked": 184, "hebrew_localities_source_checked": 240},
                   "excluded": "earlier drafts, output/ work logs, model weights and the local model manifest, LAMA 2019 download, pre-28-September run tarballs (included unpacked), connection receipts",
                   "files": manifest}
        z.writestr("SUBMISSION_MANIFEST.json", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    with zipfile.ZipFile(DEST) as z:
        if z.testzip() is not None: raise RuntimeError("corrupt member")
        for item in manifest:
            if sha(z.read(item["path"])) != item["sha256"]: raise RuntimeError("hash mismatch: " + item["path"])
        names = z.namelist()
    unlisted = sorted(set(names) - {i["path"] for i in manifest} - {"SUBMISSION_MANIFEST.json"})
    if unlisted: raise RuntimeError("members missing from manifest: " + ", ".join(unlisted[:10]))
    counts = {k: sum(n.startswith(k) for n in names) for k in ("paper_v9/", "paper_v2/", "results/", "data/", "src/")}
    print(json.dumps({"archive": str(DEST), "members": len(names), "bytes": DEST.stat().st_size,
                      "sha256": sha(DEST.read_bytes()), "paper_pdf_sha256": payload["paper_pdf_sha256"], "counts": counts}, indent=2))
if __name__ == "__main__":
    main()

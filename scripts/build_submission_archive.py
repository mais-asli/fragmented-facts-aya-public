"""Package the Aya pilot paper and its inspectable research evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output" / "Fragmented_Facts_Aya_Pilot_Reproducibility_20260926.zip"
INCLUDE_DIRS = (
    "paper",
    "src",
    "scripts",
    "slurm",
    "configs",
    "project_plan/feasibility",
    "data/curated",
    "data/review/imported_ReviewerA_20260926",
    "results/tau_research_20260926",
    "results/tau_mechanism_20260926",
    "outputs/01a09b51/aya_output_audit_20260926",
    "outputs/01a09b51/ReviewerA_review_20260926",
    "docs",
    "tests",
    "output/pdf",
)
EXCLUDE_SUFFIXES = {".pyc", ".png", ".log", ".aux", ".bbl", ".blg"}
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".cache"}


def selected_files(heldout: bool = False) -> list[Path]:
    files: list[Path] = []
    folders = INCLUDE_DIRS + ((
        "data/review/heldout_batch_v1",
        "results/heldout",
    ) if heldout else ())
    for folder in folders:
        base = ROOT / folder
        if not base.is_dir():
            raise FileNotFoundError(base)
        for path in base.rglob("*"):
            if path.is_file() and not any(part in EXCLUDE_PARTS for part in path.parts):
                if path.suffix.lower() not in EXCLUDE_SUFFIXES:
                    files.append(path)
    for path in (ROOT / "results").glob("*.json"):
        files.append(path)
    if heldout:
        for name in (
            "ff.py", "README.md", "pyproject.toml", "model_snapshot.json",
            "requirements-cluster.txt", "requirements-common.txt",
            "data/raw/candidates_final.jsonl", "data/review/subject_splits.json",
            "outputs/01a09b51/Aya_Independent_Test_Review_20260927.xlsx",
        ):
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", action="store_true",
                        help="Include the independently reviewed Aya test and its outputs")
    args = parser.parse_args()
    destination = (ROOT / "output/Fragmented_Facts_Aya_Heldout_Reproducibility_20260927.zip"
                   if args.heldout else DEST)
    files = selected_files(args.heldout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            archive.writestr(relative, content)
            manifest.append({"path": relative, "bytes": len(content),
                             "sha256": hashlib.sha256(content).hexdigest()})
        payload = {
            "title": "Fragmented Facts: Aya-23-8B reviewed pilot and held-out test" if args.heldout
                     else "Fragmented Facts: Aya-23-8B reviewed pilot",
            "created": "2026-09-27" if args.heldout else "2026-09-26",
            "scope": "Actual TAU outputs, reviewed data, analysis code, paper source and PDF; model weights excluded",
            "model": "CohereLabs/aya-23-8B",
            "revision": "89da1a0ed02d6130f93ae0ffdbedb63b760c0471",
            "pilot_only": not args.heldout,
            "files": manifest,
        }
        archive.writestr("SUBMISSION_MANIFEST.json",
                         json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt archive member: {bad}")
        for item in manifest:
            stored = archive.read(str(item["path"]))
            if hashlib.sha256(stored).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Hash mismatch: {item['path']}")
    print(json.dumps({"archive": str(destination), "files": len(manifest),
                      "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()

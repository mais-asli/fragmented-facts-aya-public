"""Extract the 28 September Aya records into compact JSONL files in paper_v3/_work.

Reads the four checksum-verified run archives in <nlp final>/results and writes
  sv_ud.jsonl  source-checked Arabic city and Hebrew locality cohorts (424 subjects, U/D, t1+t2)
  sv_e5.jsonl  source-disjoint fixed-site E5 (192 subjects)
  e6.jsonl     E6 decomposition, conditions A/B/C/D (246 fact-language pairs)
  e7.jsonl     E7 same-text forced splits, conditions mid/early/late (the same 246 pairs)
Set FF_ROOT to the project root if this folder is not <nlp final>/paper_v3.
"""
import json, os, tarfile, io
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FF_ROOT", HERE.parent.parent))
W = HERE.parent / "_work"; W.mkdir(exist_ok=True)
ARCH = {"sv_ud": "aya-source-verified-external-20260928.tar.gz", "sv_e5": "aya-source-verified-e5-20260928.tar.gz",
        "e6": "aya-e6-position-decomposition-20260928.tar.gz", "e7": "aya-e7-same-count-20260928.tar.gz"}
DROP = {"runtime"}
for name, arch in ARCH.items():
    rows = []
    with tarfile.open(ROOT / "results" / arch) as tf:
        for m in tf.getmembers():
            if m.isfile() and "/items/" in m.name and m.name.endswith(".json"):
                r = json.load(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8"))
                rows.append({k: v for k, v in r.items() if k not in DROP})
    rows.sort(key=lambda r: json.dumps(r["key"], sort_keys=True, ensure_ascii=False))
    with open(W / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))

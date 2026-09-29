# Lost in Vocalization: final paper, version 9 (29 September 2026)

The named submission PDF is deliberately omitted from this public copy. The
anonymized `main.tex` records the paper source; it has eight pages of main text,
followed by references and appendices A-E when compiled with the original
layout. The privately submitted paper retains the real author details.

Repository: https://github.com/mais-asli/fragmented-facts-aya-public
Reproduction guide: `../REPRODUCIBILITY.md` (also at the root of the repository and of the
reproducibility archive).

Changes from v8: the E5 site is named "the state at the subject's last token" throughout, and
the prediction position is defined where E4 is introduced; "investigated" replaces "answered"
for the three research questions; short explanations of cloze probing, morphemes, very low
baseline accuracy, the hierarchical bootstrap and unadjusted intervals; "cohorts by language"
replaces "cells"; a clearer abstract sentence on the letter-boundary split; the E6 paragraph
introduces A-D one at a time before the three steps; the limitations separate partial marking
from source dates; the E7 result states that its same-count figure is the average of the
early and late splits; a paragraph break inside the introduction's final sentence removed. No number
changed. Scripts: `consolidate_records.py` and `compute_accuracy.py` (pilot records and numbers,
from the first paper version) added; unused v2 figure script and files removed.

## Build

XeLaTeX is required for Hebrew niqqud and Arabic tashkil. Fonts are bundled in
`fonts/` (David Libre and Amiri, SIL Open Font License); the main font is TeX Gyre
Termes (shipped with TeX Live).

    xelatex main
    bibtex main
    xelatex main
    xelatex main

On Overleaf, set the compiler to XeLaTeX and upload this folder as is.

## Numbers and figures

From the per-prompt records in `../results/` (run from this folder; see
`../REPRODUCIBILITY.md`, Section 6):

    python scripts/consolidate_records.py  # pilot runs -> _work/e1, e2, e3, e5a, e5b, screen.jsonl
    python scripts/consolidate_v2.py       # held-out, Google-RE, E3 variants, E4 -> _work/ho_*, gre_*, e4c, e4dev
    python scripts/consolidate_v3.py       # 28 September archives -> _work/sv_ud, sv_e5, e6, e7
    python scripts/compute_accuracy.py     # pilot numbers -> _work/computed.json
    python scripts/compute_v2.py           # held-out and Google-RE -> _work/computed_v2.json
    python scripts/compute_v3.py           # source-checked, E5 replication, E6, E7 -> _work/computed_v3.json
    python scripts/make_figures_v3.py      # all figures and their plotted values

Starting from an empty `_work/`, these reproduce every file in `_work/` byte for byte.

`compute_v3.py` recomputes every new point estimate from the records and asserts
that it equals the saved analysis files exactly; it also recomputes intervals and
p-values with the same estimators (independent resampling streams). The paper
reports the saved official intervals and p-values, which the script copies into
`computed_v3.json` under `official`.

The named submission archive was built separately. The public release ZIP was
privacy-redacted and has its own file manifest; `scripts/build_v9_archive.py`
is a historical script for the private original and should not be used to
rebuild this public derivative.

## Provenance

The source-checked cohorts were accepted only where independent sources agree,
and every kept item was verified against its sources; the checks are stored per
row. Review materials for the reviewed cohorts were prepared with AI assistance
and verified by the authors, as the AI Disclosure describes.

**Public, privacy-redacted derivative.** The original frozen hashes and exact Aya rerun instructions below describe the private research copy. Redaction changed filenames, reviewer metadata, local paths, and archive hashes. See [PUBLIC_REDACTION_NOTICE.md](PUBLIC_REDACTION_NOTICE.md). The published checksums verify this public copy. Commands that compare against the original frozen protocol hashes may stop; the public archive is intended for inspecting code, outputs, and reported estimates, not for claiming a bitwise exact Aya rerun.

# Reproducing "Lost in Vocalization"

This guide explains how to check and reproduce the results of *Lost in Vocalization:
Optional Diacritics Disrupt Hebrew and Arabic Factual Recall in Aya-23-8B*
(author identities withheld from this public copy).

There are three levels of access to the research materials:

1. **Verify** that the saved files are intact (a few minutes, any computer).
2. **Recompute** analyses from the saved Aya outputs. The four 28 September
   analyses (source-checked E2 and E5, E6 and E7) were rerun on this redacted
   copy; their 5,912 comparable numeric values matched the private originals.
   Other historical recipes below are retained but may require the private
   original if they enforce its frozen hashes.
3. **Rerun Aya** using the private original protocols, model access and a GPU.
   This public derivative is not a bitwise exact rerun input.

The original commands and analysis provenance are retained for inspection. The privacy-redacted copies have different hashes and some original hash-gated commands require the private source archive.

---

## 1. Where the materials are

| Location | Contents |
|---|---|
| GitHub repository [public research repository](https://github.com/mais-asli/fragmented-facts-aya-public) | Code (`src/`, `ff.py`), experiment runners (`project_plan/feasibility/`), analysis scripts (`scripts/`), frozen protocols and analysis plans (`configs/`), curated cohorts and templates (`data/`), the eleven Aya run archives of 26 to 28 September (`results/*.tar.gz`) with their SHA-256 checksums (`ARCHIVES.sha256`), saved analyses and reports (`results/`), the consolidated records used by the post hoc margin analysis (`paper_v2/`), Slurm templates (`slurm/`), tests (`tests/`) and this guide. |
| Complete archive `Fragmented_Facts_Public_Reproducibility_v9.zip` (the repository's Releases page) | The same code, protocols, data, saved analyses, `paper_v2/` and guide. The four 28 September run archives are included as archives; the runs of 26 and 27 September are included **unpacked**, as per-prompt record folders in `results/` (`tau_research_20260926` and `tau_mechanism_20260926` for the pilot, whose records are not in the repository; `heldout`, `h27`, `gre27`, `b27`, `e3p27`, `e4c27`, `e427` for the others). It adds the paper source with the scripts that produce every number and figure (`paper_v9/`), the protocol files and runner inputs that exist only in the unpacked runs, the full package list of the cluster environment, and `SUBMISSION_MANIFEST.json` with the SHA-256 of every file. |

Which one you need:

| Goal | Use |
|---|---|
| Recheck the source-checked cohorts, their E5 replication, E6 and E7 | Repository |
| Rebuild every number and figure in the paper; recheck the pilot, held-out, Google-RE, E3 and E4 analyses | Complete archive |
| Rerun Aya exactly as originally frozen | Private original protocols and data, model access and a CUDA GPU; Section 8 documents the procedure |

The model weights are not distributed. Anyone rerunning Aya needs their own access to
the model (Section 8).

---

## 2. Settings used for every Aya run

| Item | Value |
|---|---|
| Model | `CohereLabs/aya-23-8B`, revision `89da1a0ed02d6130f93ae0ffdbedb63b760c0471` (32 layers; byte-level BPE vocabulary of 256k tokens; rotary position embeddings) |
| Precision | 4-bit NF4 weights (bitsandbytes), FP16 computation, eager attention |
| Prompting and decoding | Official Aya chat template; greedy decoding, at most 24 new tokens |
| Hardware | One NVIDIA GeForce RTX 2080 Ti (11 GB) per job, on the Tel Aviv University Slurm cluster |
| Software | Python 3.12.3, PyTorch 2.8.0 (CUDA 12.6 build), Transformers 4.56.2, Accelerate 1.10.1, bitsandbytes 0.47.0, tokenizers 0.22.0. The complete package list is `requirements-cluster-validated.txt` in the complete archive. |
| Provenance in each run | Every run folder has a `manifest.json` recording the model identity, experiment, split, shard, full configuration and the hashes of the frozen protocol, code, data and templates, and a `completion.json` recording the expected and completed items. Every per-prompt record stores the library versions and the GPU name. |

Because decoding is greedy, generations do not depend on a sampling seed.

---

## 3. Verify the files

Run these checks first, or on a separate copy: the analyses in Section 5 rewrite some
saved files (with the same content but Unix line endings), after which the archive
check reports those files as changed.

**Repository** (from its root):

```bash
sha256sum -c ARCHIVES.sha256          # macOS: shasum -a 256 -c ARCHIVES.sha256
```

All eleven archives should report `OK`.

**Complete archive** (from its root):

```bash
python -c "import json,hashlib; m=json.load(open('SUBMISSION_MANIFEST.json')); bad=[f['path'] for f in m['files'] if hashlib.sha256(open(f['path'],'rb').read()).hexdigest()!=f['sha256']]; print(len(m['files']), 'files checked,', len(bad), 'mismatches')"
```

The output should end with `0 mismatches`.

Do not repackage the run archives. Each analysis records the SHA-256 of the archive it
reads, the E7 analysis checks the E6 archive's hash against its protocol, and the
bootstrap intervals depend on the order of the records inside each archive.

---

## 4. Set up an analysis environment (CPU only)

Python 3.11 or newer. From the root of the repository or of the complete archive:

```bash
python -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
python -m pip install -r requirements-common.txt
python -m pip install -e ".[test]"
pytest -q                                # 45 tests should pass
```

`requirements-common.txt` pins the versions used for the analyses (NumPy 2.2.6,
SciPy 1.16.1, pandas 2.3.2, scikit-learn 1.7.1, statsmodels 0.14.5,
Matplotlib 3.10.6). The shipped figure PDFs were written with Matplotlib 3.10.9;
with 3.10.6 the regenerated PDFs differ only in the version string and creation
date. Sections 5 and 6 need no GPU and no model.

---

## 5. Recompute the analyses from the saved outputs

### 5.1 In the repository: source-checked cohorts, E5 replication, E6 and E7

```bash
python scripts/analyze_source_verified_aya_20260928.py        # Table 3 (upper part)
python scripts/analyze_source_verified_e5_20260928.py         # Table 3 (E5 rows)
python scripts/analyze_e6_position_decomposition_20260928.py  # Table 4, Figure 2
python scripts/analyze_e7_same_count_20260928.py              # Figure 2, E7 results
```

They take about 30 s, 1 min, 15 s and 15 s on one CPU core. Each script reads its
run archive directly, records the archive's hash, checks the protocol identifiers and
the complete grid of expected records, and then rewrites its analysis file and report
in `results/`. The rewritten files have the same content as the committed ones.
Because the committed copies use Windows line endings, `git status` lists them as
modified (together with the untracked `src/fragmented_facts.egg-info/` created by the
install), and `git diff --ignore-cr-at-eol` prints nothing.

### 5.2 In the complete archive: pilot, held-out, Google-RE, E3 and E4

The pilot, held-out and Google-RE analyses use the project's `analyze` command
(seed 17, 10,000 bootstrap resamples). The run paths are recorded relative to each
study folder, so the command runs there. The pilot records are split over two
folders, which are first linked into one (on Windows, copy the folders or use WSL):

```bash
# Pilot E1-E3, and pilot E1-E3 with E5
mkdir -p _pilot/results
ln -s "$PWD"/results/tau_research_20260926/results/*-shard* \
      "$PWD"/results/tau_mechanism_20260926/results/*-shard* _pilot/results/
cd _pilot
python ../ff.py analyze --runs results/e[123]-reviewed-pilot-20260926-shard*/* \
    --output ../check_pilot.json --seed 17 --bootstrap 10000
python ../ff.py analyze --runs results/e[1235]-reviewed-pilot-20260926-shard*/* \
    --output ../check_pilot_mechanism.json --seed 17 --bootstrap 10000
cd ..

# Held-out E1 and E2, E3, and E5
(cd results/heldout && python ../../ff.py analyze \
    --runs results/e2-aya-independent-test-v1-shard*/* results/e1-aya-independent-test-v1-shard*/* \
    --output ../../check_heldout.json --seed 17 --bootstrap 10000)
(cd results/h27 && python ../../ff.py analyze \
    --runs results/e3-aya-heldout-strengthening-20260927-shard*/* \
    --output ../../check_heldout_e3.json --seed 17 --bootstrap 10000)
(cd results/h27 && python ../../ff.py analyze \
    --runs results/e5-aya-heldout-strengthening-20260927-shard*/* \
    --output ../../check_heldout_e5.json --seed 17 --bootstrap 10000)

# Google-RE E1 and E2
(cd results/gre27 && python ../../ff.py analyze \
    --runs results/e2-aya-google-re-external-20260927-shard*/* results/e1-aya-google-re-external-20260927-shard*/* \
    --output ../../check_google_re.json --seed 17 --bootstrap 10000)
```

| New file | Saved file (in `results/`) | Agreement |
|---|---|---|
| `check_pilot.json` | `tau_research_20260926/results/analysis-reviewed-pilot-20260926.json` | Identical |
| `check_pilot_mechanism.json` | `tau_mechanism_20260926/results/analysis-reviewed-pilot-with-mechanism-20260926.json` | Identical |
| `check_heldout.json` | `heldout/results/analysis-aya-independent-test-v1.json` | Equal up to floating-point rounding (largest difference about 1.3e-14) |
| `check_heldout_e3.json` | `h27/results/analysis-e3-aya-heldout-strengthening-20260927.json` | Identical |
| `check_heldout_e5.json` | `h27/results/analysis-e5-aya-heldout-strengthening-20260927.json` | Identical |
| `check_google_re.json` | `gre27/results/analysis-aya-google-re-external-20260927.json` | Identical |

The remaining analyses rewrite their saved files in `results/` with the same content
(apart from line endings, as in Section 5.1):

```bash
python scripts/analyze_e3_byte_level_20260927.py          # Appendix D: byte-level splits
python scripts/analyze_e3_position_controls_20260927.py   # Appendix D: position controls
python scripts/analyze_e4_reviewed_cohorts_20260927.py    # Figure 3c: E4
python scripts/analyze_posthoc_margin_20260928.py         # paper §5.2 and Appendix E: post hoc margins
```

The four scripts of Section 5.1 also run unchanged in the complete archive.

---

## 6. Rebuild the paper's numbers and figures (complete archive)

```bash
cd paper_v9
python scripts/consolidate_records.py  # pilot runs          -> _work/e1, e2, e3, e5a, e5b, screen.jsonl
python scripts/consolidate_v2.py       # held-out, Google-RE, E3 variants, E4 -> _work/ho_*, gre_*, e4c, e4dev
python scripts/consolidate_v3.py       # 28 September archives -> _work/sv_ud, sv_e5, e6, e7
python scripts/compute_accuracy.py     # pilot numbers     -> _work/computed.json
python scripts/compute_v2.py           # held-out and Google-RE numbers -> _work/computed_v2.json
python scripts/compute_v3.py           # source-checked, E5 replication, E6, E7 -> _work/computed_v3.json (about 2 min)
python scripts/make_figures_v3.py      # figures/*.pdf and the plotted values
```

Starting from an empty `_work/` folder, these commands reproduce every shipped file
in `_work/` byte for byte, including the values plotted in the figures
(`forest_v3.json`, `e5_v3.json`, `segmentation_v3.json`). `compute_v3.py` also
asserts that each point estimate equals the saved analysis files. The paper reports
the intervals and p-values of the saved analyses, which `compute_v3.py` copies under
`official`; the intervals it recomputes itself serve as an additional check.
`recompute_v2.py` and `recompute_v2b.py` print independent recomputations of the
held-out and Google-RE estimates for comparison with `computed_v2.json`. The figures
use the Liberation Serif font when it is installed.

To compile the paper, see `paper_v9/README.md` (XeLaTeX; the Hebrew and Arabic
fonts are bundled).

### Where each result comes from

Section numbers such as §5.1 refer to the paper. "Archive" is a run archive in the repository's `results/`; "folder" is an unpacked run
folder in the complete archive's `results/`. Analysis files are in `results/` unless a
path says otherwise.

| Paper item | Experiment | Aya outputs | Analysis file | Produced by |
|---|---|---|---|---|
| Figure 1 (right), Table 2 (pilot), paper §5.1 | E1, E2, pilot | Folder `tau_research_20260926` (complete archive only) | `tau_research_20260926/results/analysis-reviewed-pilot-20260926.json`; `paper_v9/_work/computed.json` | `ff.py analyze`; `compute_accuracy.py` |
| Table 2 (held-out) | E1, E2 | Archive `aya-independent-test-v1-research.tar.gz`; folder `heldout` | `heldout/results/analysis-aya-independent-test-v1.json`; `paper_v9/_work/computed_v2.json` | `ff.py analyze`; `compute_v2.py` |
| Table 2 (Google-RE) | E1, E2 | Archive `aya-google-re-external-20260927.tar.gz`; folder `gre27` | `gre27/results/analysis-aya-google-re-external-20260927.json`; `paper_v9/_work/computed_v2.json` | `ff.py analyze`; `compute_v2.py` |
| Table 3, Figure 1 (right) | E2 and E5, source-checked cohorts | Archives `aya-source-verified-external-20260928.tar.gz`, `aya-source-verified-e5-20260928.tar.gz` | `aya-source-verified-external-analysis-20260928.json`, `aya-source-verified-e5-analysis-20260928.json` | `analyze_source_verified_aya_20260928.py`, `analyze_source_verified_e5_20260928.py` |
| Table 4, Figure 2, Appendix E | E6, E7 | Archives `aya-e6-position-decomposition-20260928.tar.gz`, `aya-e7-same-count-20260928.tar.gz` | `aya-e6-position-decomposition-analysis-20260928.json`, `aya-e7-same-count-analysis-20260928.json`; `paper_v9/_work/computed_v3.json` | `analyze_e6_...`, `analyze_e7_...`; `compute_v3.py` |
| Figure 3a,b, E5 table in Appendix E | E5 | Pilot: folder `tau_mechanism_20260926` (complete archive only); held-out: archive `aya-strengthening-20260927.tar.gz`, folder `h27`; source-checked (diamonds): as for Table 3 | `tau_mechanism_20260926/results/analysis-reviewed-pilot-with-mechanism-20260926.json`; `h27/results/analysis-e5-aya-heldout-strengthening-20260927.json`; `aya-source-verified-e5-analysis-20260928.json` | `ff.py analyze`; `analyze_source_verified_e5_20260928.py` |
| Figure 3c | E4 | Archive `aya-e4-reviewed-cohorts-20260927.tar.gz`; folder `e4c27` | `e4-reviewed-cohorts-analysis-20260927.json` | `analyze_e4_reviewed_cohorts_20260927.py` |
| Appendix D | E3 | Pilot: folder `tau_research_20260926`; held-out: archive `aya-strengthening-20260927.tar.gz`, folder `h27`; byte-level: archive `aya-e3-byte-level-20260927.tar.gz`, folder `b27`; position controls: archive `aya-e3-position-controls-20260927.tar.gz`, folder `e3p27` | `tau_research_20260926/results/analysis-reviewed-pilot-20260926.json`; `h27/results/analysis-e3-aya-heldout-strengthening-20260927.json`; `e3-byte-level-40-subject-analysis-20260927.json`; `e3-position-controls-analysis-20260927.json` | `ff.py analyze`; `analyze_e3_...` |
| Post hoc margins (paper §5.2, Appendix E) | E2, reviewed cohorts | As for Table 2 | `posthoc_gold_distractor_margin_20260928.json` | `analyze_posthoc_margin_20260928.py` |
| Figure 4 (Appendix E) | E2, reviewed cohorts | As for Table 2 | Answer-format, screening-subgroup, answer-cue and strict-source sensitivity files in `results/`; plotted values in `paper_v9/_work/forest_v3.json` | `make_figures_v3.py` |
| Output audits (paper §5.5) | E2 outputs | As above | `ReviewerA-output-audit-analysis-20260926.json`, `heldout-human-output-audit-20260927.json`, `aya-source-verified-output-audit-20260928.csv` | Saved audit results |
| Frequency matching (Appendix E) | E1, E2 | As above | `fragmentation-pageviews-2023-match-*.json`, `google-re-pageviews-2023-match-analysis-20260927.json`; pageviews in `pageviews-2023-*.csv` | Saved analyses |

---

## 7. Seeds and consistency checks

**Seeds and resampling.**

- Reviewed cohorts (pilot, held-out, Google-RE): subject-cluster bootstrap with
  10,000 resamples and paired sign-flip tests, seed 17.
- Source-checked cohorts and their E5 replication: answer-stratum hierarchical
  bootstrap with 10,000 replicates (strata, then subjects within each stratum).
  Sign-flip tests enumerate all flips when there are at most 18 strata, and
  otherwise use 200,000 Monte Carlo flips with a plus-one correction. Base seed 1729,
  with fixed offsets per contrast set in the scripts.
- E6 and E7: answer-cluster bootstrap within relations, 10,000 replicates, seed 1729
  plus the cell index.
- All seeds are written in the scripts and analysis plans, so repeated runs give the
  same intervals and p-values.

**Baseline consistency.** Later runs recompute the unpatched scores of earlier runs,
and these agree exactly:

| Check | Scores compared | Largest difference | Record |
|---|---|---|---|
| Pilot E5 baselines vs. pilot E2 (template 1) | 228 prompt pairs | 0.0 nats | `results/e5-baseline-consistency-20260926.json` (complete archive) |
| Held-out E5 baselines vs. held-out E2 | 160 prompt pairs | 0.0 nats | `results/heldout-e5-baseline-consistency-20260927.json` (complete archive) |
| Source-checked E5 baselines vs. source-checked E2 | 768 scores (384 per cohort) | 0.0 nats (tolerance 0.1) | `baseline_reproduction` in `results/aya-source-verified-e5-analysis-20260928.json` |
| E6 inputs A and D vs. E2 (template 1) | 492 scores (246 pairs) | 0.0 nats (the E6 analysis stops above 0.1) | Checked by `analyze_e6_position_decomposition_20260928.py` |

**Built-in checks.** Before loading the model, the 27 and 28 September runners
check the SHA-256 of their frozen inputs against the protocol, and the 28 September
runners also check the tokenizer file and the prompt token IDs; any mismatch stops
the run. `ff.py run` stops if the project code differs from the code a protocol was
frozen with (Section 8).

**Runtime erratum.** The first launches of the source-checked E2 and E5 jobs stopped
before the model loaded, because of a bug in one call that loads the model settings.
Only that call was changed. The data, prompts, name forms, distractors, model
settings, outcomes, analysis plans and E5 site were unchanged, and the fix was made
before any output of these runs existed. The original files are kept in
`configs/failed_launch_20260928/` (complete archive), and
`results/AYA_RUNTIME_ERRATUM_20260928.md` gives the details.

---

## 8. Rerun Aya

**Requirements.** A Linux machine with a CUDA GPU of at least 11 GB, and a
Hugging Face account with access to `CohereLabs/aya-23-8B` (accept the model's
licence on its page). The reported runs used the hardware and software in Section 2;
other GPUs or library versions may give small numerical differences.

**Setup** (in a fresh clone of the repository):

```bash
python -m venv venv && source venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-common.txt
python -m pip install -e ".[model,quantization]"
huggingface-cli login
python project_plan/feasibility/download_model.py --revision 89da1a0ed02d6130f93ae0ffdbedb63b760c0471
```

`download_model.py` writes `model_snapshot.json`, with the SHA-256 of every weight
file. The 28 September runners (E6, E7 and the source-checked E2 and E5) also read the
snapshot from `data/raw/aya_23_8b`: copy or link the downloaded snapshot there and set
`snapshot_path` in `model_snapshot.json` to that folder. Both are ignored by Git.

Use a clone rather than the unpacked complete archive: the runners write to the same
result paths as the original runs, several refuse any other output path, and items
already present are resumed rather than recomputed. A few inputs of the 26 and 27
September runs exist only in the complete archive; the steps below say which.

**Reviewed cohorts** (pilot, held-out, Google-RE) run through `ff.py run`:

```bash
python ff.py run --snapshot model_snapshot.json --config CONFIG --facts FACTS \
    --templates data/review/imported_ReviewerA_20260926/templates.json \
    --protocol PROTOCOL --experiment {E1,E2,E3,E5} --split SPLIT \
    --shard {0,1} --num-shards 2 --output results/RUN_NAME
```

| Cohort | `CONFIG` | `FACTS` | `PROTOCOL` (copy from the complete archive) | `SPLIT` | Code |
|---|---|---|---|---|---|
| Pilot E1-E3 | `configs/study_reviewed_pilot_20260926.json` | `data/curated/pilot-screened-20260926.jsonl` | `results/tau_research_20260926/results/protocol-reviewed-pilot-20260926.json` | `pilot` | Before 27 September |
| Pilot E5 | `configs/mechanism_reviewed_pilot_20260926.json` | as above | `results/tau_mechanism_20260926/results/protocol-mechanism-pilot-20260926.json` | `pilot` | Before 27 September |
| Held-out E1, E2 | `configs/study_independent_test_v1.json` | `data/curated/aya-independent-test-v1-screened.jsonl` | `results/heldout/results/protocol-aya-independent-test-v1.json` | `test` | Before 27 September |
| Held-out E3 | `configs/study_independent_test_v1.json` | as above | `results/h27/results/protocol-aya-strengthening-behavior-20260927.json` | `test` | Current |
| Held-out E5 | `configs/mechanism_independent_test_pilot_selected_20260927.json` | as above | `results/h27/results/protocol-mechanism-heldout-pilot-selected-20260927.json` | `test` | Current |
| Google-RE E1, E2 | `configs/study_google_re_external_20260927.json` | `data/curated/aya-google-re-external-reviewed-20260927-screened.jsonl` (extract it first, see below) | `results/gre27/results/protocol-aya-google-re-external-20260927.json` | `test` | Current |

All reviewed-cohort runs used the reviewed templates file shown above, not the
default `configs/templates.json`. Each protocol file lists the configuration, facts
and templates it was frozen with, together with their SHA-256. `RUN_NAME` follows the
original folder names, for example `e2-reviewed-pilot-20260926-shard0`.

Two preparation steps, from the repository root:

```bash
# Google-RE facts file (kept inside its run archive)
tar -xzf results/aya-google-re-external-20260927.tar.gz \
    data/curated/aya-google-re-external-reviewed-20260927-screened.jsonl

# Rows marked "Before 27 September": restore the code these protocols were frozen with
tar -xzf results/aya-strengthening-20260927.tar.gz \
    results/source-before-strengthening-experiments.py results/source-before-strengthening-protocol.py
cp results/source-before-strengthening-experiments.py src/fragmented_facts/experiments.py
cp results/source-before-strengthening-protocol.py src/fragmented_facts/protocol.py
# ...run those rows, then restore the current code for the other rows:
git checkout -- src/fragmented_facts/experiments.py src/fragmented_facts/protocol.py
```

Without the second step, `ff.py run` stops with "Code changed after freeze" for those
rows. Each run writes to a subfolder named by the hash of its manifest (model,
experiment, split, shard, configuration, and the hashes of protocol, code, data and
templates), so a rerun with the same inputs and code writes to a subfolder with the
same name as the original run.

**Later experiments** use dedicated runners with the protocol built in:

```bash
python project_plan/feasibility/source_verified_aya_run_20260928.py \
    --cohort {arabic_geonames,hebrew_cbs} --shard {0,1} \
    --output results/aya-source-verified-20260928-COHORT-shardK
python project_plan/feasibility/source_verified_e5_run_20260928.py \
    --cohort {arabic_geonames,hebrew_cbs} --output results/aya-source-verified-e5-20260928-COHORT
python project_plan/feasibility/e6_position_decomposition_run_20260928.py --shard {0,1} \
    --output results/e6-position-decomposition-20260928-shardK
python project_plan/feasibility/e7_same_count_run_20260928.py --shard {0,1} \
    --output results/e7-same-count-20260928-shardK
python project_plan/feasibility/e4_reviewed_cohorts_run_20260927.py --cohort {pilot,test} --shard {0,1} \
    --output results/e4-reviewed-COHORT-20260927-shardK
python project_plan/feasibility/e3_byte_level_run_20260927.py --shard {0,1} \
    --output results/e3-byte-level-20260927-shardK
python project_plan/feasibility/e3_position_controls_run_20260927.py --shard {0,1} \
    --output results/e3-position-controls-20260927-shardK
```

The three 27 September runners (E3 byte-level, E3 position controls, E4) also check
five files that are only in the complete archive. Copy them into the clone at the same
paths before running:

```text
project_plan/feasibility/e3_byte_level_workflow_tau_20260927.py
project_plan/feasibility/e3_position_controls_workflow_tau_20260927.py
project_plan/feasibility/e4_reviewed_cohorts_workflow_tau_20260927.py
results/e3-byte-level-eligibility-feasibility-20260927.json
results/e3-outside-control-position-audit-20260927.json
```

The 28 September runners accept `--dry-run`, which checks the frozen inputs and
tokenizes the prompts without loading the model (it needs only the tokenizer files
of the snapshot). On a Slurm cluster, the matching templates in `slurm/` run the same
commands (set `NLP_PROJECT_ROOT` to the project folder and adjust the account,
partition and GPU request). After a rerun, run the analyses of Sections 5 and 6 on
the new outputs.

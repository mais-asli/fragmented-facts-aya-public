**Public, privacy-redacted research copy.** See [PUBLIC_REDACTION_NOTICE.md](PUBLIC_REDACTION_NOTICE.md) before interpreting the original protocol hashes.

# Fragmented Facts: Aya-23-8B multilingual recall experiments

Research code, frozen protocols, curated inputs, and privacy-redacted copies of the result artifacts for experiments on how Arabic and Hebrew name vocalization affects factual recall by Aya-23-8B. The repository includes the initial pilot, an independently selected held-out cohort, a Google-RE extension, later source-backed place-name cohorts, and controlled tokenization and activation-patching experiments. The matching anonymized v11 manuscript and analysis files are in `paper_v11/`.

**Reproducing the paper:** [REPRODUCIBILITY.md](REPRODUCIBILITY.md) records the original experiments and explains which checks remain available in this privacy-redacted copy. The complete archive (`Fragmented_Facts_Public_Reproducibility_v11.zip`), with the per-prompt records of every run and the paper source, is attached to the repository's release.

## Repository map

- `src/fragmented_facts/`: model loading, prompt construction, scoring, interventions, and shared utilities.
- `project_plan/feasibility/*_run_*.py`: experiment runners, including E3, E4, E6, E7, and the source-backed E2/E5 extensions.
- `scripts/`: cohort freezing, audits, analyses, and figure-generation programs.
- `configs/`: prespecified protocols and analysis plans. E6 and E7 are exploratory follow-ups; their own inputs and analysis plans were frozen before their respective new model outputs.
- `data/curated/`: study facts and name forms. `paper_v2/_work/` contains the earlier experiment grids needed by the frozen protocols.
- `results/*.tar.gz`: archived Aya outputs and run manifests; the E4 English development bundle contains model results only, with the working review sheet omitted. `results/*analysis*.json` and the selected Markdown reports contain analyses.
- `slurm/`: cluster job templates.

## Principal findings and scope

For the source-backed Arabic GeoNames city–country cohort, the prespecified marked-minus-unmarked gold/distractor margin was **−1.370 nats** across 184 subjects (hierarchical 95% CI **[−2.159, −0.587]**, Holm-adjusted *p* = **0.00099**). For the 240-subject Hebrew CBS locality–subdistrict cohort, it was **−0.308 nats** (95% CI **[−0.847, 0.161]**, adjusted *p* = **0.20581**). These cohorts use different relations and cannot support a direct language ranking. The separate source-disjoint E5 activation-patching test had a positive same-entity patch effect in Arabic; the Hebrew interval included zero. See the dated reports in `results/` for complete contrasts and limitations.

E6 and E7 examine token boundaries, positions, and matched-count forced splits on the earlier 246 fact–language pairs. These interventions are exploratory. Forced splits are outside the model's ordinary tokenizer distribution, so they do **not** identify a pure causal effect of token count. Exact-match answer accuracy is also sensitive to answer formatting and accepted aliases; likelihood margins and answer audits are reported alongside it. The original cohorts remain separate rather than being pooled into a single confirmatory estimate.

## Reproducing the analyses

Use Python 3.11 or newer. For CPU-only analysis of saved outputs:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e '.[test]'
pytest -q    # the model tests are skipped unless PyTorch is installed
python scripts/analyze_source_verified_aya_20260928.py
python scripts/analyze_source_verified_e5_20260928.py
python scripts/analyze_e6_position_decomposition_20260928.py
python scripts/analyze_e7_same_count_20260928.py
```

The analysis programs verify the archived outputs against frozen protocol identifiers and expected grids. Run them from the repository root. The checked-in raw archives are necessary for the analyses; do not repackage them if you need the recorded SHA-256 checks to remain valid.

## Running Aya again

The Aya-23-8B weights and tokenizer are not distributed here. Obtain your own access to [`CohereLabs/aya-23-8B`](https://huggingface.co/CohereLabs/aya-23-8B) and follow its license. Install the `model` and `quantization` extras, then download revision `89da1a0ed02d6130f93ae0ffdbedb63b760c0471` with `project_plan/feasibility/download_model.py --revision 89da1a0ed02d6130f93ae0ffdbedb63b760c0471`. This creates a local `model_snapshot.json`. For the 2026-09-28 source-backed and same-count runners, also make the downloaded snapshot available at `data/raw/aya_23_8b` (copy or symlink it) and set `model_snapshot.json`'s `snapshot_path` to that directory. Both the manifest and weights are ignored by Git. The saved protocols specify the precision and prompts used in the reported runs. Exact reruns require the pinned tokenizer, CUDA GPU, compatible packages, and the unmodified input/code hashes in the frozen protocols.

The Slurm templates accept `NLP_PROJECT_ROOT` as the cluster project root. They are examples for a compatible GPU cluster; adjust resource requests for your own installation. The model weights remain subject to the model's access terms.

## Data and result integrity

The curated source-backed cohorts include source identifiers, answer fields, and names used for the runs; the associated reports describe their checks and remaining linguistic uncertainty. Personal identifiers and private execution paths are redacted in this public derivative. The public checksums validate the redacted files; see `PUBLIC_REDACTION_NOTICE.md` for the effect on original protocol hashes. No model weights, credentials, personal review worksheets, or named submission PDFs are included.

## Version 11 update

The release `v11-public-redacted` contains the matching anonymized manuscript source, complete saved-output evidence and the layer-31/token-level computations used by Table 12. `paper_v9/` remains in Git as a historical source version; the v11 release archive contains the current `paper_v11/` source. No new Aya run or scientific estimate was introduced for v11.

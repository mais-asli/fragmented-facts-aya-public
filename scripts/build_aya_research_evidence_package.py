"""Package Aya-only research evidence, without checkpoint weights or paper."""

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/AYA_RESEARCH_EVIDENCE_20260927.zip'
MANIFEST = ROOT / 'output/AYA_RESEARCH_EVIDENCE_MANIFEST_20260927.json'
TEMP = ROOT / 'output/AYA_RESEARCH_EVIDENCE_20260927.zip.tmp'

EXPLICIT = [
    'model_snapshot.json', 'ff.py', 'requirements-cluster.txt', 'requirements-common.txt',
    'slurm/experiment.slurm',
    'data/raw/candidates_final.jsonl',
    'data/curated/pilot-reviewed-20260926.jsonl',
    'data/curated/aya-independent-test-v1.jsonl',
    'data/curated/aya-independent-test-v1.freeze.json',
    'results/heldout/data/curated/aya-independent-test-v1-screened.jsonl',
    'results/heldout/data/curated/aya-independent-test-v1-screened.screening.json',
    'results/heldout/results/protocol-aya-independent-test-v1.json',
    'data/review/imported_ReviewerA_20260926/templates.json',
    'configs/study_independent_test_v1.json',
    'configs/mechanism_independent_test_pilot_selected_20260927.json',
    'results/aya-independent-test-v1-research.tar.gz',
    'results/aya-strengthening-20260927.tar.gz',
    'results/heldout/results/analysis-aya-independent-test-v1.json',
    'results/h27/results/analysis-e3-aya-heldout-strengthening-20260927.json',
    'results/h27/results/analysis-e5-aya-heldout-strengthening-20260927.json',
    'results/heldout-e3-all-contrasts-20260927.json',
    'results/heldout-e5-primary-audit-20260927.json',
    'results/heldout-e5-baseline-consistency-20260927.json',
    'results/heldout-scoring-audit-20260927.json',
    'results/ReviewerA-output-audit-analysis-20260926.json',
    'outputs/01a09b51/aya_output_audit_20260926/ReviewerA_output_review_confirmed_20260926.csv',
    'outputs/01a09b51/aya_output_audit_20260926/ReviewerA_output_review_confirmed_20260926.provenance.json',
    'outputs/01a09b51/aya_output_audit_20260926/private_sampling_map.json',
    'outputs/01a09b51/aya_output_audit_20260926/semantic_completion_adjudication_20260926.json',
    'results/heldout-output-script-diagnostic-20260927.json',
    'results/answer-format-sensitivity-heldout-20260927.json',
    'results/heldout-strict-hq-sensitivity-20260927.json',
    'results/fragmentation-frequency-match-heldout-exploratory-20260927.json',
    'results/heldout-provenance-subgroups-exploratory-20260927.json',
    'results/answer-cue-exclusion-sensitivity-20260927.json',
    'results/context-mark-control-feasibility-20260927.json',
    'results/strengthening-worklog-count-verification-20260927.json',
    'results/heldout-e3-tokenizer-eligibility-20260927.json',
    'results/tau_strengthening_resume_20260927.json',
    'results/tau_strengthening_upload_20260927.json',
    'results/protocol-mechanism-heldout-pilot-selected-local-validation.json',
    'results/pilot_selected_window_evidence_20260927.json',
    'results/analysis-reviewed-pilot-with-mechanism-20260926.json',
    'output/PROPOSAL_AUDIT_AND_AYA_RESEARCH_EXTENSION_20260927.md',
    'docs/AYA_HELDOUT_SOURCE_QA_20260927.md',
    'output/AYA_STRENGTHENING_WORKLOG_20260927.md',
    'output/data_audit_20260927/FROZEN_FACT_LEDGER_97.csv',
    'output/data_audit_20260927/LEDGER_AUDIT.json',
    'output/data_audit_20260927/SOURCE_RECHECK.md',
    'output/data_audit_20260927/HELDOUT_GENERATION_CAP_RECHECK.md',
]
SCRIPT_NAMES = [
    'tau_upload.py', 'extract_tau_research_bundle.py',
    'TAU_RESUME_EXISTING_AYA_JOB_20260927.ps1',
    'tau_strengthening_resume_files_20260927.json',
    'build_versioned_fact_ledger.py',
    'freeze_heldout_e3_eligibility.py', 'audit_heldout_scoring.py',
    'audit_heldout_output_script.py',
    'analyze_heldout_strict_hq.py', 'analyze_answer_format_sensitivity.py',
    'match_fragmentation_heldout_exploratory.py',
    'analyze_heldout_provenance_subgroups.py',
    'analyze_answer_cue_sensitivity.py',
    'check_context_mark_control_feasibility.py',
    'verify_strengthening_worklog_counts.py',
    'analyze_heldout_e3_diagnostics.py', 'audit_heldout_e5_primary.py',
    'validate_heldout_e5_baselines.py',
    'build_aya_research_evidence_package.py',
]
WORKFLOW_NAMES = ['strengthening_workflow_tau.py', 'resume_strengthening_workflow_tau.py',
                  'resume_strengthening_workflow_tau_v2.py',
                  'protocol_pilot_gate_20260927.py', 'experiments_heldout_e3_20260927.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if OUT.exists() or MANIFEST.exists() or TEMP.exists():
        raise FileExistsError('Evidence package already built; preserve its first version')
    workflow = json.loads((ROOT / 'results/tau_strengthening_resume_20260927.json').read_text(encoding='utf-8'))
    if workflow.get('status') != 'workflow_finished' or workflow.get('workflow_exit_code') != 0:
        raise ValueError('TAU strengthening result not complete')
    paths = [ROOT / name for name in EXPLICIT]
    paths += [ROOT / 'scripts' / name for name in SCRIPT_NAMES]
    paths += [ROOT / 'project_plan/feasibility' / name for name in WORKFLOW_NAMES]
    paths += sorted((ROOT / 'src/fragmented_facts').glob('*.py'))
    # The pilot predates the later test archive and has no single local tarball.
    # Retain its complete raw item/manifests and both saved analysis reports.
    paths += sorted((ROOT / 'results/tau_research_20260926/results').rglob('*.json'))
    paths += sorted((ROOT / 'results/tau_mechanism_20260926/results').rglob('*.json'))
    if not all(path.is_file() and path.resolve().is_relative_to(ROOT.resolve()) for path in paths):
        missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
        raise FileNotFoundError(f'Missing evidence: {missing}')
    relative_paths = [path.relative_to(ROOT).as_posix() for path in paths]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError('Duplicate package paths')
    records = [{'path': relative, 'bytes': path.stat().st_size,
                'sha256': sha(path.read_bytes())}
               for path, relative in zip(paths, relative_paths)]
    manifest = {
        'scope': 'Aya-only reproducibility evidence, excluding model weights and paper',
        'model_id': 'CohereLabs/aya-23-8B',
        'revision': '89da1a0ed02d6130f93ae0ffdbedb63b760c0471',
        'files': records,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if not all(p.resolve().is_relative_to(ROOT.resolve()) for p in (OUT, MANIFEST, TEMP)):
        raise ValueError('Package paths escape project')
    with zipfile.ZipFile(TEMP, 'x') as archive:
        archive.writestr('MANIFEST.json', manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
        for path, relative in zip(paths, relative_paths):
            method = zipfile.ZIP_STORED if path.name.endswith(('.tar.gz', '.zip')) else zipfile.ZIP_DEFLATED
            archive.write(path, arcname=relative, compress_type=method)
    with zipfile.ZipFile(TEMP) as archive:
        if archive.testzip() is not None:
            raise ValueError('Evidence ZIP CRC verification failed')
        packaged = json.loads(archive.read('MANIFEST.json'))
        if packaged != manifest:
            raise ValueError('Internal manifest differs')
        for record in records:
            if sha(archive.read(record['path'])) != record['sha256']:
                raise ValueError('Evidence package SHA mismatch: ' + record['path'])
    TEMP.replace(OUT)
    MANIFEST.write_bytes(manifest_bytes)
    print(json.dumps({'package': str(OUT), 'files': len(records),
                      'zip_bytes': OUT.stat().st_size, 'zip_sha256': sha(OUT.read_bytes())}, indent=2))


if __name__ == '__main__':
    main()

"""Freeze human-reviewed English E4 development inputs before Aya outputs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs/e4_english_dev_protocol_20260927.json'
PATHS = [
    'data/curated/aya-e4-english-dev-ReviewerA-reviewed-20260927.jsonl',
    'data/curated/aya-e4-english-dev-ReviewerA-reviewed-20260927.manifest.json',
    'data/review/heldout_batch_v1/E4_ENGLISH_DEV_REVIEW_20260927.csv',
    'data/review/heldout_batch_v1/E4_ENGLISH_DEV_RECONCILED_20260927.csv',
    'data/review/heldout_batch_v1/E4_ENGLISH_DEV_RECONCILIATION_20260927.json',
    'data/review/heldout_batch_v1/manifest.json',
    'data/review/imported_ReviewerA_20260926/templates.json',
    'configs/e4_english_dev_20260927.json',
    'project_plan/feasibility/e4_english_dev_run_20260927.py',
    'project_plan/feasibility/e4_english_dev_workflow_tau_20260927.py',
    'slurm/e4-english-dev-20260927.slurm',
    'src/fragmented_facts/experiments.py',
    'src/fragmented_facts/model.py',
    'src/fragmented_facts/prompts.py',
    'src/fragmented_facts/scoring.py',
    'src/fragmented_facts/hooks.py',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('E4 protocol already frozen')
    facts = [json.loads(line) for line in (ROOT / PATHS[0]).read_text(encoding='utf-8').splitlines()]
    config = json.loads((ROOT / 'configs/e4_english_dev_20260927.json').read_text(encoding='utf-8'))
    snapshot = json.loads((ROOT / 'model_snapshot.json').read_text(encoding='utf-8'))
    if (len(facts) < 10 or len({f['subject_qid'] for f in facts}) != len(facts)
            or any(f['split'] != 'dev' or not f['english_review_only'] or
                   f['review']['reviewer'] != 'Reviewer A' or
                   set(f['subject_labels']) != {'en'} for f in facts)):
        raise ValueError('Expected personally reviewed English-only E4 development subjects')
    if config['languages'] != ['en'] or snapshot['model_id'] != 'CohereLabs/aya-23-8B':
        raise ValueError('Wrong E4 model or language')
    record = {'schema_version': 1,
              'scope': 'English-only source-reviewed preassigned development facts',
              'status': 'Frozen before English E4 Aya outputs; earlier E1/E2/E5 outcomes already existed',
              'model_id': snapshot['model_id'], 'model_revision': snapshot['revision'],
              'precision': config['precision'],
              'selected_fact_ids': sorted(f['fact_id'] for f in facts),
              'screen_rule': config['screen_rule'],
              'e4_sites': ['first_subject', 'last_subject', 'prediction'],
              'e4_layers': config['readout_layers'],
              'file_sha256': {relative: sha(ROOT / relative) for relative in PATHS}}
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'protocol': str(OUT), 'sha256': sha(OUT), 'facts': len(facts)}))


if __name__ == '__main__':
    main()

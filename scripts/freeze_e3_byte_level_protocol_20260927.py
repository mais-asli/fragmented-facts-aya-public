"""Freeze inputs for the post-E3 exploratory same-text Aya control."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs/e3_byte_level_protocol_20260927.json'
PATHS = [
    'data/curated/aya-independent-test-v1-screened.jsonl',
    'data/review/imported_ReviewerA_20260926/templates.json',
    'configs/study_independent_test_v1.json',
    'results/e3-byte-level-eligibility-feasibility-20260927.json',
    'project_plan/feasibility/e3_byte_level_run_20260927.py',
    'project_plan/feasibility/e3_byte_level_workflow_tau_20260927.py',
    'slurm/e3-byte-level-20260927.slurm',
    'src/fragmented_facts/experiments.py',
    'src/fragmented_facts/model.py',
    'src/fragmented_facts/prompts.py',
    'src/fragmented_facts/scoring.py',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('A protocol is already frozen; do not silently revise it')
    facts = [json.loads(line) for line in
             (ROOT / PATHS[0]).read_text(encoding='utf-8').splitlines() if line.strip()]
    eligibility = json.loads((ROOT / PATHS[3]).read_text(encoding='utf-8'))
    snapshot = json.loads((ROOT / 'model_snapshot.json').read_text(encoding='utf-8'))
    config = json.loads((ROOT / PATHS[2]).read_text(encoding='utf-8'))
    if len(facts) != 40 or {fact['split'] for fact in facts} != {'test'}:
        raise ValueError('Expected the 40 reviewed new-subject test facts')
    if not all(row['matched'] for row in eligibility['rows']
               if row['language'] in ('he', 'ar')):
        raise ValueError('Not all Hebrew/Arabic contrasts are matched')
    if (snapshot['model_id'] != 'CohereLabs/aya-23-8B' or
            snapshot['revision'] != '89da1a0ed02d6130f93ae0ffdbedb63b760c0471'
            or config['precision'] != 'nf4'):
        raise ValueError('Aya identity or quantization differs')
    protocol = {
        'schema_version': 1,
        'method_status': 'post-E3 exploratory extension, selected without reading its new Aya outcomes',
        'model_id': snapshot['model_id'],
        'model_revision': snapshot['revision'],
        'precision': config['precision'],
        'tokenizer_sha256': eligibility['input_sha256']['tokenizer_json'],
        'selected_fact_ids': sorted(fact['fact_id'] for fact in facts),
        'conditions': ['canonical', 'subject_split', 'outside_split'],
        'languages': ['he', 'ar'],
        'template': 't1',
        'shards': 2,
        'primary_contrast': 'subject_split minus outside_split, matched one extra token and identical visible text',
        'analysis': '40 subject-clustered, relation-macro pairs per language; 10000 bootstrap/sign-flip draws; Holm over two languages',
        'file_sha256': {relative: sha(ROOT / relative) for relative in PATHS},
    }
    OUT.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(OUT), 'sha256': sha(OUT),
                      'facts': len(facts), 'files': len(PATHS)}))


if __name__ == '__main__':
    main()

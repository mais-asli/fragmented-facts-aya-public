"""Freeze retrospective pilot/test English E4 inputs before E4 outputs."""

import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs/e4_reviewed_cohorts_protocol_20260927.json'
COHORTS = {
    'pilot': 'data/curated/pilot-screened-20260926.jsonl',
    'test': 'data/curated/aya-independent-test-v1-screened.jsonl',
}
PATHS = [
    *COHORTS.values(),
    'configs/e4_reviewed_cohorts_20260927.json',
    'data/review/imported_ReviewerA_20260926/templates.json',
    'project_plan/feasibility/e4_reviewed_cohorts_run_20260927.py',
    'project_plan/feasibility/e4_reviewed_cohorts_workflow_tau_20260927.py',
    'slurm/e4-reviewed-cohorts-20260927.slurm',
    *sorted(path.relative_to(ROOT).as_posix()
            for path in (ROOT / 'src/fragmented_facts').glob('*.py')),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Reviewed-cohort E4 protocol already frozen')
    config = json.loads((ROOT / 'configs/e4_reviewed_cohorts_20260927.json').read_text(encoding='utf-8'))
    snapshot = json.loads((ROOT / 'model_snapshot.json').read_text(encoding='utf-8'))
    if (snapshot['model_id'] != 'CohereLabs/aya-23-8B' or
            snapshot['revision'] != '89da1a0ed02d6130f93ae0ffdbedb63b760c0471' or
            config['languages'] != ['en'] or config['population'] != 'core' or
            not config['readout_layers_frozen']):
        raise ValueError('Unexpected Aya or E4 configuration')
    cohorts = {}
    subjects = set()
    for label, relative in COHORTS.items():
        rows = [json.loads(line) for line in (ROOT / relative).read_text(encoding='utf-8').splitlines()]
        core = [row for row in rows if row['core_member'] and row['english_eligible']]
        if (len(core) < 10 or any(row['split'] != label or row['review']['reviewer'] != 'Reviewer A'
                                   or not row['review']['pre_release_verified'] for row in rows)):
            raise ValueError('Invalid personally reviewed E4 screened cohort: ' + label)
        if subjects & {row['subject_qid'] for row in rows}:
            raise ValueError('Pilot/test subject leakage')
        subjects.update(row['subject_qid'] for row in rows)
        for relation in ('P19', 'P20', 'P159', 'P740'):
            if len({row['object_qid'] for row in core if row['relation'] == relation}) < 2:
                raise ValueError('No distinct-answer donor in ' + label + ' ' + relation)
        cohorts[label] = {'reviewed_count': len(rows), 'core_count': len(core),
                          'core_by_relation': dict(Counter(row['relation'] for row in core)),
                          'core_fact_ids': sorted(row['fact_id'] for row in core),
                          'screen_model_hashes': sorted({row['screen_model_hash'] for row in rows})}
    if cohorts['pilot']['reviewed_count'] != 57 or cohorts['test']['reviewed_count'] != 40:
        raise ValueError('Pilot/test reviewed fact counts changed')
    if sha(ROOT / COHORTS['pilot']) != sha(ROOT / 'results/tau_research_20260926/data/curated/pilot-screened-20260926.jsonl') or \
       sha(ROOT / COHORTS['test']) != sha(ROOT / 'results/heldout/data/curated/aya-independent-test-v1-screened.jsonl'):
        raise ValueError('Screened cohort copy is not byte-identical to verified prior archive')
    record = {'schema_version': 1,
              'scope': 'Retrospective English E4 on disjoint ReviewerA-reviewed pilot and held-out strict cores',
              'timing': 'Frozen before these E4 outputs, after the earlier E5 test and the separate E4 development-gate failure',
              'claim_limit': 'Exploratory localization and cross-cohort consistency; no prospective independent E5 layer validation',
              'model_id': snapshot['model_id'], 'model_revision': snapshot['revision'],
              'precision': config['precision'], 'layers': config['readout_layers'],
              'sites': ['first_subject', 'last_subject', 'prediction'],
              'cohorts': cohorts,
              'source_archive_sha256': {
                  'pilot_screened': sha(ROOT / 'results/tau_research_20260926/data/curated/pilot-screened-20260926.jsonl'),
                  'heldout_screened': sha(ROOT / 'results/heldout/data/curated/aya-independent-test-v1-screened.jsonl')},
              'file_sha256': {path: sha(ROOT / path) for path in PATHS}}
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'protocol': str(OUT), 'sha256': sha(OUT),
                      'cores': {k: v['core_count'] for k, v in cohorts.items()}}))


if __name__ == '__main__':
    main()

"""Import only ReviewerA-approved English development facts for Aya E4."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REVIEW = ROOT / 'data/review/heldout_batch_v1/E4_ENGLISH_DEV_RECONCILED_20260927.csv'
SOURCE = ROOT / 'data/review/heldout_batch_v1/candidate_source_rows.json'
MANIFEST = ROOT / 'data/review/heldout_batch_v1/manifest.json'
OUT = ROOT / 'data/curated/aya-e4-english-dev-ReviewerA-reviewed-20260927.jsonl'
REPORT = ROOT / 'data/curated/aya-e4-english-dev-ReviewerA-reviewed-20260927.manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or REPORT.exists():
        raise FileExistsError('E4 English development cohort is already frozen')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    expected = {r['fact_id'] for r in manifest['selected'] if r['split'] == 'dev'}
    with REVIEW.open(encoding='utf-8-sig', newline='') as source:
        decisions = list(csv.DictReader(source))
    if len(decisions) != 20 or {r['fact_id'] for r in decisions} != expected:
        raise ValueError('Expected all 20 preassigned dev candidates')
    source = {r['fact_id']: r for r in json.loads(SOURCE.read_text(encoding='utf-8'))}
    kept, exclusions = [], []
    for row in decisions:
        choice = row['ReviewerA_decision_keep_or_reject'].strip().lower()
        if choice not in ('keep', 'reject'):
            raise ValueError('ReviewerA must decide keep or reject on every dev row: ' + row['fact_id'])
        if row['ReviewerA_reviewer'].strip() != 'Reviewer A' or row['ReviewerA_verified_source_and_granularity'].strip().lower() not in ('yes', 'no'):
            raise ValueError('Missing personal verification and reviewer: ' + row['fact_id'])
        if row['analysis_decision'] not in ('keep', 'exclude'):
            raise ValueError('Missing analysis decision: ' + row['fact_id'])
        if choice == 'reject' and row['analysis_decision'] != 'exclude':
            raise ValueError('Cannot override ReviewerA rejection: ' + row['fact_id'])
        if row['analysis_decision'] == 'exclude':
            exclusions.append({'fact_id': row['fact_id'], 'ReviewerA_decision': choice,
                               'reason': row['analysis_note'] or row['ReviewerA_notes'] or row['precheck_note']})
            continue
        if (row['ReviewerA_verified_source_and_granularity'].strip().lower() != 'yes' or
                not row['source_url'].startswith('https://')):
            raise ValueError('Retained English fact lacks verified source: ' + row['fact_id'])
        original = source[row['fact_id']]
        fact = {key: original[key] for key in (
            'data_kind', 'fact_id', 'subject_qid', 'object_qid', 'relation',
            'split', 'source', 'popularity')}
        fact['subject_labels'] = {'en': original['subject_labels']['en']}
        fact['object_labels'] = {'en': original['object_labels']['en']}
        fact['object_aliases'] = {'en': original['object_aliases']['en']}
        if fact['split'] != 'dev' or fact['relation'] != row['relation'] or fact['subject_labels']['en'] != row['subject_en'] or fact['object_labels']['en'] != row['answer_en']:
            raise ValueError('Reviewed identity differs from original candidate: ' + row['fact_id'])
        fact['pairs'] = {}  # English-only E4; no claim of approved Hebrew/Arabic spellings.
        fact['review'] = {'status': 'approved', 'reviewer': 'Reviewer A',
                          'pre_release_verified': True,
                          'scope': 'English subject, relation, answer, source and granularity for E4 development only',
                          'evidence_url': row['source_url'], 'notes': row['ReviewerA_notes']}
        fact['english_review_only'] = True
        fact['english_review_source_url'] = row['source_url']
        kept.append(fact)
    by_relation = Counter(f['relation'] for f in kept)
    if len(kept) < 10 or any(by_relation[r] < 2 for r in ('P19', 'P20', 'P159', 'P740')):
        raise ValueError('E4 gate needs at least 10 approved subjects and two facts per relation')
    for relation in by_relation:
        if len({f['object_qid'] for f in kept if f['relation'] == relation}) < 2:
            raise ValueError('No distinct same-relation donor answer for ' + relation)
    if len({f['fact_id'] for f in kept}) != len(kept) or len({f['subject_qid'] for f in kept}) != len(kept):
        raise ValueError('E4 development cohort contains duplicate subjects or facts')
    if any(f['data_kind'] != 'research' or not f['subject_labels']['en'] or
           not f['object_labels']['en'] or f['review']['status'] != 'approved'
           for f in kept):
        raise ValueError('English-only development row is incomplete')
    with OUT.open('w', encoding='utf-8') as target:
        for fact in sorted(kept, key=lambda f: f['fact_id']):
            target.write(json.dumps(fact, ensure_ascii=False, sort_keys=True) + '\n')
    report = {'scope': 'Personally reviewed English development facts only; no target-language spelling approval claimed',
              'source_candidate_manifest_sha256': sha(MANIFEST), 'decision_csv_sha256': sha(REVIEW),
              'source_rows_sha256': sha(SOURCE), 'output_sha256': sha(OUT),
              'retained': len(kept), 'by_relation': dict(by_relation), 'excluded': exclusions}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'retained': len(kept), 'by_relation': by_relation, 'output': str(OUT)}, default=dict))


if __name__ == '__main__':
    main()

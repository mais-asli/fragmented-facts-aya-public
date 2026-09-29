"""Freeze ReviewerA-reviewed external Google-RE facts before any Aya inference."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.data import validate_facts  # noqa: E402
from fragmented_facts.unicode import nfc, validate_pair  # noqa: E402

BASE = ROOT / 'data/review/google_re_external_20260927'
SOURCE = BASE / 'CANDIDATES_32_GOOGLE_RE_20260927.csv'
REVIEW = BASE / 'CANDIDATES_32_WITH_MARK_SUGGESTIONS_20260927.csv'
SELECTION = BASE / 'CANDIDATES_32_GOOGLE_RE_20260927.manifest.json'
OUT = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.jsonl'
FREEZE = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.freeze.json'
FIXED = ['relation', 'subject_qid', 'answer_qid', 'subject_en_google_re',
         'subject_en_wikidata', 'answer_en_google_re', 'answer_en_reviewed',
         'answer_he_reviewed', 'answer_ar_reviewed', 'evidence_url',
         'evidence_snippet', 'judgments_yes', 'judgments_total',
         'source_path', 'source_line', 'source_uuid']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def main():
    if OUT.exists() or FREEZE.exists():
        raise FileExistsError('External review cohort already frozen')
    source, reviews = read(SOURCE), read(REVIEW)
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    if sha(SOURCE) != selection['candidate_csv_sha256'] or len(source) != 32 or len(reviews) != 32:
        raise ValueError('External source selection changed')
    approved, excluded = [], []
    for index, (frozen, row) in enumerate(zip(source, reviews), 1):
        if any(frozen[key] != row[key] for key in FIXED):
            raise ValueError(f'Frozen identity, source or answer changed at row {index}')
        decision = row['ReviewerA_fact_and_evidence_keep_or_reject'].strip().lower()
        if decision not in ('keep', 'reject') or row['ReviewerA_reviewer'].strip() != 'Reviewer A':
            raise ValueError(f'ReviewerA decision/reviewer missing at row {index}')
        if not row['ReviewerA_notes'].strip():
            raise ValueError(f'ReviewerA source explanation or exclusion reason missing at row {index}')
        fact_id = f"{row['subject_qid']}-{row['relation']}-{row['answer_qid']}"
        if decision == 'reject':
            excluded.append({'fact_id': fact_id, 'reason': row['ReviewerA_notes']})
            continue
        if row['ReviewerA_he_U_D_approved'].strip().lower() != 'yes' or row['ReviewerA_ar_U_D_approved'].strip().lower() != 'yes':
            raise ValueError(f'Both language pairs must be personally approved: {fact_id}')
        for language in ('he', 'ar'):
            validate_pair(row[f'subject_{language}_U_candidate'].strip(),
                          row[f'subject_{language}_D_proposed'].strip(), language)
        subject_en = nfc((row['subject_en_wikidata'] or row['subject_en_google_re']).strip())
        if not subject_en:
            raise ValueError(f'Missing English subject label: {fact_id}')
        subject_labels = {'en': subject_en,
                          'he': nfc(row['subject_he_U_candidate'].strip()),
                          'ar': nfc(row['subject_ar_U_candidate'].strip())}
        object_labels = {'en': nfc(row['answer_en_reviewed'].strip()),
                         'he': nfc(row['answer_he_reviewed'].strip()),
                         'ar': nfc(row['answer_ar_reviewed'].strip())}
        fact = {'data_kind': 'research', 'fact_id': fact_id,
                'subject_qid': row['subject_qid'], 'object_qid': row['answer_qid'],
                'relation': row['relation'], 'split': 'test',
                'source': {'dataset': 'Meta LAMA Google-RE 2019',
                           'archive_sha256': selection['google_archive_sha256'],
                           'path': row['source_path'], 'line': int(row['source_line']),
                           'uuid': row['source_uuid'], 'evidence_url': row['evidence_url'],
                           'evidence_snippet': row['evidence_snippet']},
                'subject_labels': subject_labels, 'object_labels': object_labels,
                'object_aliases': {language: [label] for language, label in object_labels.items()},
                'pairs': {language: {'U': subject_labels[language],
                                     'D': nfc(row[f'subject_{language}_D_proposed'].strip()),
                                     'status': 'approved', 'reviewer': 'Reviewer A',
                                     'notes': row['ReviewerA_notes']}
                          for language in ('he', 'ar')},
                'review': {'status': 'approved', 'reviewer': 'Reviewer A',
                           'pre_release_verified': True,
                           'scope': 'Fact, source and Hebrew/Arabic U/D name pairs',
                           'notes': row['ReviewerA_notes'],
                           'evidence_url': row['evidence_url']}}
        approved.append(fact)
    if len({f['subject_qid'] for f in approved}) != len(approved):
        raise ValueError('Duplicate external subjects')
    relation_counts = Counter(f['relation'] for f in approved)
    if any(relation_counts[r] < 2 or len({f['object_qid'] for f in approved if f['relation'] == r}) < 2
           for r in ('P19', 'P20')):
        raise ValueError('External study needs two approved subjects and answer cities per relation')
    validate_facts(approved, require_review=True)
    with OUT.open('w', encoding='utf-8') as file:
        for fact in approved:
            file.write(json.dumps(fact, ensure_ascii=False, sort_keys=True) + '\n')
    FREEZE.write_text(json.dumps({
        'schema_version': 1, 'cohort': 'google_re_external_pre_aya_20260927',
        'reviewer': 'Reviewer A', 'candidate_count': 32,
        'approved_count': len(approved), 'excluded': excluded,
        'by_relation': dict(relation_counts),
        'source_description': 'Different archived dataset from mLAMA; Google-RE underlying evidence also from Wikipedia',
        'selection_manifest_sha256': sha(SELECTION),
        'source_candidates_sha256': sha(SOURCE), 'ReviewerA_review_csv_sha256': sha(REVIEW),
        'curated_sha256': sha(OUT),
        'status': 'Frozen only after ReviewerA completed the review and before Aya external-cohort inference',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'approved': len(approved), 'rejected': len(excluded),
                      'by_relation': dict(relation_counts), 'output': str(OUT)}))


if __name__ == '__main__':
    main()

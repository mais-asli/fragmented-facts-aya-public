"""Create a tidy, read-only provenance ledger from both frozen research cohorts.

The ledger adds audit flags; it never edits approved fact text, answers or splits.
"""

import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'pilot': ROOT / 'data/curated/pilot-reviewed-20260926.jsonl',
    'test': ROOT / 'data/curated/aya-independent-test-v1.jsonl',
}
OUT = ROOT / 'output/data_audit_20260927'
HQ_TIERS = {
    'Q863432': ('A', 'Official source explicitly identifies Dhaka head office.'),
    'Q1320232': ('A', 'Official source places main administration in Detroit; city appears in subject.'),
    'Q150068': ('B', 'Official party legal domicile is Barcelona; verify period of continuity.'),
    'Q29052': ('B', 'Saved source states Nashville main campus; post-output Chancellor-office support exists.'),
    'Q37548': ('B', 'Saved accreditor source states Boston main campus; post-output President-office support exists.'),
    'Q271110': ('C', 'Treasury lists Tehran location but not explicit headquarters.'),
}


def main():
    cohorts = {name: [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
               for name, path in FILES.items()}
    pilot_answers = {row['object_qid'] for row in cohorts['pilot']}
    ledger = []
    for split, facts in cohorts.items():
        for fact in facts:
            subject_qid = fact['subject_qid']
            tier, tier_note = HQ_TIERS.get(subject_qid, ('', '')) if fact['relation'] == 'P159' and split == 'test' else ('', '')
            review = fact['review']
            row = {
                'split': split, 'fact_id': fact['fact_id'], 'relation': fact['relation'],
                'subject_qid': subject_qid, 'answer_qid': fact['object_qid'],
                'subject_en': fact['subject_labels']['en'],
                'subject_he_U': fact['pairs']['he']['U'], 'subject_he_D': fact['pairs']['he']['D'],
                'subject_ar_U': fact['pairs']['ar']['U'], 'subject_ar_D': fact['pairs']['ar']['D'],
                'answer_en': fact['object_labels']['en'],
                'answer_he': fact['object_labels']['he'],
                'answer_ar': fact['object_labels']['ar'],
                'source_dataset': fact['source']['dataset'],
                'source_archive_sha256': fact['source']['archive_sha256'],
                'evidence_url': review['evidence_url'],
                'evidence_domain': urlparse(review['evidence_url']).hostname or '',
                'reviewer_recorded': review['reviewer'],
                'review_notes': review.get('notes', ''),
                'wikidata_single_answer_agrees': fact['wikidata']['single_answer_agrees'],
                'sitelinks': fact['popularity']['sitelinks'],
                'pageviews_available': any(value is not None for value in fact['popularity']['pageviews'].values()),
                'pilot_answer_qid_reused': split == 'test' and fact['object_qid'] in pilot_answers,
                'answer_text_in_subject_en': fact['object_labels']['en'].casefold() in fact['subject_labels']['en'].casefold(),
                'answer_text_in_subject_he': fact['object_labels']['he'] in fact['subject_labels']['he'],
                'answer_text_in_subject_ar': fact['object_labels']['ar'] in fact['subject_labels']['ar'],
                'provisional_HQ_evidence_tier': tier,
                'provisional_HQ_evidence_note': tier_note,
                'needs_temporal_source_check': 'Tofiq Bahramov' in fact['subject_labels']['en'],
                'needs_corporate_continuity_check': 'San Miguel Corporation' in fact['subject_labels']['en'],
            }
            ledger.append(row)
    if len(ledger) != 97 or len({row['fact_id'] for row in ledger}) != 97:
        raise ValueError('Unexpected cohort size or duplicate fact')
    OUT.mkdir(exist_ok=True)
    with (OUT / 'FROZEN_FACT_LEDGER_97.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    report = {
        'scope': 'Tidy provenance and recheck flags only; no changes to frozen facts or Aya results',
        'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in FILES.items()},
        'facts': len(ledger),
        'all_single_answer_wikidata_agree': all(row['wikidata_single_answer_agrees'] for row in ledger),
        'test_answer_qids_seen_in_pilot': sum(row['pilot_answer_qid_reused'] for row in ledger),
        'answer_text_in_subject_by_language': {
            lang: sum(row[f'answer_text_in_subject_{lang}'] for row in ledger)
            for lang in ('en', 'he', 'ar')},
        'answer_text_in_subject_any_language': sum(
            any(row[f'answer_text_in_subject_{lang}'] for lang in ('en', 'he', 'ar'))
            for row in ledger),
        'hq_test_evidence_tiers': {tier: sum(row['provisional_HQ_evidence_tier'] == tier for row in ledger)
                                    for tier in ('A', 'B', 'C')},
        'source_note': 'A URL and a reviewer flag do not prove that every cited page predates Aya or explicitly supports the relation.',
    }
    (OUT / 'LEDGER_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

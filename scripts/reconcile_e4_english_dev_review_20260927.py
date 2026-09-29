"""Preserve ReviewerA's raw E4 decisions and create an auditable analysis copy."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/review/heldout_batch_v1'
RAW = BASE / 'E4_ENGLISH_DEV_REVIEW_20260927.csv'
OUT = BASE / 'E4_ENGLISH_DEV_RECONCILED_20260927.csv'
MANIFEST = BASE / 'E4_ENGLISH_DEV_RECONCILIATION_20260927.json'

SOURCE_FIX = {
    'Q77253-P19-Q1726': 'https://www.deutsche-biographie.de/sfz69312.html?language=en',
    'Q190368-P19-Q87': 'https://www.livius.org/sources/content/appian/',
    'Q549786-P20-Q84': 'https://www.britannica.com/biography/John-Soane',
    'Q239663-P159-Q11194': 'https://www.nfsbih.ba/stats/team/6353021/409-fk-sarajevo/',
    'Q912887-P159-Q350': 'https://www.cambridge.org/um/legal',
    'Q152433-P740-Q49218': 'https://www.encyclopedia.com/social-sciences-and-law/economics-business-and-labor/businesses-and-occupations/xerox-corp',
    'Q910379-P740-Q220': 'https://www.treccani.it/enciclopedia/finmeccanica/',
}

ANALYSIS_EXCLUSIONS = {
    'Q1130172-P159-Q486479': 'Official page supports Rochester as main campus, not an explicit headquarters.',
    'Q3695910-P159-Q171866': 'University main address does not establish a headquarters in the same sense as company or federation records.',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError('Reconciled E4 review already exists')
    with RAW.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames
    if len(rows) != 20 or set(SOURCE_FIX) - {r['fact_id'] for r in rows}:
        raise ValueError('Expected the 20 reviewed E4 development candidates')
    for row in rows:
        choice = row['ReviewerA_decision_keep_or_reject'].strip().lower()
        if choice not in ('keep', 'reject') or not row['ReviewerA_notes'].strip():
            raise ValueError('Missing decision or notes for ' + row['fact_id'])
        row['ReviewerA_reviewer'] = 'Reviewer A'
        if row['fact_id'] in SOURCE_FIX:
            if choice != 'keep' or row['ReviewerA_verified_source_and_granularity'] != 'no':
                raise ValueError('Unexpected source correction state: ' + row['fact_id'])
            row['source_url'] = SOURCE_FIX[row['fact_id']]
            row['ReviewerA_verified_source_and_granularity'] = 'yes'
        row['analysis_decision'] = ('exclude' if choice == 'reject' or row['fact_id'] in ANALYSIS_EXCLUSIONS
                                    else 'keep')
        row['analysis_note'] = ANALYSIS_EXCLUSIONS.get(row['fact_id'], '')
    with OUT.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns + ['analysis_decision', 'analysis_note'])
        writer.writeheader()
        writer.writerows(rows)
    MANIFEST.write_text(json.dumps({
        'raw_review_sha256': sha(RAW),
        'reconciled_sha256': sha(OUT),
        'reviewer_confirmation': 'ReviewerA explicitly confirmed in chat that she personally checked all 20 decisions and replacement source URLs. The original CSV still had an AI placeholder; this copy reconciles it without changing the original.',
        'source_url_fixes': SOURCE_FIX,
        'analysis_exclusions_overriding_ReviewerA_keep': ANALYSIS_EXCLUSIONS,
        'user_keep': sum(r['ReviewerA_decision_keep_or_reject'] == 'keep' for r in rows),
        'analysis_keep': sum(r['analysis_decision'] == 'keep' for r in rows),
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Reconciled 20 E4 decisions; {sum(r["analysis_decision"] == "keep" for r in rows)} analysis keeps')


if __name__ == '__main__':
    main()

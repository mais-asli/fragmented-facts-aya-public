"""Prepare a blinded, stratified output-audit sheet from real Aya E1/E2 runs."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tau_research_20260926'
ANALYSIS = BASE / 'results/analysis-reviewed-pilot-20260926.json'
OUT = ROOT / 'outputs/01a09b51/aya_output_audit_20260926'
LIMIT = {'en': 20, 'he': 30, 'ar': 30}


def rank(row):
    key = json.dumps(row['key'], sort_keys=True)
    return hashlib.sha256(('17/output-audit/' + key).encode()).hexdigest()


def clean_question(text):
    text = re.sub(r'<\|[^>]+\|>|<BOS_TOKEN>', '', text)
    return text.strip()


def main():
    report = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    if report['data_kind'] != 'research':
        raise ValueError('No actual Aya research results')
    rows = []
    for relative in report['run_paths']:
        for path in sorted((BASE / relative / 'items').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            exp, lang = row['key']['experiment'], row['key']['language']
            if (lang == 'en' and exp == 'E1') or (lang in ('he', 'ar') and exp == 'E2'):
                rows.append(row)
    if not rows:
        raise ValueError('No generated E1/E2 outputs')
    by_language = defaultdict(lambda: defaultdict(list))
    disagreements = []
    for row in rows:
        language = row['key']['language']
        stratum = row['key']['variant'] + '/' + row['evaluation']['category']
        by_language[language][stratum].append(row)
        broad = row.get('evaluation_all_aliases')
        if broad and broad['entity_correct'] != row['evaluation']['entity_correct']:
            disagreements.append(row)
    selected = []
    for language in ('en', 'he', 'ar'):
        buckets = {key: sorted(group, key=rank) for key, group in by_language[language].items()}
        while len([r for r in selected if r['key']['language'] == language]) < LIMIT[language] and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len([r for r in selected if r['key']['language'] == language]) < LIMIT[language]:
                    selected.append(buckets[key].pop(0))
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = OUT / 'ReviewerA_output_review.csv'
    mapping = OUT / 'private_sampling_map.json'
    if sheet.exists() or mapping.exists():
        raise FileExistsError('Output audit already prepared; preserve annotations')
    private = []
    with sheet.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'Audit ID', 'Language', 'Relation', 'Question', 'Gold answer',
            'Aya output', 'ReviewerA: entity correct (yes/no)',
            'ReviewerA: requested language correct (yes/no)', 'ReviewerA: error type',
            'ReviewerA: notes', 'Reviewer'])
        writer.writeheader()
        for row in selected:
            audit_id = rank(row)[:16]
            writer.writerow({
                'Audit ID': audit_id, 'Language': row['key']['language'],
                'Relation': row['fact']['relation'],
                'Question': clean_question(row['prompt']['text']),
                'Gold answer': row['gold_score']['answer'],
                'Aya output': row['generation']['text']})
            private.append({'audit_id': audit_id, 'key': row['key'],
                            'machine_primary': row['evaluation'],
                            'machine_all_aliases': row.get('evaluation_all_aliases'),
                            'sampling_stratum': row['key']['variant'] + '/' + row['evaluation']['category'],
                            'stratum_size': len(by_language[row['key']['language']][
                                row['key']['variant'] + '/' + row['evaluation']['category']])})
    mapping.write_text(json.dumps(private, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    disagreement_path = OUT / 'canonical_vs_alias_disagreements.json'
    disagreement_path.write_text(json.dumps([{
        'key': row['key'], 'gold': row['gold_score']['answer'],
        'output': row['generation']['text'], 'primary': row['evaluation'],
        'all_aliases': row['evaluation_all_aliases']} for row in disagreements],
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rows_for_ReviewerA': len(selected),
                      'by_language': {lang: sum(r['key']['language'] == lang for r in selected)
                                      for lang in LIMIT},
                      'canonical_vs_alias_disagreements': len(disagreements),
                      'audit_csv': str(sheet)}, indent=2))


if __name__ == '__main__':
    main()

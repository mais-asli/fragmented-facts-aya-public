"""Deterministic, provenance-preserving audit of held-out E2 output scoring."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/heldout'
ANALYSIS = BASE / 'results/analysis-aya-independent-test-v1.json'
FORMAT = ROOT / 'results/answer-format-sensitivity-heldout-20260927.json'
OUT = ROOT / 'results/heldout-scoring-audit-20260927.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    if analysis['data_kind'] != 'research' or analysis['model']['model_id'] != 'CohereLabs/aya-23-8B':
        raise ValueError('Research Aya result required')
    totals = defaultdict(Counter)
    relations = defaultdict(Counter)
    categories = defaultdict(Counter)
    disagreements = []
    caps = []
    keys = set()
    for run in analysis['run_paths']:
        if '/e2-' not in '/' + run:
            continue
        for path in (BASE / run / 'items').glob('*.json'):
            row = json.loads(path.read_text(encoding='utf-8'))
            key = row['key']
            if key['experiment'] != 'E2' or key['language'] not in ('he', 'ar'):
                raise ValueError('Unexpected E2 row identity')
            unique = (key['fact_id'], key['language'], key['template'], key['variant'])
            if unique in keys:
                raise ValueError('Duplicate result')
            keys.add(unique)
            group = (key['language'], key['variant'])
            totals[group]['rows'] += 1
            relations[(key['language'], row['fact']['relation'], key['variant'])]['rows'] += 1
            primary = row['evaluation']
            alias = row['evaluation_all_aliases']
            totals[group]['primary_correct'] += bool(primary['entity_correct'])
            totals[group]['all_aliases_correct'] += bool(alias['entity_correct'])
            relations[(key['language'], row['fact']['relation'], key['variant'])]['primary_correct'] += bool(primary['entity_correct'])
            relations[(key['language'], row['fact']['relation'], key['variant'])]['all_aliases_correct'] += bool(alias['entity_correct'])
            categories[group][primary['category']] += 1
            if primary['entity_correct'] != alias['entity_correct']:
                disagreements.append({'key': key, 'gold': row['gold_score']['answer'],
                                      'output': row['generation']['text'],
                                      'primary_correct': primary['entity_correct'],
                                      'all_aliases_correct': alias['entity_correct']})
            if primary['category'] == 'malformed_truncated':
                caps.append({'key': key, 'output': row['generation']['text'],
                             'machine_category': primary['category'],
                             'note': 'Mechanical generation-cap category; semantic completeness requires separate review'})
    if len(keys) != 480:
        raise ValueError(f'Expected 480 E2 rows, got {len(keys)}')
    sensitivity = json.loads(FORMAT.read_text(encoding='utf-8'))
    if sensitivity['records'] != 840:
        raise ValueError('Saved E1/E2 format sensitivity coverage changed')
    report = {
        'scope': 'Mechanical score audit, no new Aya inference and no new human output adjudication',
        'input_sha256': {'analysis': sha(ANALYSIS), 'format_sensitivity': sha(FORMAT)},
        'e2_rows': len(keys),
        'language_variant': [
            {'language': lang, 'variant': var, **dict(totals[lang, var]),
             'primary_category_counts': dict(categories[lang, var])}
            for lang in ('he', 'ar') for var in ('U', 'D')],
        'relation_language_variant': [
            {'language': lang, 'relation': rel, 'variant': var,
             **dict(relations[lang, rel, var])}
            for lang in ('he', 'ar') for rel in ('P19', 'P20', 'P159', 'P740') for var in ('U', 'D')],
        'primary_vs_all_aliases_disagreements': disagreements,
        'mechanical_cap_cases': caps,
        'format_sensitivity_counts_all_e1_e2': sensitivity['counts'],
        'interpretation_limits': [
            'requested_language_correct is a joint entity-and-language scorer flag; it is not a standalone language judgment on wrong-entity outputs',
            'all-aliases correctness may count a same-place name in an unexpected output language',
            'prefix-stripping sensitivity leaves strict complete-answer scoring unchanged',
            'mechanical generation-cap status does not prove the sentence is semantically incomplete',
            'no assistant audit is recorded as an independent human review',
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'e2_rows': len(keys), 'alias_disagreements': len(disagreements),
                      'cap_cases': len(caps), 'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

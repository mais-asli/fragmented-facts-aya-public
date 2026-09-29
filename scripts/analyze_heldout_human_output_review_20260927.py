"""Validate ReviewerA's independent held-out output labels and weighted agreement."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/review/heldout_output_audit_20260927'
REVIEW = BASE / 'ReviewerA_HELDOUT_80_OUTPUT_REVIEW_20260927.csv'
ORIGINAL = BASE / 'BLIND_ORIGINAL_80_20260927.csv'
MANIFEST = BASE / 'MACHINE_AND_SAMPLE_MANIFEST_20260927.json'
OUT = ROOT / 'results/heldout-human-output-audit-20260927.json'
LABELS = ['ReviewerA: entity correct (yes/no)',
          'ReviewerA: requested language correct (yes/no)',
          'ReviewerA: error type', 'ReviewerA: notes', 'Reviewer']
CATEGORIES = {'correct', 'wrong_entity', 'wrong_granularity', 'wrong_language',
              'ambiguous_or_multiple', 'abstention', 'truncated', 'other'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def weighted(rows, field):
    den = sum(r['weight'] for r in rows)
    return sum(r['weight'] * int(r[field]) for r in rows) / den if den else None


def cluster_interval(rows, field, seed):
    by_relation = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_relation[row['relation']][row['subject_qid']].append(row)
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        picked = []
        for subjects in by_relation.values():
            keys = sorted(subjects)
            for _ in keys:
                picked.extend(subjects[rng.choice(keys)])
        draws.append(weighted(picked, field))
    draws.sort()
    return [draws[249], draws[9749]]


def main():
    if OUT.exists():
        raise FileExistsError('Held-out human audit already finalized')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['visible_csv_sha256_before_review'] != sha(ORIGINAL):
        raise ValueError('Original blinded audit sheet changed')
    fields, originals = read(ORIGINAL)
    reviewed_fields, reviews = read(REVIEW)
    if fields != reviewed_fields or len(originals) != 80 or len(reviews) != 80:
        raise ValueError('Held-out review sheet structure changed')
    machine = {r['audit_id']: r for r in manifest['machine_rows']}
    if len(machine) != 80:
        raise ValueError('Frozen machine manifest incomplete')
    entries = []
    for original, review in zip(originals, reviews):
        audit_id = original['Audit ID']
        if (audit_id not in machine or review['Audit ID'] != audit_id or
                any(original[k] != review[k] for k in fields if k not in LABELS)):
            raise ValueError('Fixed sample, question or output changed: ' + audit_id)
        entity = review[LABELS[0]].strip().lower()
        language = review[LABELS[1]].strip().lower()
        category = review[LABELS[2]].strip().lower()
        if (entity not in ('yes', 'no') or language not in ('yes', 'no') or
                category not in CATEGORIES or review['Reviewer'].strip() != 'Reviewer A'):
            raise ValueError('ReviewerA judgment/reviewer incomplete: ' + audit_id)
        if category == 'correct' and (entity != 'yes' or language != 'yes'):
            raise ValueError('Correct category contradicts ReviewerA yes/no labels: ' + audit_id)
        m = machine[audit_id]
        weight = m['stratum_size'] / m['sampled_in_stratum']
        entries.append({
            'audit_id': audit_id, 'fact_id': m['fact_id'],
            'subject_qid': m['subject_qid'], 'language': m['key']['language'],
            'relation': m['relation'], 'variant': m['key']['variant'],
            'template': m['key']['template'], 'stratum': m['stratum'],
            'weight': weight, 'human_entity_correct': entity == 'yes',
            'human_requested_language_correct': language == 'yes',
            'human_error_type': category, 'human_notes': review['ReviewerA: notes'],
            'machine_entity_correct': bool(m['machine_entity_correct']),
            'machine_category': m['machine_category'],
            'machine_joint_language_flag': bool(m['machine_joint_language_flag']),
            'machine_all_aliases_correct': bool(m['machine_all_aliases_correct']),
            'machine_truncated_flag': bool(m['generation_truncated_flag']),
        })
    if len({r['audit_id'] for r in entries}) != 80:
        raise ValueError('Duplicate human audit ID')
    groups = {}
    for language in ('he', 'ar'):
        for variant in ('U', 'D'):
            rows = [r for r in entries if r['language'] == language and r['variant'] == variant]
            if len(rows) != 20 or len({r['subject_qid'] for r in rows}) < 8:
                raise ValueError('Held-out audit group incomplete')
            eligible_language = [r for r in rows if r['human_entity_correct'] and
                                 r['machine_entity_correct']]
            for row in rows:
                row['entity_agree'] = row['human_entity_correct'] == row['machine_entity_correct']
                row['language_agree_conditional'] = (
                    row['human_requested_language_correct'] == row['machine_joint_language_flag'])
            groups[f'{language}/{variant}'] = {
                'sample_n': len(rows), 'distinct_facts': len({r['subject_qid'] for r in rows}),
                'population_n_from_weights': sum(r['weight'] for r in rows),
                'human_entity_correct_weighted': weighted(rows, 'human_entity_correct'),
                'human_entity_correct_fact_cluster_bootstrap_95_interval':
                    cluster_interval(rows, 'human_entity_correct', 17 + len(groups)),
                'human_language_correct_weighted_independent_of_entity':
                    weighted(rows, 'human_requested_language_correct'),
                'machine_human_entity_agreement_weighted': weighted(rows, 'entity_agree'),
                'conditional_language_comparison_sample_n': len(eligible_language),
                'conditional_machine_human_language_agreement_sample':
                    sum(r['language_agree_conditional'] for r in eligible_language) /
                    len(eligible_language) if eligible_language else None,
                'human_error_type_sample_counts': dict(Counter(r['human_error_type'] for r in rows)),
            }
    output = {
        'schema_version': 1, 'scope': 'Personally reviewed held-out generated-output audit',
        'reviewer': 'Reviewer A', 'sample_n': len(entries),
        'sampling_manifest_sha256': sha(MANIFEST), 'review_csv_sha256': sha(REVIEW),
        'groups': groups, 'rows': entries,
        'interpretation_limits': [
            'The 80 outputs are stratified, not a proportional random sample. Group population estimates use inverse sampling fractions within frozen strata.',
            'Facts appear across multiple prompts and conditions; bootstrap resamples facts within relation, preserving all selected rows for a fact.',
            'Bootstrap intervals condition on the fixed stratified sample and weights; they do not include all annotation uncertainty.',
            'Machine requested_language_correct is joint with entity correctness. Machine/human language agreement is reported only when both label the entity correct.',
            'Strict machine exact-answer scoring remains the primary outcome; human judgments diagnose output-format and alias errors.',
        ],
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'sample': len(entries), 'groups': {
        name: {'human_entity_weighted': value['human_entity_correct_weighted'],
               'human_language_weighted': value['human_language_correct_weighted_independent_of_entity']}
        for name, value in groups.items()}}, indent=2))


if __name__ == '__main__':
    main()

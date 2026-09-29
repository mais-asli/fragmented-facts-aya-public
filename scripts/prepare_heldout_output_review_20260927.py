"""Freeze a blinded 80-output held-out human audit with known sample weights."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/heldout/results'
FACTS = ROOT / 'data/curated/aya-independent-test-v1-screened.jsonl'
OUT_DIR = ROOT / 'data/review/heldout_output_audit_20260927'
VISIBLE = OUT_DIR / 'ReviewerA_HELDOUT_80_OUTPUT_REVIEW_20260927.csv'
HIDDEN = OUT_DIR / 'MACHINE_AND_SAMPLE_MANIFEST_20260927.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_key(row):
    k = row['key']
    value = '|'.join([k['fact_id'], k['language'], k['template'], k['variant'], 'heldout-output-audit-v1'])
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    if VISIBLE.exists() or HIDDEN.exists():
        raise FileExistsError('Held-out human output sample already frozen')
    facts = {f['fact_id']: f for f in
             (json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines())}
    paths = sorted(BASE.glob('e2-aya-independent-test-v1-shard*/*/items/*.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if len(rows) != 480 or len(facts) != 40 or any(r['key']['experiment'] != 'E2'
                                                    for r in rows):
        raise ValueError('Incomplete held-out E2 population')
    strata = defaultdict(list)
    for row in rows:
        key = row['key']
        stratum = (key['language'], row['fact']['relation'], key['variant'],
                   bool(row['evaluation']['entity_correct']))
        strata[stratum].append(row)
    selected = []
    for language in ('he', 'ar'):
        for relation in ('P19', 'P20', 'P159', 'P740'):
            for variant in ('U', 'D'):
                correct = sorted(strata[language, relation, variant, True], key=selection_key)
                wrong = sorted(strata[language, relation, variant, False], key=selection_key)
                n_correct = min(2, len(correct))
                n_wrong = min(5 - n_correct, len(wrong))
                if n_correct + n_wrong < 5:
                    n_correct = min(5 - n_wrong, len(correct))
                chosen = correct[:n_correct] + wrong[:n_wrong]
                if len(chosen) != 5:
                    raise ValueError('Cannot sample five rows per relation/language/variant')
                selected.extend(chosen)
    if len(selected) != 80 or len({selection_key(r) for r in selected}) != 80:
        raise ValueError('Held-out output sample has duplicates or wrong size')
    selected.sort(key=selection_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ['Audit ID', 'Language', 'Relation', 'Variant', 'Template',
              'Subject name', 'Gold place (English)', 'Gold place (question language)',
              'Question', 'Aya output', 'ReviewerA: entity correct (yes/no)',
              'ReviewerA: requested language correct (yes/no)', 'ReviewerA: error type',
              'ReviewerA: notes', 'Reviewer']
    machine = []
    with VISIBLE.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            k = row['key']
            fact = facts[k['fact_id']]
            if row['fact']['object_qid'] != fact['object_qid']:
                raise ValueError('Frozen fact identity changed')
            audit_id = selection_key(row)[:16]
            writer.writerow({
                'Audit ID': audit_id, 'Language': k['language'],
                'Relation': fact['relation'], 'Variant': k['variant'],
                'Template': k['template'], 'Subject name': row['prompt']['subject'],
                'Gold place (English)': fact['object_labels']['en'],
                'Gold place (question language)': fact['object_labels'][k['language']],
                'Question': row['prompt']['text'], 'Aya output': row['generation']['text'],
                'ReviewerA: entity correct (yes/no)': '',
                'ReviewerA: requested language correct (yes/no)': '',
                'ReviewerA: error type': '', 'ReviewerA: notes': '', 'Reviewer': '',
            })
            stratum = (k['language'], fact['relation'], k['variant'],
                       bool(row['evaluation']['entity_correct']))
            machine.append({
                'audit_id': audit_id, 'key': k, 'fact_id': k['fact_id'],
                'subject_qid': fact['subject_qid'], 'relation': fact['relation'],
                'machine_entity_correct': row['evaluation']['entity_correct'],
                'machine_category': row['evaluation']['category'],
                'machine_joint_language_flag': row['evaluation']['requested_language_correct'],
                'machine_all_aliases_correct': row['evaluation_all_aliases']['entity_correct'],
                'generation_truncated_flag': row['generation']['truncated'],
                'stratum': list(stratum), 'stratum_size': len(strata[stratum]),
                'sampled_in_stratum': sum(bool(x['evaluation']['entity_correct']) == stratum[3]
                                          for x in selected if
                                          (x['key']['language'], x['fact']['relation'], x['key']['variant'])
                                          == stratum[:3]),
            })
    HIDDEN.write_text(json.dumps({
        'schema_version': 1,
        'scope': 'Held-out E2 generated-output audit, machine judgments hidden from ReviewerA',
        'population': 480, 'sample': 80,
        'selection': 'Five per language/relation/U-D cell; up to two strict-machine-correct and remaining incorrect, hash order within each correctness stratum',
        'weighting': 'Inverse sample fraction within language/relation/variant/machine correctness stratum; fact-cluster variance',
        'facts_sha256': sha(FACTS),
        'visible_csv_sha256_before_review': sha(VISIBLE),
        'machine_rows': machine,
        'population_by_stratum': [
            {'language': a, 'relation': b, 'variant': c, 'machine_correct': d,
             'population_size': len(v)}
            for (a, b, c, d), v in sorted(strata.items())],
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'visible': str(VISIBLE), 'sample': len(selected),
                      'subjects': len({x['subject_qid'] for x in machine}),
                      'machine_manifest': str(HIDDEN)}))


if __name__ == '__main__':
    main()

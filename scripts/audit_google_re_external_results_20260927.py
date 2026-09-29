"""Audit the exact Google-RE Aya E1/E2 grid and pinned run identity."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/gre27/results'
FACTS = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.jsonl'
FREEZE = ROOT / 'data/curated/aya-google-re-external-reviewed-20260927.freeze.json'
OUT = ROOT / 'results/google-re-external-aya-grid-audit-20260927.json'
MODEL = 'CohereLabs/aya-23-8B'
REVISION = '89da1a0ed02d6130f93ae0ffdbedb63b760c0471'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    if OUT.exists():
        raise FileExistsError('External result grid audit already exists')
    frozen = read(FREEZE)
    if frozen['curated_sha256'] != sha(FACTS) or frozen['approved_count'] != 26:
        raise ValueError('Reviewed Google-RE cohort differs from its freeze')
    facts = {f['fact_id']: f for f in
             (json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines())}
    if len(facts) != 26 or len({f['subject_qid'] for f in facts.values()}) != 26:
        raise ValueError('Google-RE fact/subject count invalid')
    summary = {'scope': 'Independent exact-grid and model-identity audit of external Aya E1/E2',
               'reviewed_facts_sha256': sha(FACTS),
               'review_freeze_sha256': sha(FREEZE), 'model_id': MODEL,
               'revision': REVISION, 'experiments': {}}
    all_protocol_hashes = set()
    for experiment in ('E1', 'E2'):
        prefix = f'{experiment.lower()}-aya-google-re-external-20260927-shard'
        item_paths = sorted(BASE.glob(f'{prefix}*/*/items/*.json'))
        manifest_paths = sorted(BASE.glob(f'{prefix}*/*/manifest.json'))
        completion_paths = sorted(BASE.glob(f'{prefix}*/*/completion.json'))
        rows = [read(path) for path in item_paths]
        manifests = [read(path) for path in manifest_paths]
        completions = [read(path) for path in completion_paths]
        expected = {(fid, lang, f't{i}', variant)
                    for fid in facts
                    for lang in (('en', 'he', 'ar') if experiment == 'E1' else ('he', 'ar'))
                    for i in (1, 2, 3)
                    for variant in (('U',) if experiment == 'E1' else ('U', 'D'))}
        if (len(manifests) != 2 or len(completions) != 2 or len(rows) != len(expected) or
                len({m['shard'] for m in manifests}) != 2 or
                any(m['model']['model_id'] != MODEL or m['model']['revision'] != REVISION or
                    m['model']['precision'] != 'nf4' or m['experiment'] != experiment or
                    m['split'] != 'test' or m['data_kind'] != 'research'
                    for m in manifests) or
                any(c['status'] != 'complete' or c['experiment'] != experiment or
                    c['split'] != 'test' or c['total_items'] != c['expected_count']
                    for c in completions) or
                sum(c['total_items'] for c in completions) != len(expected)):
            raise ValueError(f'{experiment}: incomplete or mismatched archive/run identity')
        hashes = {m['protocol_hash'] for m in manifests}
        if len(hashes) != 1:
            raise ValueError(f'{experiment}: mixed protocol hashes')
        all_protocol_hashes.update(hashes)
        observed = set()
        strict = Counter()
        for row in rows:
            key = row['key']
            cell = (key['fact_id'], key['language'], key['template'], key['variant'])
            if cell in observed or cell not in expected:
                raise ValueError(f'{experiment}: duplicate or unexpected result cell')
            observed.add(cell)
            fact = facts[key['fact_id']]
            language = key['language']
            subject = (fact['subject_labels'][language] if experiment == 'E1'
                       else fact['pairs'][language][key['variant']])
            if (key['experiment'] != experiment or key['seed'] != 17 or
                    row['fact']['subject_qid'] != fact['subject_qid'] or
                    row['fact']['object_qid'] != fact['object_qid'] or
                    row['fact']['relation'] != fact['relation'] or
                    row['fact']['split'] != 'test' or
                    row['prompt']['subject'] != subject or
                    row['gold_score']['answer'] != fact['object_labels'][language] or
                    not math.isfinite(row['gold_score']['sum_logprob']) or
                    not isinstance(row['generation']['text'], str)):
                raise ValueError(f'{experiment}: fact, prompt, answer or score mismatch')
            if row['evaluation']['requested_language_correct'] and not row['evaluation']['entity_correct']:
                raise ValueError(f'{experiment}: scorer joint-language invariant broken')
            strict[(language, key['variant'])] += int(row['evaluation']['entity_correct'])
        if observed != expected:
            raise ValueError(f'{experiment}: exact key grid incomplete')
        summary['experiments'][experiment] = {
            'planned_cells': len(expected), 'returned_cells': len(rows),
            'manifests': len(manifests), 'completions': len(completions),
            'protocol_hash': next(iter(hashes)),
            'strict_correct_counts_by_language_variant':
                {f'{language}/{variant}': count for (language, variant), count in sorted(strict.items())},
        }
    if len(all_protocol_hashes) != 1:
        raise ValueError('E1/E2 used different frozen protocols')
    summary['status'] = 'complete_and_identity_verified'
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'cells': {
        e: v['returned_cells'] for e, v in summary['experiments'].items()}}, indent=2))


if __name__ == '__main__':
    main()

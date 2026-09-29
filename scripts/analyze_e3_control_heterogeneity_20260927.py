"""Descriptive relation and control-arm diagnostics for the byte-level E3."""

from collections import defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/b27/results'
OUT = ROOT / 'results/e3-byte-level-control-heterogeneity-20260927.json'


def main():
    paths = sorted(BASE.glob('e3-byte-level-20260927-shard*/*/items/*.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if len(rows) != 240:
        raise ValueError('Expected full 240-observation E3 run')
    indexed = defaultdict(dict)
    for row in rows:
        key = (row['key']['language'], row['fact']['relation'], row['key']['fact_id'])
        indexed[key][row['key']['variant']] = row['gold_score']['sum_logprob']
    grouped = defaultdict(list)
    for (language, relation, _), scores in indexed.items():
        if set(scores) != {'canonical', 'subject_split', 'outside_split'}:
            raise ValueError('Incomplete E3 conditions')
        grouped[(language, relation)].append({
            'inside_minus_outside': scores['subject_split'] - scores['outside_split'],
            'inside_minus_canonical': scores['subject_split'] - scores['canonical'],
            'outside_minus_canonical': scores['outside_split'] - scores['canonical'],
        })
    result = {'scope': 'Post-result descriptive control-arm heterogeneity; no new Aya run or new hypothesis test',
              'relation_means': {}, 'leave_one_relation_out_macro': {}}
    for language in ('he', 'ar'):
        relation_means = {}
        for relation in ('P19', 'P20', 'P159', 'P740'):
            values = grouped[(language, relation)]
            relation_means[relation] = {'n_subjects': len(values),
                                        **{name: statistics.mean(v[name] for v in values)
                                           for name in values[0]}}
        result['relation_means'][language] = relation_means
        result['leave_one_relation_out_macro'][language] = {
            omitted: statistics.mean(rel['inside_minus_outside']
                                     for relation, rel in relation_means.items() if relation != omitted)
            for omitted in relation_means}
    result['interpretation_limit'] = ('The outside control changes the same fixed template token for every '
                                      'subject in a language. Strong relation differences can therefore reflect '
                                      'interactions with relation wording or answer distribution, not a clean '
                                      'subject-token-count mechanism.')
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'relation_means': result['relation_means']}, indent=2))


if __name__ == '__main__':
    main()

"""Compare all three outside-control placements against one fixed E3 inside arm."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.analysis import cluster_effect  # noqa: E402

OLD = ROOT / 'results/b27/results'
NEW = ROOT / 'results/e3p27/results'
PROTOCOL = ROOT / 'configs/e3_position_controls_protocol_20260927.json'
OUT = ROOT / 'results/e3-position-controls-analysis-20260927.json'


def read_rows(base, pattern, expected_count, experiment):
    paths = sorted(base.glob(pattern + '/*/items/*.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    completions = [json.loads(path.read_text(encoding='utf-8')) for path in
                   sorted(base.glob(pattern + '/*/completion.json'))]
    if (len(rows) != expected_count or len(completions) != 2 or
            sum(c['total_items'] for c in completions) != expected_count or
            any(c['status'] != 'complete' or c['total_items'] != c['expected_count'] or
                c['experiment'] != experiment for c in completions)):
        raise ValueError('Incomplete E3 result collection: ' + experiment)
    return rows


def effect(pairs):
    values = [b['gold_score']['sum_logprob'] - a['gold_score']['sum_logprob'] for a, b in pairs]
    subjects = [b['fact']['subject_qid'] for _, b in pairs]
    relations = [b['fact']['relation'] for _, b in pairs]
    return cluster_effect(values, subjects, relations, 10000, 17)


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    prior = read_rows(OLD, 'e3-byte-level-20260927-shard*', 240, 'E3_byte_level')
    newer = read_rows(NEW, 'e3-position-controls-20260927-shard*', 160,
                      'E3_position_controls')
    if len({(r['key']['fact_id'], r['key']['language'], r['key']['variant'])
            for r in newer}) != 160:
        raise ValueError('Duplicate positional-control rows')
    report = {'scope': protocol['method_status'],
              'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'original_records': len(prior), 'new_records': len(newer), 'languages': [],
              'limits': [
                  'Controls were chosen after inspecting the first byte-level E3 result; this is exploratory sensitivity, not a new confirmatory hypothesis test.',
                  'All controls preserve rendered text and add exactly one token; token IDs, attention positions and context still change.',
                  'The three outside controls affect three distinct template tokens, fixed across facts within each language.',
                  'Intervals for several correlated comparisons are descriptive; no favorable control is selected as the primary causal estimate.',
              ]}
    for language in ('he', 'ar'):
        selected = [r for r in prior + newer if r['key']['language'] == language]
        index = {(r['key']['fact_id'], r['key']['variant']): r for r in selected}
        facts = sorted({r['key']['fact_id'] for r in selected})
        if len(facts) != 40 or len(index) != 200:
            raise ValueError('Missing condition in E3 positional analysis')
        conditions = ['canonical', 'subject_split', 'outside_split',
                      'nearest_before', 'nearest_after']
        contrasts = {}
        for outside in conditions[2:]:
            pairs = [(index[(fact, outside)], index[(fact, 'subject_split')])
                     for fact in facts]
            for a, b in pairs:
                if (a['prompt']['text'] != b['prompt']['text'] or
                        a['gold_score']['answer'] != b['gold_score']['answer'] or
                        len(a['prompt']['input_ids']) != len(b['prompt']['input_ids'])):
                    raise ValueError('Same-text, same-answer, equal-length invariant failed')
            contrasts[f'inside_minus_{outside}'] = effect(pairs)
            base = [(index[(fact, 'canonical')], index[(fact, outside)])
                    for fact in facts]
            if any(len(a['prompt']['input_ids']) + 1 != len(b['prompt']['input_ids'])
                   for a, b in base):
                raise ValueError('Outside control does not add exactly one token')
            contrasts[f'{outside}_minus_canonical'] = effect(base)
        by_relation = defaultdict(dict)
        for relation in ('P19', 'P20', 'P159', 'P740'):
            ids = [fact for fact in facts if index[(fact, 'canonical')]['fact']['relation'] == relation]
            for outside in conditions[2:]:
                differences = [index[(fact, 'subject_split')]['gold_score']['sum_logprob'] -
                               index[(fact, outside)]['gold_score']['sum_logprob'] for fact in ids]
                by_relation[relation][f'inside_minus_{outside}_mean_nats'] = (
                    sum(differences) / len(differences) if differences else None)
            by_relation[relation]['n_facts'] = len(ids)
        report['languages'].append({
            'language': language, 'facts': len(facts),
            'relations': dict(Counter(index[(fact, 'canonical')]['fact']['relation'] for fact in facts)),
            'contrasts': contrasts, 'relation_diagnostics': dict(by_relation),
        })
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'summary': [
        {'language': item['language'],
         'inside_minus_outside_by_control': {name: item['contrasts'][f'inside_minus_{name}']['estimate']
                                             for name in ('outside_split', 'nearest_before', 'nearest_after')}}
        for item in report['languages']]}, indent=2))


if __name__ == '__main__':
    main()

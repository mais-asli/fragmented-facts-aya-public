"""Join frozen, outcome-unseen external matches to returned Aya E2 outcomes."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
MATCH = ROOT / 'results/google-re-pageviews-2023-pre-aya-match-20260927.json'
BASE = ROOT / 'results/gre27/results'
OUT = ROOT / 'results/google-re-pageviews-2023-match-analysis-20260927.json'


def cluster_interval(pairs, key, seed):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair['relation']].append(pair[key])
    if not groups:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        by_relation = [statistics.mean(rng.choice(values) for _ in values)
                       for values in groups.values()]
        draws.append(statistics.mean(by_relation))
    draws.sort()
    return [draws[249], draws[9749]]


def main():
    if OUT.exists():
        raise FileExistsError('External matched analysis already exists')
    match = json.loads(MATCH.read_text(encoding='utf-8'))
    if match['status'] != 'Assigned before inspecting any Google-RE Aya outputs':
        raise ValueError('External match assignment is not frozen')
    paths = sorted(BASE.glob('e2-aya-google-re-external-20260927-shard*/*/items/*.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    completions = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(
        BASE.glob('e2-aya-google-re-external-20260927-shard*/*/completion.json'))]
    if (len(rows) != 312 or len(completions) != 2 or
            sum(c['total_items'] for c in completions) != 312 or
            any(c['status'] != 'complete' or c['total_items'] != c['expected_count'] or
                c['experiment'] != 'E2' or c['split'] != 'test' for c in completions)):
        raise ValueError('External E2 result grid incomplete')
    index = {(r['key']['fact_id'], r['key']['language'], r['key']['template'],
              r['key']['variant']): r for r in rows}
    if len(index) != 312:
        raise ValueError('Duplicate external E2 key')
    summary = {}
    for language in ('he', 'ar'):
        assigned = [p for g in match['groups'] for p in g['pairs'] if p['language'] == language]
        pair_rows = []
        for pair in assigned:
            values = {}
            for side in ('high', 'low'):
                fid = pair[f'{side}_fact_id']
                u = [index[(fid, language, f't{i}', 'U')] for i in (1, 2, 3)]
                d = [index[(fid, language, f't{i}', 'D')] for i in (1, 2, 3)]
                values[side] = {
                    'u_accuracy': statistics.mean(int(x['evaluation']['entity_correct']) for x in u),
                    'u_gold_loglik': statistics.mean(x['gold_score']['sum_logprob'] for x in u),
                    'd_minus_u_accuracy': statistics.mean(
                        int(y['evaluation']['entity_correct']) - int(x['evaluation']['entity_correct'])
                        for x, y in zip(u, d)),
                    'd_minus_u_gold_loglik': statistics.mean(
                        y['gold_score']['sum_logprob'] - x['gold_score']['sum_logprob']
                        for x, y in zip(u, d)),
                }
            pair_rows.append({**pair, **{
                f'high_minus_low_{metric}': values['high'][metric] - values['low'][metric]
                for metric in values['high']}})
        relation_groups = defaultdict(list)
        for pair in pair_rows:
            relation_groups[pair['relation']].append(pair)
        effects = {}
        for metric in ('u_accuracy', 'u_gold_loglik',
                       'd_minus_u_accuracy', 'd_minus_u_gold_loglik'):
            key = 'high_minus_low_' + metric
            relation_means = {relation: statistics.mean(row[key] for row in group)
                              for relation, group in relation_groups.items()}
            effects[metric] = {
                'relation_macro_estimate': statistics.mean(relation_means.values())
                    if relation_means else None,
                'relation_means': relation_means,
                'relation_stratified_pair_bootstrap_95_interval': cluster_interval(
                    pair_rows, key, 17 + ('he', 'ar').index(language) * 100 +
                    ('u_accuracy', 'u_gold_loglik', 'd_minus_u_accuracy',
                     'd_minus_u_gold_loglik').index(metric)),
            }
        summary[language] = {'matched_pairs': len(pair_rows),
                             'relations_with_pairs': {r: len(v) for r, v in relation_groups.items()},
                             'effects': effects, 'pairs': pair_rows}
    payload = {
        'scope': 'Outcome-unseen pairing, Aya-only Google-RE external archive; descriptive matched analysis',
        'match_sha256': hashlib.sha256(MATCH.read_bytes()).hexdigest(),
        'external_e2_records': len(rows), 'languages': summary,
        'limits': ['Only 26 personally reviewed P19/P20 facts; 15 language-specific matched pairs.',
                   'The two language samples share some subjects, so cross-language intervals are not independent.',
                   'The relation-stratified intervals condition on the selected pairs and omit selection uncertainty.',
                   'Pageviews proxy public attention and cannot identify Aya pretraining counts.',
                   'Matched subjects differ in identity, name content and token IDs; no pure token-count causal claim.',
                   'No confirmatory p-value is estimated from this small matched cohort.'],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'pairs': {language: item['matched_pairs']
                                                  for language, item in summary.items()}}))


if __name__ == '__main__':
    main()

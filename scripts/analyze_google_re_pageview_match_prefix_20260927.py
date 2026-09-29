"""Apply the existing exact Arabic-prefix sensitivity to frozen external pairs."""

from collections import defaultdict
import json
from pathlib import Path
import statistics

from analyze_google_re_pageview_match_20260927 import cluster_interval

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/gre27/results'
MATCH = ROOT / 'results/google-re-pageviews-2023-pre-aya-match-20260927.json'
FORMAT = ROOT / 'results/answer-format-sensitivity-google-re-external-20260927.json'
OUT = ROOT / 'results/google-re-pageviews-2023-match-prefix-sensitivity-20260927.json'


def key_tuple(key):
    return (key['fact_id'], key['language'], key['template'], key['variant'])


def main():
    if OUT.exists():
        raise FileExistsError('External matched prefix sensitivity already exists')
    frozen = json.loads(MATCH.read_text(encoding='utf-8'))
    fmt = json.loads(FORMAT.read_text(encoding='utf-8'))
    if (frozen['status'] != 'Assigned before inspecting any Google-RE Aya outputs' or
            fmt['records'] != 546):
        raise ValueError('Matching or format-sensitivity inputs invalid')
    rescued = {key_tuple(r['key']) for r in fmt['rescue_candidates_for_human_check']
               if r['rule'] == 'arabic_prefix_exact' and r['key']['experiment'] == 'E2'}
    paths = sorted(BASE.glob('e2-aya-google-re-external-20260927-shard*/*/items/*.json'))
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    index = {key_tuple(r['key']): r for r in rows}
    if len(rows) != 312 or len(index) != 312 or len(rescued) != 21:
        raise ValueError('External E2 grid or exact-prefix rescues incomplete')
    result = {}
    for language in ('he', 'ar'):
        pairs = [pair for group in frozen['groups'] for pair in group['pairs']
                 if pair['language'] == language]
        items = []
        for pair in pairs:
            scores = {}
            for side in ('high', 'low'):
                fid = pair[f'{side}_fact_id']
                u = [index[(fid, language, f't{i}', 'U')] for i in (1, 2, 3)]
                d = [index[(fid, language, f't{i}', 'D')] for i in (1, 2, 3)]
                def correct(row):
                    return int(row['evaluation']['entity_correct'] or
                               key_tuple(row['key']) in rescued)
                scores[side] = {
                    'u_prefix_sensitive_accuracy': statistics.mean(map(correct, u)),
                    'd_minus_u_prefix_sensitive_accuracy': statistics.mean(
                        correct(drow) - correct(urow) for urow, drow in zip(u, d)),
                }
            items.append({**pair, **{
                f'high_minus_low_{name}': scores['high'][name] - scores['low'][name]
                for name in scores['high']}})
        relation_groups = defaultdict(list)
        for item in items:
            relation_groups[item['relation']].append(item)
        effects = {}
        for index_metric, name in enumerate(('u_prefix_sensitive_accuracy',
                                              'd_minus_u_prefix_sensitive_accuracy')):
            key = f'high_minus_low_{name}'
            means = {rel: statistics.mean(item[key] for item in group)
                     for rel, group in relation_groups.items()}
            effects[name] = {
                'relation_macro_estimate': statistics.mean(means.values()),
                'relation_means': means,
                'conditional_pair_bootstrap_95_interval': cluster_interval(
                    items, key, 317 + 10 * ('he', 'ar').index(language) + index_metric),
            }
        result[language] = {'matched_pairs': len(items), 'effects': effects}
    OUT.write_text(json.dumps({
        'scope': 'Post-hoc application of preexisting exact Arabic answer-prefix rule to outcome-unseen external pairs',
        'primary_unchanged': True, 'arabic_e2_exact_prefix_rescues': len(rescued),
        'languages': result,
        'limits': ['The pairs were fixed without Aya outcomes, but this prefix-sensitive pairing summary was computed after seeing the strict floor.',
                   'Only an exact anchored Arabic prefix followed by a complete reviewed canonical place is rescued.',
                   'Bootstrap intervals condition on fixed pairs, omit matching uncertainty, and are descriptive.'],
    }, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

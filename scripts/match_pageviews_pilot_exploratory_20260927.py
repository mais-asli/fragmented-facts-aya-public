"""Retrospective pilot-only pageview match, kept separate from held-out pairs."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

from match_fragmentation_heldout_exploratory import pilot_thresholds
from match_pageviews_heldout_exploratory_20260927 import match_group

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/tau_research_20260926/results'
PAGEVIEWS = ROOT / 'results/pageviews-2023-frozen-97.csv'
OUT = ROOT / 'results/fragmentation-pageviews-2023-match-pilot-exploratory-20260927.json'


def main():
    paths = sorted(SOURCE.glob('e2-reviewed-pilot-20260926-shard*/*/items/*.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if len(rows) != 684:
        raise ValueError('Expected 684 pilot E2 records')
    index = {(r['key']['fact_id'], r['key']['language'], r['key']['template'], r['key']['variant']): r
             for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate pilot E2 key')
    with PAGEVIEWS.open(encoding='utf-8', newline='') as file:
        views = {r['fact_id']: r for r in csv.DictReader(file)}
    thresholds = pilot_thresholds()
    entities = []
    for language in ('he', 'ar'):
        fact_ids = sorted({key[0] for key in index if key[1] == language})
        if len(fact_ids) != 57:
            raise ValueError('Pilot E2 fact coverage differs')
        for fid in fact_ids:
            t1 = index[fid, language, 't1', 'U']
            relation = t1['fact']['relation']
            if views[fid]['split'] != 'pilot':
                raise ValueError('Historical view split mismatch')
            count = int(views[fid]['views_2023']) if views[fid]['views_2023'] else None
            u = [index[fid, language, f't{i}', 'U'] for i in (1, 2, 3)]
            d = [index[fid, language, f't{i}', 'D'] for i in (1, 2, 3)]
            density = t1['prompt']['tokens_per_base_character']
            entities.append({'fact_id': fid, 'language': language, 'relation': relation,
                             'views_2023': count,
                             'base_characters': t1['prompt']['base_characters'],
                             'u_token_density_t1': density,
                             'group': 'high' if density > thresholds[language, relation] else 'low',
                             'u_accuracy_three_templates': statistics.mean(
                                 int(row['evaluation']['entity_correct']) for row in u),
                             'u_gold_loglik_three_templates': statistics.mean(
                                 row['gold_score']['sum_logprob'] for row in u),
                             'd_minus_u_accuracy_three_templates': statistics.mean(
                                 int(a['evaluation']['entity_correct']) - int(b['evaluation']['entity_correct'])
                                 for a, b in zip(d, u))})
    groups, all_pairs = [], []
    for language in ('he', 'ar'):
        for relation in ('P19', 'P20', 'P159', 'P740'):
            candidates = [r for r in entities if r['language'] == language and r['relation'] == relation]
            matched = match_group([r for r in candidates if r['views_2023'] is not None])
            pairs = [{
                'high_fact_id': h['fact_id'], 'low_fact_id': l['fact_id'],
                'high_minus_low_u_accuracy': h['u_accuracy_three_templates'] - l['u_accuracy_three_templates'],
                'high_minus_low_u_gold_loglik': h['u_gold_loglik_three_templates'] - l['u_gold_loglik_three_templates'],
                'high_minus_low_d_minus_u_accuracy': h['d_minus_u_accuracy_three_templates'] - l['d_minus_u_accuracy_three_templates'],
            } for h, l in matched]
            groups.append({'language': language, 'relation': relation, 'subjects': len(candidates),
                           'missing_views': sum(r['views_2023'] is None for r in candidates),
                           'matched_pairs': len(pairs), 'pairs': pairs})
            all_pairs += [{**pair, 'language': language, 'relation': relation} for pair in pairs]
    report = {'scope': 'Post-outcome retrospective pilot-only association; not an independent test',
              'source_sha256': {'pageviews_csv': hashlib.sha256(PAGEVIEWS.read_bytes()).hexdigest()},
              'method': 'Same frozen pilot relation/language medians and pageview/length calipers as held-out analysis; within pilot only, no replacement',
              'groups': groups,
              'summary': {language: {
                  'matched_pairs': len(selected := [p for p in all_pairs if p['language'] == language]),
                  'relations_with_pairs': dict(Counter(p['relation'] for p in selected)),
                  'mean_high_minus_low_u_accuracy': statistics.mean(p['high_minus_low_u_accuracy'] for p in selected) if selected else None,
              } for language in ('he', 'ar')},
              'limit': 'The pilot supplies the median cutpoints and also contributes its own matched outcome pairs. This is descriptive and may be overfit.'}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'summary': report['summary']}))


if __name__ == '__main__':
    main()

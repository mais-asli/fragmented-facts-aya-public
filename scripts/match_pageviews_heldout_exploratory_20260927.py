"""Retrospective held-out high/low token-density match with 2023 pageviews.

Matches use only historical pageviews and name length, within relation and
language. Aya outcomes are joined after assignment. This cannot identify a
causal token-count effect because entity and token identities still differ.
"""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from match_fragmentation_heldout_exploratory import (
    PILOT_TOKENS, TEST_ANALYSIS, load_test, pilot_thresholds, subjects,
)

ROOT = Path(__file__).resolve().parents[1]
PAGEVIEWS = ROOT / 'results/pageviews-2023-frozen-97.csv'
PAGEVIEWS_MANIFEST = ROOT / 'results/pageviews-2023-frozen-97.manifest.json'
OUT = ROOT / 'results/fragmentation-pageviews-2023-match-heldout-exploratory-20260927.json'
LOG_VIEW_CALIPER = 1.5
LENGTH_CALIPER = 5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_group(rows):
    high = sorted((r for r in rows if r['group'] == 'high'), key=lambda r: r['fact_id'])
    low = sorted((r for r in rows if r['group'] == 'low'), key=lambda r: r['fact_id'])
    if not high or not low:
        return []
    cost = np.full((len(high), len(low) + len(high)), 1000.0)
    for i, h in enumerate(high):
        for j, l in enumerate(low):
            view_gap = abs(math.log1p(h['views_2023']) - math.log1p(l['views_2023']))
            length_gap = abs(h['base_characters'] - l['base_characters'])
            if view_gap <= LOG_VIEW_CALIPER and length_gap <= LENGTH_CALIPER:
                cost[i, j] = (view_gap / LOG_VIEW_CALIPER + length_gap / LENGTH_CALIPER) / 3
        cost[i, len(low) + i] = 100.0
    ii, jj = linear_sum_assignment(cost)
    result = []
    for i, j in zip(ii, jj):
        if j >= len(low):
            continue
        if cost[i, j] >= 1:
            raise ValueError('Infeasible pair selected')
        result.append((high[i], low[j]))
    return result


def main():
    if OUT.exists():
        raise FileExistsError('Historical-view matching report already exists')
    manifest = json.loads(PAGEVIEWS_MANIFEST.read_text(encoding='utf-8'))
    if manifest['result_csv_sha256'] != sha(PAGEVIEWS):
        raise ValueError('Pageview evidence differs from the saved manifest')
    with PAGEVIEWS.open(encoding='utf-8', newline='') as source:
        pageviews = {r['fact_id']: r for r in csv.DictReader(source)}
    if len(pageviews) != 97:
        raise ValueError('Expected 97 facts in historical pageview ledger')
    thresholds = pilot_thresholds()
    entities = subjects(load_test(), thresholds)
    for entity in entities:
        record = pageviews[entity['fact_id']]
        if record['split'] != 'test':
            raise ValueError('Pageview/test split mismatch')
        entity['views_2023'] = int(record['views_2023']) if record['views_2023'] else None
    groups = []
    all_pairs = []
    for language in ('he', 'ar'):
        for relation in ('P19', 'P20', 'P159', 'P740'):
            group = [r for r in entities if r['language'] == language and r['relation'] == relation]
            observed = [r for r in group if r['views_2023'] is not None]
            matched = match_group(observed)
            pairs = []
            for high, low in matched:
                pair = {
                    'language': language, 'relation': relation,
                    'high_fact_id': high['fact_id'], 'low_fact_id': low['fact_id'],
                    'high_views_2023': high['views_2023'],
                    'low_views_2023': low['views_2023'],
                    'abs_log1p_views_gap': abs(math.log1p(high['views_2023']) - math.log1p(low['views_2023'])),
                    'abs_base_character_gap': abs(high['base_characters'] - low['base_characters']),
                    'high_minus_low_token_density': high['u_token_density_t1'] - low['u_token_density_t1'],
                    'high_minus_low_u_accuracy': high['u_accuracy_three_templates'] - low['u_accuracy_three_templates'],
                    'high_minus_low_u_gold_loglik': high['u_gold_loglik_three_templates'] - low['u_gold_loglik_three_templates'],
                    'high_minus_low_d_minus_u_accuracy': high['d_minus_u_accuracy_three_templates'] - low['d_minus_u_accuracy_three_templates'],
                }
                pairs.append(pair)
                all_pairs.append(pair)
            groups.append({
                'language': language, 'relation': relation, 'subjects': len(group),
                'missing_views': len(group) - len(observed),
                'high_with_views': sum(r['group'] == 'high' for r in observed),
                'low_with_views': sum(r['group'] == 'low' for r in observed),
                'matched_pairs': len(pairs), 'pairs': pairs,
            })
    summary = {}
    for language in ('he', 'ar'):
        pairs = [p for p in all_pairs if p['language'] == language]
        summary[language] = {
            'matched_pairs': len(pairs),
            'relations_with_pairs': dict(Counter(p['relation'] for p in pairs)),
            'mean_abs_log1p_views_gap': float(np.mean([p['abs_log1p_views_gap'] for p in pairs])) if pairs else None,
            'mean_high_minus_low_u_accuracy': float(np.mean([p['high_minus_low_u_accuracy'] for p in pairs])) if pairs else None,
            'mean_high_minus_low_u_gold_loglik': float(np.mean([p['high_minus_low_u_gold_loglik'] for p in pairs])) if pairs else None,
        }
    report = {
        'scope': 'Post-outcome retrospective exploratory match; no causal identification',
        'source_sha256': {'pilot_tokenization': sha(PILOT_TOKENS),
                          'heldout_analysis': sha(TEST_ANALYSIS),
                          'pageviews_csv': sha(PAGEVIEWS),
                          'pageviews_manifest': sha(PAGEVIEWS_MANIFEST)},
        'method': {'population': '40 held-out mLAMA subjects per language',
                   'fragmentation_rule': 'U tokens/base character above the pilot relation/language median',
                   'matching': 'within relation/language, no replacement, maximum cardinality then minimum distance',
                   'distance_variables': ['log1p 2023 enwiki pageviews', 'unvocalized base-character length'],
                   'log1p_pageviews_caliper': LOG_VIEW_CALIPER,
                   'base_characters_caliper': LENGTH_CALIPER,
                   'missing': 'excluded without imputation',
                   'outcome': 'three-template subject means from saved Aya E2 results'},
        'pageviews': {'complete_facts': manifest['complete_2023_subjects'],
                      'missing_by_reason': manifest['missing_by_reason']},
        'groups': groups, 'summary': summary,
        'warning': 'Pageviews are an imperfect article-traffic proxy, not Aya pretraining frequency. '
                   'Different subjects have different names, spelling, semantics, and token IDs. '
                   'Matched estimates describe association only; do not call token count causal.',
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(OUT), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()

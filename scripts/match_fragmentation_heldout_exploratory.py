"""Exploratory relation/frequency-matched Aya recall comparison.

Pilot-only median thresholds define low/high *unmarked* token density. Held-out
matches are formed from names, sitelinks and length, without using Aya outputs.
Outcomes are then joined from the already saved E2 rows. Because this analysis
was devised after those outputs existed, it is explicitly retrospective.
"""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
PILOT_TOKENS = ROOT / 'outputs/01a09b51/reviewed_tokenization_20260926/reviewed_57_prompt_tokenization.csv'
TEST_ANALYSIS = ROOT / 'results/heldout/results/analysis-aya-independent-test-v1.json'
TEST_ROOT = ROOT / 'results/heldout'
OUT = ROOT / 'results/fragmentation-frequency-match-heldout-exploratory-20260927.json'
SITELINK_LOG_CALIPER = 1.5
BASE_CHAR_CALIPER = 5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_test():
    analysis = json.loads(TEST_ANALYSIS.read_text(encoding='utf-8'))
    if analysis['data_kind'] != 'research' or analysis['model']['model_id'] != 'CohereLabs/aya-23-8B':
        raise ValueError('Aya research result required')
    rows = {}
    for run in analysis['run_paths']:
        if '/e2-' not in '/' + run:
            continue
        for path in (TEST_ROOT / run / 'items').glob('*.json'):
            row = json.loads(path.read_text(encoding='utf-8'))
            key = row['key']
            if key['experiment'] != 'E2' or key['language'] not in ('he', 'ar'):
                continue
            tuple_key = (key['fact_id'], key['language'], key['template'], key['variant'])
            if tuple_key in rows:
                raise ValueError('Duplicate E2 row')
            rows[tuple_key] = row
    if len(rows) != 480:
        raise ValueError(f'Expected 480 E2 records, got {len(rows)}')
    return rows


def pilot_thresholds():
    values = defaultdict(list)
    with PILOT_TOKENS.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['template'] == 't1':
                values[row['language'], row['relation']].append(
                    int(row['u_subject_tokens']) / int(row['base_length']))
    if any(len(v) not in (14, 15) for v in values.values()) or len(values) != 8:
        raise ValueError('Pilot threshold cohort is incomplete')
    return {key: float(np.median(value)) for key, value in values.items()}


def subjects(rows, thresholds):
    entities = []
    for language in ('he', 'ar'):
        fact_ids = sorted({fact_id for fact_id, lang, _, _ in rows if lang == language})
        if len(fact_ids) != 40:
            raise ValueError('Test fact coverage mismatch')
        for fact_id in fact_ids:
            t1 = rows[fact_id, language, 't1', 'U']
            rel = t1['fact']['relation']
            density = float(t1['prompt']['tokens_per_base_character'])
            threshold = thresholds[language, rel]
            u = [rows[fact_id, language, f't{i}', 'U'] for i in (1, 2, 3)]
            d = [rows[fact_id, language, f't{i}', 'D'] for i in (1, 2, 3)]
            entities.append({
                'fact_id': fact_id, 'language': language, 'relation': rel,
                'sitelinks': int(t1['popularity']['sitelinks']),
                'base_characters': int(t1['prompt']['base_characters']),
                'u_tokens_t1': int(t1['prompt']['subject_tokens']),
                'u_token_density_t1': density,
                'pilot_relation_median_density': threshold,
                'group': 'high' if density > threshold else 'low',
                'u_accuracy_three_templates': float(np.mean([int(x['evaluation']['entity_correct']) for x in u])),
                'd_minus_u_accuracy_three_templates': float(np.mean([
                    int(y['evaluation']['entity_correct']) - int(x['evaluation']['entity_correct'])
                    for x, y in zip(u, d)])),
                'u_gold_loglik_three_templates': float(np.mean([x['gold_score']['sum_logprob'] for x in u])),
                'd_minus_u_gold_loglik_three_templates': float(np.mean([
                    y['gold_score']['sum_logprob'] - x['gold_score']['sum_logprob'] for x, y in zip(u, d)])),
            })
    return entities


def match_group(candidates):
    high = [r for r in candidates if r['group'] == 'high']
    low = [r for r in candidates if r['group'] == 'low']
    if not high or not low:
        return []
    # Feasible edges have cost < 1; infeasible edges are assigned a very large
    # cost. Dummy columns permit unmatched highs; the 100-point dummy penalty
    # prioritizes the number of valid pairs over within-pair closeness.
    cost = np.full((len(high), len(low) + len(high)), 100.0)
    for i, h in enumerate(high):
        for j, l in enumerate(low):
            sitelink_gap = abs(math.log1p(h['sitelinks']) - math.log1p(l['sitelinks']))
            length_gap = abs(h['base_characters'] - l['base_characters'])
            if sitelink_gap <= SITELINK_LOG_CALIPER and length_gap <= BASE_CHAR_CALIPER:
                cost[i, j] = (sitelink_gap / SITELINK_LOG_CALIPER +
                              length_gap / BASE_CHAR_CALIPER) / 3
            else:
                cost[i, j] = 1000.0
    for i in range(len(high)):
        cost[i, len(low) + i] = 100.0
    selected_i, selected_j = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(selected_i, selected_j):
        if j >= len(low):
            continue
        if cost[i, j] >= 1:
            raise ValueError('Infeasible pair selected')
        h, l = high[i], low[j]
        pairs.append({
            'high_fact_id': h['fact_id'], 'low_fact_id': l['fact_id'],
            'high_sitelinks': h['sitelinks'], 'low_sitelinks': l['sitelinks'],
            'high_base_characters': h['base_characters'], 'low_base_characters': l['base_characters'],
            'high_u_token_density': h['u_token_density_t1'],
            'low_u_token_density': l['u_token_density_t1'],
            'high_minus_low_u_accuracy': h['u_accuracy_three_templates'] - l['u_accuracy_three_templates'],
            'high_minus_low_u_gold_loglik': h['u_gold_loglik_three_templates'] - l['u_gold_loglik_three_templates'],
            'high_minus_low_d_minus_u_accuracy': h['d_minus_u_accuracy_three_templates'] - l['d_minus_u_accuracy_three_templates'],
            'high_minus_low_d_minus_u_gold_loglik': h['d_minus_u_gold_loglik_three_templates'] - l['d_minus_u_gold_loglik_three_templates'],
        })
    return pairs


def main():
    thresholds = pilot_thresholds()
    entities = subjects(load_test(), thresholds)
    groups = []
    for language in ('he', 'ar'):
        for relation in ('P19', 'P20', 'P159', 'P740'):
            cells = [r for r in entities if r['language'] == language and r['relation'] == relation]
            pairs = match_group(cells)
            balance = None
            if pairs:
                balance = {
                    'mean_abs_log1p_sitelink_gap': float(np.mean([
                        abs(math.log1p(p['high_sitelinks']) - math.log1p(p['low_sitelinks']))
                        for p in pairs])),
                    'mean_abs_base_character_gap': float(np.mean([
                        abs(p['high_base_characters'] - p['low_base_characters'])
                        for p in pairs])),
                    'mean_high_minus_low_u_token_density': float(np.mean([
                        p['high_u_token_density'] - p['low_u_token_density']
                        for p in pairs])),
                    'mean_high_minus_low_u_accuracy': float(np.mean([
                        p['high_minus_low_u_accuracy'] for p in pairs])),
                    'mean_high_minus_low_u_gold_loglik': float(np.mean([
                        p['high_minus_low_u_gold_loglik'] for p in pairs])),
                }
            groups.append({'language': language, 'relation': relation,
                           'subjects': len(cells),
                           'high_count': sum(r['group'] == 'high' for r in cells),
                           'low_count': sum(r['group'] == 'low' for r in cells),
                           'matched_pairs': len(pairs), 'balance_and_outcome_descriptives': balance,
                           'pairs': pairs})
    output = {
        'scope': 'Post-outcome exploratory observational comparison; not a prospective test or causal identification',
        'source_hashes': {'pilot_tokenization': sha(PILOT_TOKENS), 'heldout_analysis': sha(TEST_ANALYSIS)},
        'method': {'threshold': 'pilot relation/language median of U tokens per base character at t1',
                   'high_rule': 'strictly greater than pilot median',
                   'matching': 'within language/relation, no replacement, maximum-cardinality/minimum-distance assignment using sitelinks and base-letter length only',
                   'log1p_sitelinks_caliper': SITELINK_LOG_CALIPER,
                   'base_characters_caliper': BASE_CHAR_CALIPER,
                   'outcome': 'three-template mean per subject; no hypothesis test due small matched sample'},
        'pilot_thresholds': {f'{a}/{b}': v for (a, b), v in sorted(thresholds.items())},
        'test_subjects': len(entities), 'groups': groups,
        'warning': 'Sitelinks are a coarse popularity proxy, not Aya training frequency. Matched entities differ in identity, spelling and semantics. Matching cannot isolate token count.'
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(OUT)
    for group in groups:
        print(group['language'], group['relation'], group['matched_pairs'], '/', group['subjects'])


if __name__ == '__main__':
    main()

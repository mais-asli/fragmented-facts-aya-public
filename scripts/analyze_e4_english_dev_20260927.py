"""Descriptive, subject-clustered analysis of the Aya English E4 dev sweep."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/e427/results'
FACTS = ROOT / 'data/curated/aya-e4-english-dev-ReviewerA-reviewed-20260927.jsonl'
SCREENED = ROOT / 'results/e427/data/curated/aya-e4-english-dev-screened-20260927.jsonl'
OUT = ROOT / 'results/e4-english-dev-analysis-20260927.json'
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]
SITES = ['first_subject', 'last_subject', 'prediction']


def interval(values, relations, seed):
    if len(values) != len(relations) or not values:
        raise ValueError('Missing E4 values')
    groups = defaultdict(list)
    for value, relation in zip(values, relations):
        groups[relation].append(value)
    rng = random.Random(seed)
    trials = []
    for _ in range(10000):
        drawn = [rng.choice(group) for group in groups.values() for _ in group]
        trials.append(statistics.mean(drawn))
    trials.sort()
    return [trials[249], trials[9749]]


def main():
    paths = sorted(BASE.glob('e4-english-dev-20260927-shard*/*/items/*.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    completions = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(
        BASE.glob('e4-english-dev-20260927-shard*/*/completion.json'))]
    if len(completions) != 2 or any(c['status'] != 'complete' or c['experiment'] != 'E4'
                                    or c['total_items'] != c['expected_count'] for c in completions):
        raise ValueError('Incomplete English E4 shards')
    if sum(c['total_items'] for c in completions) != len(rows):
        raise ValueError('E4 item count disagrees with shard receipts')
    reviewed = {json.loads(line)['fact_id'] for line in FACTS.read_text(encoding='utf-8').splitlines()}
    screened = [json.loads(line) for line in SCREENED.read_text(encoding='utf-8').splitlines()]
    eligible = {f['fact_id'] for f in screened if f.get('core_member')}
    if not 10 <= len(eligible) <= len(reviewed) or not eligible <= reviewed:
        raise ValueError('E4 screened core differs from reviewed development cohort')
    index = {(r['key']['fact_id'], r['key']['layer'], r['key']['site']): r for r in rows}
    expected = {(fact, layer, site) for fact in eligible for layer in LAYERS for site in SITES}
    if set(index) != expected or len(rows) != len(index):
        raise ValueError('E4 layer/site grid incomplete or contains duplicates')
    relation_by_fact = {f['fact_id']: f['relation'] for f in screened if f['fact_id'] in eligible}
    cells = []
    for site in SITES:
        for layer in LAYERS:
            chosen = [index[(fact, layer, site)] for fact in sorted(eligible)]
            values = [r['restoration_gain'] for r in chosen]
            relations = [relation_by_fact[fact] for fact in sorted(eligible)]
            valid = [r['readout']['first_token_margin'] for r in chosen if r['readout']['eligible']]
            cells.append({'site': site, 'layer': layer, 'n_subjects': len(chosen),
                          'restoration_gain_mean_nats': statistics.mean(values),
                          'restoration_gain_median_nats': statistics.median(values),
                          'restoration_gain_positive_fraction': sum(v > 0 for v in values) / len(values),
                          'restoration_gain_stratified_subject_bootstrap_95_interval':
                              interval(values, relations, seed=17 + layer + 100 * SITES.index(site)),
                          'first_token_readout_eligible': len(valid),
                          'first_token_margin_mean': statistics.mean(valid) if valid else None})
    selected = max(cells, key=lambda row: row['restoration_gain_mean_nats'])
    result = {'scope': 'Exploratory English-only E4 on preassigned, personally reviewed development facts',
              'n_reviewed': len(reviewed), 'n_screened_eligible': len(eligible),
              'by_relation': dict(Counter(relation_by_fact.values())), 'raw_items': len(rows),
              'sites': SITES, 'layers': LAYERS, 'cells': cells,
              'descriptive_peak': {'site': selected['site'], 'layer': selected['layer'],
                                   'mean_gain_nats': selected['restoration_gain_mean_nats']},
              'limits': [
                  'The descriptive peak was selected from 24 tested site/layer cells and is optimistically biased.',
                  'The intervals describe each fixed cell separately; they do not adjust for peak selection or 24-way multiplicity.',
                  'Earlier E5 held-out outcomes existed before this E4 sweep; E4 cannot retroactively prespecify E5 layers.',
                  'Only English and development subjects were tested in E4. This does not establish Hebrew or Arabic localization.',
                  'All subjects were originally selected from the mLAMA candidate archive, albeit checked against separate source pages.'
              ]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'n': len(eligible), 'descriptive_peak': result['descriptive_peak']}))


if __name__ == '__main__':
    main()

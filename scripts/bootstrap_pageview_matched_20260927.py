"""Pair-level, relation-stratified uncertainty for the retrospective match."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/fragmentation-pageviews-2023-match-heldout-exploratory-20260927.json'
OUT = ROOT / 'results/fragmentation-pageviews-2023-match-intervals-20260927.json'
METRICS = ('high_minus_low_u_accuracy', 'high_minus_low_u_gold_loglik',
           'high_minus_low_d_minus_u_accuracy')


def main():
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    report = {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'method': '10,000 relation-stratified bootstrap resamples of distinct matched pairs; descriptive percentile intervals',
              'scope': 'retrospective exploratory; unadjusted for three outcomes and two languages',
              'languages': {}}
    for language in ('he', 'ar'):
        groups = {g['relation']: g['pairs'] for g in source['groups']
                  if g['language'] == language and g['pairs']}
        pairs = [p for group in groups.values() for p in group]
        if len(pairs) != source['summary'][language]['matched_pairs']:
            raise ValueError('Match count changed')
        rng = random.Random(1729 + (language == 'ar'))
        result = {'n_pairs': len(pairs), 'by_relation': {k: len(v) for k, v in groups.items()},
                  'outcomes': {}}
        for metric in METRICS:
            trials = []
            for _ in range(10000):
                sampled = [rng.choice(group)[metric] for group in groups.values() for _ in group]
                trials.append(statistics.mean(sampled))
            trials.sort()
            result['outcomes'][metric] = {
                'mean': statistics.mean(p[metric] for p in pairs),
                'bootstrap_percentile_95_interval': [trials[249], trials[9749]],
            }
        report['languages'][language] = result
    report['warning'] = ('The sample is small and the matching was chosen after Aya outcomes existed. '
                         'These intervals do not quantify selection uncertainty, confounding, or Aya training exposure.')
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'languages': report['languages']}, indent=2))


if __name__ == '__main__':
    main()

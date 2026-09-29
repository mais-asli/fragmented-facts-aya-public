"""Descriptive paired site contrasts at the prior E5 lower-middle anchor layer."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.analysis import cluster_effect  # noqa: E402

BASE = ROOT / 'results/e4c27/results'
OUT = ROOT / 'results/e4-anchor-site-specificity-20260927.json'
LAYER = 12


def main():
    output = {'scope': 'Exploratory post-E5 English E4 anchor-site paired descriptives',
              'anchor_layer': LAYER,
              'anchor_reason': 'Layer 12 is in the previously E5-selected lower-middle window; E4 was run later',
              'cohorts': {},
              'limits': ['Site contrasts were summarized after E4 output; no confirmatory p-values.',
                         'Pilot and held-out core subjects differ but come from the same mLAMA archive.',
                         'Only the English prompt is tested here.']}
    for cohort, expected in (('pilot', 17), ('test', 16)):
        paths = sorted(BASE.glob(f'e4-reviewed-{cohort}-20260927-shard*/*/items/*.json'))
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
        selected = [r for r in rows if r['key']['layer'] == LAYER]
        index = {(r['key']['fact_id'], r['key']['site']): r for r in selected}
        facts = sorted({r['key']['fact_id'] for r in selected})
        if len(facts) != expected or len(index) != expected * 3:
            raise ValueError('Incomplete anchor-site grid')
        relation_counts = Counter(index[(fid, 'last_subject')]['fact']['relation'] for fid in facts)
        comparisons = {}
        for other in ('first_subject', 'prediction'):
            effects = [index[(fid, 'last_subject')]['restoration_gain'] -
                       index[(fid, other)]['restoration_gain'] for fid in facts]
            subjects = [index[(fid, other)]['fact']['subject_qid'] for fid in facts]
            relations = [index[(fid, other)]['fact']['relation'] for fid in facts]
            comparisons['last_subject_minus_' + other] = {
                **cluster_effect(effects, subjects, relations, 10000, 17),
                'positive_subjects': sum(v > 0 for v in effects),
                'relation_means': {r: sum(v for v, rel in zip(effects, relations) if rel == r) / n
                                   for r, n in relation_counts.items()},
            }
        output['cohorts'][cohort] = {'core_subjects': expected,
                                     'by_relation': dict(relation_counts),
                                     'comparisons': comparisons}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'cohorts': {
        cohort: {name: {'estimate': result['estimate'], 'ci95': result['ci95']}
                 for name, result in info['comparisons'].items()}
        for cohort, info in output['cohorts'].items()}}, indent=2))


if __name__ == '__main__':
    main()

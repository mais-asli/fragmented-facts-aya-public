"""Analyze the retrospective Aya English E4 sweep without selecting a peak test."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/e4c27/results'
PROTOCOL = ROOT / 'configs/e4_reviewed_cohorts_protocol_20260927.json'
OUT = ROOT / 'results/e4-reviewed-cohorts-analysis-20260927.json'
FACTS = {
    'pilot': ROOT / 'data/curated/pilot-screened-20260926.jsonl',
    'test': ROOT / 'data/curated/aya-independent-test-v1-screened.jsonl',
}


def interval(values, relations, seed):
    groups = defaultdict(list)
    for value, relation in zip(values, relations):
        groups[relation].append(value)
    rng = random.Random(seed)
    samples = []
    for _ in range(10000):
        draw = [rng.choice(group) for group in groups.values() for _ in group]
        samples.append(statistics.mean(draw))
    samples.sort()
    return [samples[249], samples[9749]]


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    output = {'scope': protocol['scope'], 'timing': protocol['timing'],
              'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'sites': protocol['sites'], 'layers': protocol['layers'], 'cohorts': {},
              'limits': [
                  'E4 data were collected after earlier E5 outcomes and after the separate E4 development gate failed.',
                  'The two cohorts are subject-disjoint but from the same mLAMA candidate archive.',
                  'English readout/restoration is not direct Hebrew or Arabic localization evidence.',
                  'Each bootstrap interval is for a fixed cell; 24 cells per cohort are shown without peak-selection or multiplicity adjustment.',
                  'The first-token readout may be ineligible if gold and distractor first tokens collide; full-answer restoration remains separate.',
              ]}
    for cohort, file in FACTS.items():
        facts = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
        core = {f['fact_id']: f for f in facts if f['core_member'] and f['english_eligible']}
        if len(core) != protocol['cohorts'][cohort]['core_count']:
            raise ValueError('Core count differs from E4 protocol')
        paths = sorted(BASE.glob(f'e4-reviewed-{cohort}-20260927-shard*/*/items/*.json'))
        rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
        completions = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(
            BASE.glob(f'e4-reviewed-{cohort}-20260927-shard*/*/completion.json'))]
        expected = {(fid, layer, site) for fid in core
                    for layer in protocol['layers'] for site in protocol['sites']}
        index = {(r['key']['fact_id'], r['key']['layer'], r['key']['site']): r for r in rows}
        if (len(completions) != 2 or len(rows) != len(expected) or set(index) != expected
                or any(c['experiment'] != 'E4' or c['split'] != cohort or
                       c['total_items'] != c['expected_count'] or c['status'] != 'complete'
                       for c in completions)):
            raise ValueError('Incomplete E4 result grid for ' + cohort)
        cells = []
        for site in protocol['sites']:
            for layer in protocol['layers']:
                ordered = sorted(core)
                selected = [index[(fid, layer, site)] for fid in ordered]
                gains = [r['restoration_gain'] for r in selected]
                relations = [core[fid]['relation'] for fid in ordered]
                eligible_readouts = [r['readout']['first_token_margin'] for r in selected
                                     if r['readout']['eligible']]
                cells.append({'site': site, 'layer': layer,
                              'n_subjects': len(selected),
                              'gain_mean_nats': statistics.mean(gains),
                              'gain_median_nats': statistics.median(gains),
                              'gain_positive_fraction': sum(g > 0 for g in gains) / len(gains),
                              'gain_subject_bootstrap_95_interval': interval(
                                  gains, relations, 17 + layer + 100 * protocol['sites'].index(site) +
                                  (1000 if cohort == 'test' else 0)),
                              'readout_eligible_n': len(eligible_readouts),
                              'readout_first_token_margin_mean':
                                  statistics.mean(eligible_readouts) if eligible_readouts else None})
        output['cohorts'][cohort] = {
            'reviewed_n': len(facts), 'core_n': len(core),
            'core_by_relation': dict(Counter(f['relation'] for f in core.values())),
            'records': len(rows), 'cells': cells,
        }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'cohorts': {
        k: {'core_n': v['core_n'], 'records': v['records']} for k, v in output['cohorts'].items()}}))


if __name__ == '__main__':
    main()

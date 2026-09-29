"""Cross-check stratified human-audit weights against all 480 held-out Aya outputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'results/heldout-human-output-audit-20260927.json'
BASE = ROOT / 'results/heldout/results'
OUT = ROOT / 'results/heldout-human-output-weighting-check-20260927.json'


def main():
    if OUT.exists():
        raise FileExistsError('Held-out audit weighting check already exists')
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    population = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(
        BASE.glob('e2-aya-independent-test-v1-shard*/*/items/*.json'))]
    if len(population) != 480 or audit['sample_n'] != 80:
        raise ValueError('Held-out E2 population or reviewed sample incomplete')
    groups = {}
    for language in ('he', 'ar'):
        for variant in ('U', 'D'):
            group = f'{language}/{variant}'
            all_rows = [r for r in population if r['key']['language'] == language and
                        r['key']['variant'] == variant]
            sample = [r for r in audit['rows'] if r['language'] == language and
                      r['variant'] == variant]
            if len(all_rows) != 120 or len(sample) != 20:
                raise ValueError(group + ' has incomplete population or sample')
            total_weight = sum(r['weight'] for r in sample)
            if abs(total_weight - 120) > 1e-9:
                raise ValueError(group + ' inverse sampling weights do not cover population')
            full_machine = sum(int(r['evaluation']['entity_correct']) for r in all_rows) / 120
            weighted_machine = sum(r['weight'] * int(r['machine_entity_correct']) for r in sample) / 120
            weighted_human = sum(r['weight'] * int(r['human_entity_correct']) for r in sample) / 120
            if abs(full_machine - weighted_machine) > 1e-12:
                raise ValueError(group + ' machine prevalence not reproduced by frozen weights')
            if abs(weighted_human - audit['groups'][group]['human_entity_correct_weighted']) > 1e-12:
                raise ValueError(group + ' human weighted result differs from audit report')
            groups[group] = {
                'population_outputs': 120, 'sample_outputs': 20,
                'weight_sum': total_weight,
                'full_population_strict_machine_correct': full_machine,
                'sample_weighted_strict_machine_correct': weighted_machine,
                'sample_weighted_human_entity_correct': weighted_human,
                'weighted_human_minus_strict_machine': weighted_human - full_machine,
            }
    OUT.write_text(json.dumps({
        'status': 'All four weighted machine rates exactly reproduce the full E2 population',
        'scope': 'Descriptive sample weighting check; human-vs-strict differences are not a revised primary score',
        'groups': groups,
    }, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({g: x['weighted_human_minus_strict_machine']
                      for g, x in groups.items()}, indent=2))


if __name__ == '__main__':
    main()

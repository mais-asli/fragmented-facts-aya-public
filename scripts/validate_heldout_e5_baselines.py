"""Check held-out E5 U/D baselines against the older frozen E2 t1 results."""

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / 'results/heldout/results'
E5 = ROOT / 'results/h27/results'
OUT = ROOT / 'results/heldout-e5-baseline-consistency-20260927.json'


def load(paths, experiment, variants):
    index = {}
    for path in paths:
        row = json.loads(path.read_text(encoding='utf-8'))
        key = row['key']
        if key['experiment'] != experiment or key['template'] != 't1':
            continue
        if key['variant'] not in variants:
            continue
        tuple_key = (key['fact_id'], key['language'], variants[key['variant']])
        if tuple_key in index:
            raise ValueError('Duplicate baseline record')
        index[tuple_key] = row
    return index


def main():
    e2 = load(E2.glob('e2-aya-independent-test-v1-shard*/*/items/*.json'),
              'E2', {'U': 'U', 'D': 'D'})
    e5 = load(E5.glob('e5-aya-heldout-strengthening-20260927-shard*/*/items/*.json'),
              'E5', {'baseline_U': 'U', 'baseline_D': 'D'})
    if set(e2) != set(e5) or len(e2) != 160:
        raise ValueError(f'E2/E5 baseline coverage mismatch: E2={len(e2)}, E5={len(e5)}')
    deltas = []
    prompt_differences = []
    generation_differences = []
    evaluation_differences = []
    answer_differences = []
    for key in sorted(e2):
        a, b = e2[key], e5[key]
        if a['prompt']['text'] != b['prompt']['text']:
            prompt_differences.append(key)
        if a['gold_score']['answer'] != b['gold_score']['answer']:
            answer_differences.append(key)
        deltas.append(abs(a['gold_score']['sum_logprob'] - b['gold_score']['sum_logprob']))
        if a['generation']['text'] != b['generation']['text']:
            generation_differences.append(key)
        if a['evaluation']['entity_correct'] != b['evaluation']['entity_correct']:
            evaluation_differences.append(key)
    report = {
        'scope': 'Independent repeated E5 U/D baselines versus earlier held-out E2 t1 on the same Aya snapshot',
        'pairs': len(e2),
        'prompt_differences': prompt_differences,
        'answer_differences': answer_differences,
        'max_abs_gold_log_likelihood_difference': max(deltas),
        'median_abs_gold_log_likelihood_difference': statistics.median(deltas),
        'gold_log_likelihood_differences_over_1e-3': sum(x > 1e-3 for x in deltas),
        'generation_differences': generation_differences,
        'entity_evaluation_differences': evaluation_differences,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pairs': len(e2), 'max_abs_score_delta': max(deltas),
                      'generation_differences': len(generation_differences),
                      'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()

"""Give an auditable relation-level breakdown of the separate Google-RE run."""

from collections import defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/gre27/results'
FORMAT = ROOT / 'results/answer-format-sensitivity-google-re-external-20260927.json'
GRID = ROOT / 'results/google-re-external-aya-grid-audit-20260927.json'
OUT = ROOT / 'results/google-re-external-relation-breakdown-20260927.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def key_tuple(key):
    return (key['experiment'], key['fact_id'], key['language'],
            key['template'], key['variant'])


def main():
    if OUT.exists():
        raise FileExistsError('External relation breakdown already exists')
    grid = read(GRID)
    fmt = read(FORMAT)
    if grid['status'] != 'complete_and_identity_verified' or fmt['records'] != 546:
        raise ValueError('External grid or format-sensitivity analysis incomplete')
    rescued = {key_tuple(r['key']) for r in fmt['rescue_candidates_for_human_check']
               if r['rule'] == 'arabic_prefix_exact'}
    rows = [read(p) for p in sorted(BASE.glob('*aya-google-re-external-20260927-shard*/*/items/*.json'))]
    if len(rows) != 546:
        raise ValueError('Expected all 546 E1/E2 output cells')
    by_key = {key_tuple(r['key']): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError('Duplicate external output key')
    summary = {}
    for relation in ('P19', 'P20'):
        for language in ('en', 'he', 'ar'):
            baseline = [r for r in rows if r['key']['experiment'] == 'E1' and
                        r['fact']['relation'] == relation and r['key']['language'] == language]
            subjects = {r['fact']['subject_qid'] for r in baseline}
            if len(baseline) != 3 * len(subjects) or len(subjects) != (14 if relation == 'P19' else 12):
                raise ValueError('Incomplete relation-level E1 baseline')
            entry = {
                'subjects': len(subjects), 'e1_outputs': len(baseline),
                'e1_strict_correct': sum(int(r['evaluation']['entity_correct']) for r in baseline),
                'e1_arabic_exact_prefix_rescues': sum(key_tuple(r['key']) in rescued for r in baseline),
            }
            if language != 'en':
                pairs = []
                for fact_id in {r['key']['fact_id'] for r in baseline}:
                    for template in ('t1', 't2', 't3'):
                        u = by_key[('E2', fact_id, language, template, 'U')]
                        d = by_key[('E2', fact_id, language, template, 'D')]
                        pairs.append((u, d))
                if len(pairs) != len(baseline):
                    raise ValueError('Incomplete relation-level E2 pairs')
                entry.update({
                    'e2_pairs': len(pairs),
                    'e2_u_strict_correct': sum(int(u['evaluation']['entity_correct']) for u, _ in pairs),
                    'e2_d_strict_correct': sum(int(d['evaluation']['entity_correct']) for _, d in pairs),
                    'e2_u_arabic_exact_prefix_rescues': sum(key_tuple(u['key']) in rescued for u, _ in pairs),
                    'e2_d_arabic_exact_prefix_rescues': sum(key_tuple(d['key']) in rescued for _, d in pairs),
                    'e2_mean_d_minus_u_gold_loglik_nats': statistics.mean(
                        d['gold_score']['sum_logprob'] - u['gold_score']['sum_logprob'] for u, d in pairs),
                })
            summary[f'{relation}/{language}'] = entry
    OUT.write_text(json.dumps({
        'scope': 'Descriptive relation breakdown; repeated templates are not independent subjects',
        'primary': 'Frozen strict complete-answer scoring',
        'sensitivity': 'Exact Arabic answer-prefix removal only, post-hoc for this external cohort',
        'cells': summary,
    }, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

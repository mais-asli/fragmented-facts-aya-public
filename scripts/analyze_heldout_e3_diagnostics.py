"""Report all eligible same-text E3 contrasts, including Hebrew inside-only."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.analysis import cluster_effect, paired_rows, sign_flip_p  # noqa: E402

BASE = ROOT / 'results/h27'
ELIGIBILITY = ROOT / 'results/heldout-e3-tokenizer-eligibility-20260927.json'
OUT = ROOT / 'results/heldout-e3-all-contrasts-20260927.json'


def estimate(pairs):
    if not pairs:
        return None
    deltas = [b['gold_score']['sum_logprob'] - a['gold_score']['sum_logprob'] for a, b in pairs]
    subjects = [b['fact']['subject_qid'] for _, b in pairs]
    relations = [b['fact']['relation'] for _, b in pairs]
    result = cluster_effect(deltas, subjects, relations, 10000, 17)
    result['paired_sign_flip_p_unadjusted_exploratory'] = sign_flip_p(deltas, subjects, relations, 10000, 17)
    result['n_distinct_facts'] = len({b['key']['fact_id'] for _, b in pairs})
    result['relation_subject_counts'] = dict(Counter(b['fact']['relation'] for _, b in pairs))
    return result


def main():
    eligible = json.loads(ELIGIBILITY.read_text(encoding='utf-8'))
    rows = [json.loads(p.read_text(encoding='utf-8'))
            for p in (BASE / 'results').glob('e3-aya-heldout-strengthening-20260927-shard*/*/items/*.json')]
    if not rows or {r['key']['experiment'] for r in rows} != {'E3'}:
        raise ValueError('Complete new-subject Aya E3 records not found')
    report = {'scope': 'All frozen eligible same-rendered-text E3 contrasts; exploratory 40-subject replication',
              'tokenizer_eligibility_path': str(ELIGIBILITY.relative_to(ROOT)),
              'n_records': len(rows), 'languages': [],
              'limits': ['The full rendered chat prompt is byte-identical across split variants, but token identities and downstream positions differ.',
                         'Hebrew has no eligible outside-subject split under the frozen rule; inside-versus-canonical is not a matched-length location control.',
                         'Sign-flip p-values are exploratory and unadjusted; no pure token-count causal claim follows.']}
    for language in ('en', 'he', 'ar'):
        inside, missing_inside = paired_rows(rows, 'E3', 'canonical', 'subject_split', language)
        outside, missing_outside = paired_rows(rows, 'E3', 'canonical', 'outside_split', language)
        matched, missing_matched = paired_rows(rows, 'E3', 'outside_split', 'subject_split', language)
        # If one arm is ineligible, paired_rows reports it as missing. The
        # tokenizer-only eligibility file fixes the expected counts in advance.
        expected = [r for r in eligible['rows'] if r['language'] == language]
        expected_inside = sum(r['inside_eligible'] for r in expected)
        expected_outside = sum(r['outside_eligible'] for r in expected)
        expected_matched = sum(r['matched_eligible'] for r in expected)
        if (len(inside), len(outside), len(matched)) != (expected_inside, expected_outside, expected_matched):
            raise ValueError(f'E3 coverage differs from tokenizer-only freeze for {language}')
        report['languages'].append({
            'language': language, 'facts': len(expected),
            'inside_eligible': expected_inside, 'outside_eligible': expected_outside,
            'matched_eligible': expected_matched,
            'inside_minus_canonical_gold_loglik': estimate(inside),
            'outside_minus_canonical_gold_loglik': estimate(outside),
            'inside_minus_outside_gold_loglik': estimate(matched),
            'missing_inside_pair_keys': missing_inside,
            'missing_outside_pair_keys': missing_outside,
            'missing_matched_pair_keys': missing_matched,
        })
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'records': len(rows),
                      'matched': {r['language']: r['matched_eligible'] for r in report['languages']}}, indent=2))


if __name__ == '__main__':
    main()

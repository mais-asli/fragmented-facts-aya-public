"""Independently analyze the versioned Aya byte-level E3 run on 40 facts."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.analysis import cluster_effect, holm, sign_flip_p  # noqa: E402

BASE = ROOT / 'results/b27/results'
OLD = ROOT / 'results/h27/results'
OUT = ROOT / 'results/e3-byte-level-40-subject-analysis-20260927.json'


def effect(pairs):
    values = [b['gold_score']['sum_logprob'] - a['gold_score']['sum_logprob'] for a, b in pairs]
    subjects = [b['fact']['subject_qid'] for _, b in pairs]
    relations = [b['fact']['relation'] for _, b in pairs]
    result = cluster_effect(values, subjects, relations, 10000, 17)
    result['paired_sign_flip_p_unadjusted'] = sign_flip_p(values, subjects, relations, 10000, 17)
    return result


def main():
    paths = sorted(BASE.glob('e3-byte-level-20260927-shard*/*/items/*.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if len(rows) != 240 or {r['key']['experiment'] for r in rows} != {'E3_byte_level'}:
        raise ValueError('Expected exactly 240 actual byte-level E3 records')
    completions = [json.loads(path.read_text(encoding='utf-8'))
                   for path in sorted(BASE.glob('e3-byte-level-20260927-shard*/*/completion.json'))]
    if len(completions) != 2 or sum(c['total_items'] for c in completions) != 240 or any(
            c['status'] != 'complete' or c['total_items'] != c['expected_count'] for c in completions):
        raise ValueError('Incomplete raw E3 shards')
    old_paths = sorted(OLD.glob('e3-aya-heldout-strengthening-20260927-shard*/*/items/*.json'))
    old_rows = [json.loads(path.read_text(encoding='utf-8')) for path in old_paths]
    old_canonical = {(r['key']['fact_id'], r['key']['language']): r
                     for r in old_rows if r['key']['variant'] == 'canonical'
                     and r['key']['language'] in ('he', 'ar')}
    report = {'scope': 'Post-E3 method extension on the same 40 test subjects, exploratory despite frozen pre-run inputs',
              'data_kind': 'research', 'conditions': ['canonical', 'subject_split', 'outside_split'],
              'records': len(rows), 'languages': [],
              'limits': ['The rendered chat prompt and answer are identical within each fact; token IDs and positions still differ.',
                         'The broader byte-level rule was selected after the original E3 coverage and outcomes were known; this is not an independent confirmatory test.',
                         'This test changes segmentation location with one extra token in both arms; it cannot isolate token count itself.']}
    for language in ('he', 'ar'):
        selected = [r for r in rows if r['key']['language'] == language]
        index = {(r['key']['fact_id'], r['key']['variant']): r for r in selected}
        facts = sorted({r['key']['fact_id'] for r in selected})
        if len(facts) != 40 or len(index) != 120:
            raise ValueError('Duplicate or missing E3 fact/variant keys')
        contrasts = {}
        for name, a_name, b_name in (
            ('inside_minus_outside', 'outside_split', 'subject_split'),
            ('inside_minus_canonical', 'canonical', 'subject_split'),
            ('outside_minus_canonical', 'canonical', 'outside_split'),
        ):
            pairs = [(index[(fact, a_name)], index[(fact, b_name)]) for fact in facts]
            if any(a['prompt']['text'] != b['prompt']['text'] or
                   a['gold_score']['answer'] != b['gold_score']['answer'] or
                   len(a['prompt']['input_ids']) + (a_name == 'canonical')
                   != len(b['prompt']['input_ids']) + (b_name == 'canonical')
                   for a, b in pairs):
                raise ValueError('Same-string or matched-length invariant failed')
            contrasts[name] = effect(pairs)
        for fact in facts:
            new = index[(fact, 'canonical')]
            old = old_canonical.get((fact, language))
            if (old is None or new['prompt']['input_ids'] != old['prompt']['input_ids'] or
                    new['gold_score']['answer_ids'] != old['gold_score']['answer_ids'] or
                    abs(new['gold_score']['sum_logprob'] - old['gold_score']['sum_logprob']) > 1e-9):
                raise ValueError('Repeated E3 canonical baseline differs from the previous held-out run')
        report['languages'].append({'language': language, 'facts': len(facts),
                                    'relations': dict(Counter(index[(fact, 'canonical')]['fact']['relation']
                                                              for fact in facts)),
                                    'canonical_baselines_exactly_reproduced': len(facts),
                                    **contrasts})
    adjusted = holm([row['inside_minus_outside']['paired_sign_flip_p_unadjusted']
                     for row in report['languages']])
    for row, p in zip(report['languages'], adjusted):
        row['inside_minus_outside_holm_p_two_languages'] = p
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'effects': [
        {'language': row['language'],
         'estimate': row['inside_minus_outside']['estimate'],
         'holm_p': row['inside_minus_outside_holm_p_two_languages']}
        for row in report['languages']]}, indent=2))


if __name__ == '__main__':
    main()

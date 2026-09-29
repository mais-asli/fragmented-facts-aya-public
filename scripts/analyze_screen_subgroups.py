"""Analyze predeclared English-screen subgroups from saved actual Aya outputs."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.analysis import cluster_effect, holm, paired_rows, sign_flip_p

BASE = ROOT / 'results/tau_research_20260926'
SCREENED = BASE / 'data/curated/pilot-screened-20260926.jsonl'
ANALYSIS = BASE / 'results/analysis-reviewed-pilot-20260926.json'
OUT = ROOT / 'results/english-screen-subgroups-20260926.json'


def estimate(values, records, paired=True):
    subjects = [r['fact']['subject_qid'] for r in records]
    relations = [r['fact']['relation'] for r in records]
    result = cluster_effect(values, subjects, relations, repeats=10000, seed=17)
    if paired:
        result['paired_sign_flip_p'] = sign_flip_p(values, subjects, relations, repeats=10000, seed=17)
    return result


def main():
    report = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    if report['data_kind'] != 'research':
        raise ValueError('Actual Aya outputs required')
    facts = [json.loads(line) for line in SCREENED.read_text(encoding='utf-8').splitlines()]
    if len(facts) != 57:
        raise ValueError('Expected the fixed reviewed pilot cohort')
    runs = []
    for relative in report['run_paths']:
        for path in (BASE / relative / 'items').glob('*.json'):
            runs.append(json.loads(path.read_text(encoding='utf-8')))
    result = {'scope': 'Exploratory reviewed-pilot subgroups selected by two separate English prompts',
              'cohort_facts': 57, 'subgroups': []}
    for name, field in (('strict_two_of_two', 'english_eligible'),
                        ('one_of_two', 'english_one_of_two')):
        included = {fact['fact_id'] for fact in facts if fact[field]}
        subset = [row for row in runs if row['fact']['fact_id'] in included]
        group = {'name': name, 'subjects': len(included), 'baseline': [], 'orthography': []}
        for lang in ('en', 'he', 'ar'):
            records = [row for row in subset if row['key']['experiment'] == 'E1'
                       and row['key']['language'] == lang]
            if records:
                group['baseline'].append({'language': lang, 'entity_correct': estimate(
                    [int(row['evaluation']['entity_correct']) for row in records], records,
                    paired=False),
                    'all_alias_entity_correct': estimate(
                    [int(row['evaluation_all_aliases']['entity_correct']) for row in records], records,
                    paired=False)})
        for lang in ('he', 'ar'):
            pairs, missing = paired_rows(subset, 'E2', 'U', 'D', lang)
            if not pairs:
                continue
            records = [d for u, d in pairs]
            entry = {'language': lang, 'n_prompt_pairs': len(pairs), 'missing_pairs': missing,
                     'accuracy': estimate([int(d['evaluation']['entity_correct']) -
                                           int(u['evaluation']['entity_correct']) for u, d in pairs], records),
                     'all_alias_accuracy': estimate([int(d['evaluation_all_aliases']['entity_correct']) -
                                                     int(u['evaluation_all_aliases']['entity_correct'])
                                                     for u, d in pairs], records),
                     'log_likelihood': estimate([d['gold_score']['sum_logprob'] -
                                                 u['gold_score']['sum_logprob'] for u, d in pairs], records)}
            group['orthography'].append(entry)
        for metric in ('accuracy', 'log_likelihood'):
            corrected = holm([entry[metric]['paired_sign_flip_p']
                              for entry in group['orthography']])
            for entry, pvalue in zip(group['orthography'], corrected):
                entry[metric]['exploratory_holm_p'] = pvalue
        result['subgroups'].append(group)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT),
                      'strict_subjects': result['subgroups'][0]['subjects'],
                      'one_of_two_subjects': result['subgroups'][1]['subjects']}, indent=2))


if __name__ == '__main__':
    main()

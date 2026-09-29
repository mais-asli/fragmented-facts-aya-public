"""Post-hoc sensitivity excluding subjects that visibly contain an answer label.

The frozen pilot/test cohorts and their primary results remain unchanged.
"""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.analysis import cluster_effect, paired_rows  # noqa: E402

LEDGER = ROOT / 'output/data_audit_20260927/FROZEN_FACT_LEDGER_97.csv'
OUT = ROOT / 'results/answer-cue-exclusion-sensitivity-20260927.json'
COHORTS = {
    'pilot': (ROOT / 'results/tau_research_20260926',
              'results/analysis-reviewed-pilot-20260926.json', 57),
    'test': (ROOT / 'results/heldout',
             'results/analysis-aya-independent-test-v1.json', 40),
}


def effect(pairs, field):
    delta = [(d['gold_score']['sum_logprob'] - u['gold_score']['sum_logprob'])
             if field == 'gold_sum_loglik'
             else (int(d['evaluation']['entity_correct']) - int(u['evaluation']['entity_correct']))
             for u, d in pairs]
    return cluster_effect(delta,
                          [d['fact']['subject_qid'] for _, d in pairs],
                          [d['fact']['relation'] for _, d in pairs],
                          10000, 17)


def main():
    with LEDGER.open(encoding='utf-8-sig', newline='') as source:
        ledger = list(csv.DictReader(source))
    if len(ledger) != 97:
        raise ValueError('Expected 97 frozen fact ledger rows')
    flagged = {split: {r['fact_id'] for r in ledger if r['split'] == split
                       and any(r[f'answer_text_in_subject_{language}'] == 'True'
                               for language in ('en', 'he', 'ar'))}
               for split in COHORTS}
    output = {
        'scope': 'Post-outcome, descriptive answer-cue exclusion; do not replace the frozen primary cohort',
        'cue_rule': 'Exclude a whole fact if any recorded English/Hebrew/Arabic subject spelling contains its answer label',
        'cohorts': [],
    }
    for split, (base, analysis_path, facts) in COHORTS.items():
        analysis = json.loads((base / analysis_path).read_text(encoding='utf-8'))
        if analysis['data_kind'] != 'research' or analysis['model']['model_id'] != 'CohereLabs/aya-23-8B':
            raise ValueError('Expected real Aya research analysis')
        rows = []
        for relative in analysis['run_paths']:
            for path in (base / relative / 'items').glob('*.json'):
                row = json.loads(path.read_text(encoding='utf-8'))
                if row['key']['experiment'] == 'E2':
                    rows.append(row)
        cohort = {'split': split, 'original_facts': facts,
                  'excluded_fact_ids': sorted(flagged[split]), 'languages': []}
        for language in ('he', 'ar'):
            pairs, missing = paired_rows(rows, 'E2', 'U', 'D', language)
            if missing or len(pairs) != 3 * facts:
                raise ValueError(f'Incomplete original E2 pairs: {split}/{language}')
            reduced = [(u, d) for u, d in pairs if d['key']['fact_id'] not in flagged[split]]
            if len(reduced) != 3 * (facts - len(flagged[split])):
                raise ValueError(f'Cue exclusion coverage mismatch: {split}/{language}')
            cohort['languages'].append({
                'language': language,
                'original': {'facts': facts, 'prompt_pairs': len(pairs),
                             'accuracy_D_minus_U': effect(pairs, 'accuracy'),
                             'gold_sum_loglik_D_minus_U': effect(pairs, 'gold_sum_loglik')},
                'without_answer_cues': {
                    'facts': facts - len(flagged[split]), 'prompt_pairs': len(reduced),
                    'accuracy_D_minus_U': effect(reduced, 'accuracy'),
                    'gold_sum_loglik_D_minus_U': effect(reduced, 'gold_sum_loglik')},
            })
            saved = [r for r in analysis['orthography']
                     if r['language'] == language and r['split'] == split]
            original = cohort['languages'][-1]['original']
            if (len(saved) != 1 or
                    abs(original['accuracy_D_minus_U']['estimate'] - saved[0]['accuracy']['estimate']) > 1e-10 or
                    abs(original['gold_sum_loglik_D_minus_U']['estimate'] - saved[0]['log_likelihood']['estimate']) > 1e-10):
                raise ValueError(f'Original primary effect did not reproduce: {split}/{language}')
        output['cohorts'].append(cohort)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT),
                      'excluded': {x['split']: len(x['excluded_fact_ids']) for x in output['cohorts']}}, indent=2))


if __name__ == '__main__':
    main()

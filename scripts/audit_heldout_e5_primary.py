"""Audit the pilot-selected primary held-out E5 site and its controls."""

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.analysis import holm  # noqa: E402

ANALYSIS = ROOT / 'results/h27/results/analysis-e5-aya-heldout-strengthening-20260927.json'
CONFIG = ROOT / 'configs/mechanism_independent_test_pilot_selected_20260927.json'
OUT = ROOT / 'results/heldout-e5-primary-audit-20260927.json'


def one(records, language, window, condition):
    selected = [r for r in records if r['language'] == language and
                r['window'] == window and r['condition'] == condition and
                r['component'] == 'residual']
    if len(selected) != 1 or selected[0]['missing_pairs']:
        raise ValueError(f'Missing/duplicate controlled E5 effect: {language}/{window}/{condition}')
    return selected[0]


def verify_raw_controls(language, window, controls):
    paths = sorted((ROOT / 'results/h27/results').glob(
        'e5-aya-heldout-strengthening-20260927-shard*/*/items/*.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if len(rows) != 2160 or {r['key']['experiment'] for r in rows} != {'E5'}:
        raise ValueError('Incomplete raw held-out E5 records')
    verified = []
    for control in controls:
        condition = control['condition']
        baseline_name = 'baseline_U' if condition == 'reverse' else 'baseline_D'
        baselines = {r['key']['fact_id']: r for r in rows
                     if r['key']['language'] == language and r['key']['variant'] == baseline_name}
        patched = {r['key']['fact_id']: r for r in rows
                   if r['key']['language'] == language and r['key']['variant'] == condition
                   and r['key'].get('window') == window and r['key'].get('component') == 'residual'}
        if len(baselines) != 40 or len(patched) != 40 or set(baselines) != set(patched):
            raise ValueError(f'Incomplete raw E5 control: {language}/{condition}')
        by_relation = {}
        accuracy_by_relation = {}
        for fact_id, answer in patched.items():
            baseline = baselines[fact_id]
            if (answer['gold_score']['answer'] != baseline['gold_score']['answer'] or
                    answer['fact']['relation'] != baseline['fact']['relation']):
                raise ValueError('Raw E5 pair differs in answer or relation')
            relation = answer['fact']['relation']
            by_relation.setdefault(relation, []).append(
                answer['gold_score']['sum_logprob'] - baseline['gold_score']['sum_logprob'])
            accuracy_by_relation.setdefault(relation, []).append(
                int(answer['evaluation']['entity_correct']) - int(baseline['evaluation']['entity_correct']))
        if len(by_relation) != 4:
            raise ValueError('Raw E5 relation coverage differs')
        score = statistics.mean(statistics.mean(v) for v in by_relation.values())
        accuracy = statistics.mean(statistics.mean(v) for v in accuracy_by_relation.values())
        if (abs(score - control['score_gain']['estimate']) > 1e-9 or
                abs(accuracy - control['accuracy_gain']['estimate']) > 1e-9):
            raise ValueError(f'Raw E5 effect differs from TAU analysis: {language}/{condition}')
        verified.append({'condition': condition, 'baseline': baseline_name,
                         'facts': len(patched), 'relation_macro_score_gain': score,
                         'relation_macro_accuracy_gain': accuracy})
    return verified


def main():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    report = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    if report['data_kind'] != 'research' or report['model']['model_id'] != config['model_id']:
        raise ValueError('Pinned Aya research result required')
    primary = config['primary_window']
    if primary != 'lower_middle' or [w['layers'] for w in config['windows'] if w['name'] == primary] != [[12, 13]]:
        raise ValueError('Pilot-selected primary window changed')
    controls = report['patch_controls']
    if len(report['patch_specificity']) != 8 or len(controls) != 48:
        raise ValueError('Expected four windows, two languages and all six controls')
    out = {'scope': 'Pilot-selected Aya E5 replication on subject-disjoint test facts; site fixed before test E5 outcomes, after test E1/E2 behavior was available',
           'primary_window': primary, 'primary_layers': [12, 13],
           'selection_source': 'pilot E5 exploratory result, not independent English E4 localization',
           'languages': [], 'interpretation_limits': [
               'Same-entity versus unpatched D is the direct restoration comparison.',
               'Same-entity versus unrelated donor can be large because unrelated donors harm answers; show both gains.',
               'Whole-residual patching does not identify a unique circuit or isolate token count from token identities and spelling familiarity.',
               'The held-out subjects are new but drawn from the same mLAMA source archive.']}
    for language in ('he', 'ar'):
        same = one(controls, language, primary, 'same_entity')
        unrelated = one(controls, language, primary, 'unrelated')
        # Preserve every control at the primary site and all windows' direct
        # same-entity gains for depth inspection; only the selected site is
        # treated as primary.
        primary_controls = [r for r in controls if r['language'] == language and r['window'] == primary]
        all_window_same = [one(controls, language, w['name'], 'same_entity')
                           for w in config['windows']]
        specificity = [r for r in report['patch_specificity']
                       if r['language'] == language and r['window'] == primary]
        if len(specificity) != 1 or not specificity[0]['primary']:
            raise ValueError('Primary specificity site mismatch')
        out['languages'].append({
            'language': language,
            'same_entity_minus_D': same['score_gain'],
            'same_entity_accuracy_gain': same['accuracy_gain'],
            'same_entity_harmed_initially_correct_items': same['harmed_items'],
            'unrelated_donor_minus_D': unrelated['score_gain'],
            'same_entity_minus_unrelated_donor': specificity[0]['specificity'],
            'primary_site_all_controls': primary_controls,
            'same_entity_gain_by_fixed_window': all_window_same,
            'raw_recalculated_primary_controls': verify_raw_controls(language, primary, primary_controls),
        })
    adjusted = holm([r['same_entity_minus_D']['paired_sign_flip_p'] for r in out['languages']])
    for row, value in zip(out['languages'], adjusted):
        row['same_entity_minus_D_holm_p_two_languages'] = value
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'primary_window': primary,
                      'effects': [{'language': r['language'],
                                   'score_gain': r['same_entity_minus_D']['estimate'],
                                   'holm_p': r['same_entity_minus_D_holm_p_two_languages']}
                                  for r in out['languages']]}, indent=2))


if __name__ == '__main__':
    main()

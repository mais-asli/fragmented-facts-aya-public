"""Reconcile cited audit counts with frozen research artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/strengthening-worklog-count-verification-20260927.json'


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    ledger = read('output/data_audit_20260927/LEDGER_AUDIT.json')
    gap = read('results/proposal_data_gap_audit_20260927.json')
    scoring = read('results/heldout-scoring-audit-20260927.json')
    pilot_audit = read('results/ReviewerA-output-audit-analysis-20260926.json')
    context = read('results/context-mark-control-feasibility-20260927.json')
    e3 = read('results/heldout-e3-tokenizer-eligibility-20260927.json')
    matched = read('results/fragmentation-frequency-match-heldout-exploratory-20260927.json')
    format_sensitivity = read('results/answer-format-sensitivity-heldout-20260927.json')
    by_relation = {(r['language'], r['relation']): r for r in gap['e2_relation_descriptives']}
    checks = {
        'pilot_facts_57': gap['pilot_facts'] == 57,
        'test_facts_40': gap['test_facts'] == 40,
        'ledger_facts_97': ledger['facts'] == 97,
        'all_from_mLAMA': gap['source_datasets'] == {'mLAMA-1.1': 97},
        'no_saved_pageviews': gap['non_null_pageview_facts'] == 0,
        'test_answer_reused_17': ledger['test_answer_qids_seen_in_pilot'] == 17,
        'subject_answer_cues_4': ledger['answer_text_in_subject_any_language'] == 4,
        'hq_tiers_2_3_1': ledger['hq_test_evidence_tiers'] == {'A': 2, 'B': 3, 'C': 1},
        'heldout_e2_480': scoring['e2_rows'] == 480,
        'heldout_alias_disagreements_0': len(scoring['primary_vs_all_aliases_disagreements']) == 0,
        'heldout_cap_cases_4': len(scoring['mechanical_cap_cases']) == 4,
        'pilot_human_80_distinct_43': pilot_audit['reviewed'] == 80 and pilot_audit['distinct_facts_sampled'] == 43,
        'pilot_entity_agreement_79': pilot_audit['entity_agreement_count'] == 79,
        'pilot_conditional_language_38_of_38': pilot_audit['conditional_language_agreement_count'] == 38 and pilot_audit['conditional_language_denominator'] == 38,
        'arabic_p20_strict_floor': by_relation['ar', 'P20']['u_correct'] == 0 and by_relation['ar', 'P20']['d_correct'] == 0 and by_relation['ar', 'P20']['prompt_pairs'] == 39,
        'arabic_p20_prefix_14_10': by_relation['ar', 'P20']['u_prefix_stripped_correct'] == 14 and by_relation['ar', 'P20']['d_prefix_stripped_correct'] == 10,
        'arabic_p19_prefix_0_1': by_relation['ar', 'P19']['u_prefix_stripped_correct'] == 0 and by_relation['ar', 'P19']['d_prefix_stripped_correct'] == 1,
        'e3_arabic_40_matched': sum(v['matched'] for v in e3['counts']['ar'].values()) == 40,
        'e3_hebrew_30_inside_0_outside': sum(v['inside'] for v in e3['counts']['he'].values()) == 30 and sum(v['outside'] for v in e3['counts']['he'].values()) == 0,
        'context_mark_length_matches_2_1': context['counts']['he']['whole_prompt_length_matched'] == 2 and context['counts']['ar']['whole_prompt_length_matched'] == 1,
        'matched_pairs_he16_ar15': sum(g['matched_pairs'] for g in matched['groups'] if g['language'] == 'he') == 16 and sum(g['matched_pairs'] for g in matched['groups'] if g['language'] == 'ar') == 15,
        'matched_arabic_p159_zero': next(g for g in matched['groups'] if g['language'] == 'ar' and g['relation'] == 'P159')['matched_pairs'] == 0,
        'format_sensitivity_840': format_sensitivity['records'] == 840,
    }
    result = {'scope': 'Mechanical audit of counts cited in the Aya strengthening work log',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'all_passed': all(checks.values())}
    OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if not result['all_passed']:
        raise ValueError('Work-log count check failed: ' + ', '.join(k for k, v in checks.items() if not v))
    print(json.dumps({'passed': result['passed'], 'total': result['total'], 'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()

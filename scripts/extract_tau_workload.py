"""Save and audit the actual Aya workload returned from the authenticated TAU job."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.data import alias_registry
from fragmented_facts.scoring import evaluate_answer

WORKFLOW = ROOT / 'results/tau_approved_pilot_workflow.json'
FACTS = ROOT / 'data/curated/pilot-reviewed-20260926.jsonl'
REPORT = ROOT / 'results/workload-ReviewerA-20260926.json'
AUDIT = ROOT / 'results/workload-ReviewerA-20260926.audit.json'


def main():
    workflow = json.loads(WORKFLOW.read_text(encoding='utf-8'))
    if workflow.get('status') != 'workflow_finished' or workflow.get('workflow_exit_code') != 0:
        raise ValueError('TAU workflow has not finished successfully')
    report = workflow['workflow']['model_report']
    if (report['status'] != 'workload_passed' or report['subject_count'] != 50 or
            report['revision'] != '89da1a0ed02d6130f93ae0ffdbedb63b760c0471'):
        raise ValueError('Aya workload identity/count mismatch')
    if len(report['examples']) != 250 or not all(check['passed'] for check in report['checks']):
        raise ValueError('Incomplete workload examples or failed mechanical checks')
    if REPORT.exists() or AUDIT.exists():
        raise FileExistsError('Workload results already extracted; preserve the first record')
    facts = [json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines()]
    by_id = {fact['fact_id']: fact for fact in facts}
    canonical = alias_registry(facts, mode='canonical')
    broad = alias_registry(facts, mode='all')
    rows = []
    groups = defaultdict(list)
    disagreements = []
    for item in report['examples']:
        fact = by_id[item['fact_id']]
        language = item['language']
        answer = item['generation']['text']
        truncated = item['generation']['truncated']
        primary = evaluate_answer(answer, fact['object_qid'], language, canonical, truncated)
        sensitivity = evaluate_answer(answer, fact['object_qid'], language, broad, truncated)
        row = {'fact_id': item['fact_id'], 'subject_qid': fact['subject_qid'],
               'subject_en': fact['subject_labels']['en'], 'relation': fact['relation'],
               'language': language, 'variant': item['variant'],
               'gold_canonical': fact['object_labels'][language],
               'generated': answer, 'truncated': truncated,
               'primary': primary, 'all_aliases': sensitivity,
               'answer_logprob': item['score']['sum_logprob'],
               'generation_seconds': item['generation']['seconds']}
        rows.append(row)
        groups[language + '/' + item['variant']].append(row)
        if primary['entity_correct'] != sensitivity['entity_correct']:
            disagreements.append(row)
    aggregate = {}
    for name, group in groups.items():
        aggregate[name] = {'n': len(group),
                           'canonical_entity_correct': sum(r['primary']['entity_correct'] for r in group),
                           'all_alias_entity_correct': sum(r['all_aliases']['entity_correct'] for r in group),
                           'requested_language_correct': sum(r['primary']['requested_language_correct'] for r in group),
                           'truncated': sum(r['truncated'] for r in group),
                           'categories': dict(Counter(r['primary']['category'] for r in group))}
    audit = {'scope': 'Exploratory longest-prompt workload gate; not E1/E2',
             'human_output_review': 'pending; these are deterministic machine classifications',
             'subject_count': report['subject_count'], 'examples': len(rows),
             'max_input_tokens': report['max_input_tokens'],
             'workload_seconds': report['seconds'],
             'aggregate': aggregate,
             'canonical_vs_all_alias_disagreements': disagreements,
             'rows': rows}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'rows'}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

"""Leave-one-subject-out relation-majority answer baseline for the reviewed cohort."""

from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / 'data/curated/pilot-reviewed-20260926.jsonl'
OUT = ROOT / 'results/reviewed-majority-baseline-20260926.json'


def main():
    facts = [json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines()]
    if len(facts) != 57 or len({f['subject_qid'] for f in facts}) != 57:
        raise ValueError('Unexpected reviewed cohort')
    predictions = []
    for fact in facts:
        other = [f for f in facts if f['relation'] == fact['relation']
                 and f['subject_qid'] != fact['subject_qid']]
        counts = Counter(f['object_qid'] for f in other)
        winner = min(counts, key=lambda qid: (-counts[qid], qid))
        predictions.append({'fact_id': fact['fact_id'], 'relation': fact['relation'],
                            'gold_qid': fact['object_qid'], 'predicted_qid': winner,
                            'correct': winner == fact['object_qid']})
    by_relation = {}
    for relation in sorted({r['relation'] for r in predictions}):
        group = [r for r in predictions if r['relation'] == relation]
        by_relation[relation] = {'n': len(group), 'correct': sum(r['correct'] for r in group),
                                 'accuracy': sum(r['correct'] for r in group) / len(group)}
    report = {'scope': 'Leave-one-subject-out relation-majority baseline on 57 reviewed pilot facts',
              'uses_aya': False, 'n_subjects': 57, 'correct': sum(r['correct'] for r in predictions),
              'micro_accuracy': sum(r['correct'] for r in predictions) / 57,
              'relation_macro_accuracy': sum(r['accuracy'] for r in by_relation.values()) / len(by_relation),
              'by_relation': by_relation, 'predictions': predictions}
    OUT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))


if __name__ == '__main__':
    main()

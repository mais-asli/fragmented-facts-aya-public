"""Exploratory Aya effects by answer reuse and answer cue, preserving E2."""

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.analysis import cluster_effect, paired_rows  # noqa: E402

BASE = ROOT / 'results/heldout'
ANALYSIS = BASE / 'results/analysis-aya-independent-test-v1.json'
LEDGER = ROOT / 'output/data_audit_20260927/FROZEN_FACT_LEDGER_97.csv'
OUT = ROOT / 'results/heldout-provenance-subgroups-exploratory-20260927.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(pairs):
    if not pairs:
        return None
    subjects = [d['fact']['subject_qid'] for _, d in pairs]
    relations = [d['fact']['relation'] for _, d in pairs]
    accuracy = [int(d['evaluation']['entity_correct']) - int(u['evaluation']['entity_correct'])
                for u, d in pairs]
    loglik = [d['gold_score']['sum_logprob'] - u['gold_score']['sum_logprob']
              for u, d in pairs]
    return {'n_subjects': len(set(subjects)), 'n_prompt_pairs': len(pairs),
            'relation_subject_counts': {rel: len({u['fact']['subject_qid'] for u, _ in pairs
                                                  if u['fact']['relation'] == rel})
                                        for rel in ('P19', 'P20', 'P159', 'P740')},
            'accuracy_d_minus_u': cluster_effect(accuracy, subjects, relations, 10000, 17),
            'gold_loglik_d_minus_u': cluster_effect(loglik, subjects, relations, 10000, 17)}


def main():
    report = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    if report['data_kind'] != 'research' or report['model']['model_id'] != 'CohereLabs/aya-23-8B':
        raise ValueError('Aya research result required')
    rows = []
    for relative in report['run_paths']:
        if not relative.startswith('results/e2-'):
            continue
        rows.extend(json.loads(p.read_text(encoding='utf-8'))
                    for p in (BASE / relative / 'items').glob('*.json'))
    if len(rows) != 480:
        raise ValueError('E2 coverage incomplete')
    with LEDGER.open(encoding='utf-8-sig', newline='') as stream:
        ledger = {r['fact_id']: r for r in csv.DictReader(stream) if r['split'] == 'test'}
    if len(ledger) != 40:
        raise ValueError('Frozen test ledger coverage mismatch')
    out = {'scope': 'Post-outcome exploratory descriptive sensitivity, not a new primary test',
           'input_sha256': {'analysis': sha(ANALYSIS), 'ledger': sha(LEDGER)},
           'languages': [],
           'limits': ['Answer-QID reuse is a place-overlap diagnostic, not proof of memorization.',
                      'Answer text in subject can supply an answer cue; exclusion is post-outcome exploratory.',
                      'The same mLAMA source underlies both cohorts; neither subgroup is an external dataset.']}
    for language in ('he', 'ar'):
        pairs, missing = paired_rows(rows, 'E2', 'U', 'D', language)
        if len(pairs) != 120 or missing:
            raise ValueError('E2 pairing incomplete')
        novel = [(u, d) for u, d in pairs if ledger[u['key']['fact_id']]['pilot_answer_qid_reused'] == 'False']
        reused = [(u, d) for u, d in pairs if ledger[u['key']['fact_id']]['pilot_answer_qid_reused'] == 'True']
        without_cue = [(u, d) for u, d in pairs if ledger[u['key']['fact_id']][f'answer_text_in_subject_{language}'] == 'False']
        out['languages'].append({'language': language,
                                 'answer_qid_not_seen_in_pilot': summary(novel),
                                 'answer_qid_seen_in_pilot': summary(reused),
                                 'answer_text_not_in_subject': summary(without_cue)})
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()

"""Audit every returned English E4 cell, donor and frozen model identity."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/e4c27/results'
PROTOCOL = ROOT / 'configs/e4_reviewed_cohorts_protocol_20260927.json'
OUT = ROOT / 'results/e4-reviewed-cohorts-audit-20260927.json'
FACTS = {
    'pilot': ROOT / 'data/curated/pilot-screened-20260926.jsonl',
    'test': ROOT / 'data/curated/aya-independent-test-v1-screened.jsonl',
}


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    summary = {'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
               'status': 'all 792 cells audited', 'cohorts': {}}
    total = 0
    for cohort, file in FACTS.items():
        facts = {f['fact_id']: f for f in
                 (json.loads(line) for line in file.read_text(encoding='utf-8').splitlines())}
        core = {fid: f for fid, f in facts.items() if f['core_member'] and f['english_eligible']}
        paths = sorted(BASE.glob(f'e4-reviewed-{cohort}-20260927-shard*/*/items/*.json'))
        manifests = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(
            BASE.glob(f'e4-reviewed-{cohort}-20260927-shard*/*/manifest.json'))]
        completions = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(
            BASE.glob(f'e4-reviewed-{cohort}-20260927-shard*/*/completion.json'))]
        if (len(core) != protocol['cohorts'][cohort]['core_count'] or
                len(paths) != len(core) * 24 or len(manifests) != 2 or len(completions) != 2 or
                any(m['model']['model_id'] != 'CohereLabs/aya-23-8B' or
                    m['model']['revision'] != protocol['model_revision'] or
                    m['model']['precision'] != 'nf4' or m['split'] != cohort or
                    m['experiment'] != 'E4' for m in manifests)):
            raise ValueError('E4 cohort identity or coverage invalid: ' + cohort)
        cells = set()
        donors = Counter()
        for path in paths:
            row = json.loads(path.read_text(encoding='utf-8'))
            key = row['key']
            fid = key['fact_id']
            donor_id = row['donor_fact_id']
            if fid not in core or donor_id not in core:
                raise ValueError('E4 subject or donor outside strict core')
            fact, donor = core[fid], core[donor_id]
            if (fid == donor_id or fact['relation'] != donor['relation'] or
                    fact['object_qid'] == donor['object_qid'] or
                    row['fact']['object_qid'] != fact['object_qid'] or
                    row['prompt']['subject'] != fact['subject_labels']['en'] or
                    row['corrupt_score']['answer'] != fact['object_labels']['en'] or
                    row['restored_score']['answer'] != fact['object_labels']['en'] or
                    key['language'] != 'en' or key['template'] != 't1' or
                    key['layer'] not in protocol['layers'] or
                    key['site'] not in protocol['sites']):
                raise ValueError('E4 cell or donor identity invalid')
            gain = (row['restored_score']['sum_logprob'] -
                    row['corrupt_score']['sum_logprob'])
            if (not math.isfinite(gain) or abs(row['restoration_gain'] - gain) > 1e-9):
                raise ValueError('E4 restoration score arithmetic invalid')
            cell = (fid, key['layer'], key['site'])
            if cell in cells:
                raise ValueError('Duplicate E4 cell')
            cells.add(cell)
            donors[donor_id] += 1
        expected = {(fid, layer, site) for fid in core
                    for layer in protocol['layers'] for site in protocol['sites']}
        if cells != expected:
            raise ValueError('E4 planned grid differs from returned grid')
        summary['cohorts'][cohort] = {
            'reviewed_facts': len(facts), 'strict_core_facts': len(core),
            'result_cells': len(paths), 'distinct_donors': len(donors),
            'relation_counts': dict(Counter(f['relation'] for f in core.values())),
            'same_relation_distinct_answer_donor_validated': len(paths),
            'score_arithmetic_validated': len(paths),
        }
        total += len(paths)
    if total != 792:
        raise ValueError('Expected exactly 792 E4 cells')
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'audited_cells': total,
                      'by_cohort': {c: v['result_cells'] for c, v in summary['cohorts'].items()}}))


if __name__ == '__main__':
    main()

"""Actual Aya tokenizer diagnostics for all 57 source- and language-reviewed facts."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.prompts import paired_prompts, template_for
from fragmented_facts.tokenization import split_token_alternative
from fragmented_facts.unicode import base_length

FACTS = ROOT / 'data/curated/pilot-reviewed-20260926.jsonl'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
SNAPSHOT = ROOT / 'model_snapshot.json'
OUT = ROOT / 'outputs/01a09b51/reviewed_tokenization_20260926'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows):
    return {
        'prompt_pairs': len(rows),
        'subjects': len({r['subject_qid'] for r in rows}),
        'mean_u_subject_tokens': round(statistics.mean(r['u_subject_tokens'] for r in rows), 3),
        'mean_d_subject_tokens': round(statistics.mean(r['d_subject_tokens'] for r in rows), 3),
        'mean_delta': round(statistics.mean(r['token_delta'] for r in rows), 3),
        'median_delta': statistics.median(r['token_delta'] for r in rows),
        'positive_delta': sum(r['token_delta'] > 0 for r in rows),
        'zero_delta': sum(r['token_delta'] == 0 for r in rows),
        'negative_delta': sum(r['token_delta'] < 0 for r in rows),
        'u_first_token_boundary_crossing': sum(r['u_boundary_crossing'] for r in rows),
        'd_first_token_boundary_crossing': sum(r['d_boundary_crossing'] for r in rows),
        'u_last_token_boundary_crossing': sum(r['u_last_token_crossing'] for r in rows),
        'd_last_token_boundary_crossing': sum(r['d_last_token_crossing'] for r in rows),
    }


def main():
    facts = [json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines()]
    templates = json.loads(TEMPLATES.read_text(encoding='utf-8'))
    snapshot = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    if len(facts) != 57 or len({f['subject_qid'] for f in facts}) != 57:
        raise ValueError('Expected the 57-subject reviewed cohort')
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / 'data/raw/aya_23_8b'), local_files_only=True, use_fast=True)
    rows = []
    coverage = {lang: {'u': 0, 'd': 0, 'both': 0} for lang in ('he', 'ar')}
    for fact in facts:
        for lang in ('he', 'ar'):
            pair = fact['pairs'][lang]
            for tid in ('t1', 't2', 't3'):
                template = template_for(templates, fact['relation'], lang, tid)
                u, d = paired_prompts(tokenizer, template, pair, lang)
                if tid == 't1':
                    alt_u = split_token_alternative(tokenizer, u) is not None
                    alt_d = split_token_alternative(tokenizer, d) is not None
                    coverage[lang]['u'] += alt_u
                    coverage[lang]['d'] += alt_d
                    coverage[lang]['both'] += alt_u and alt_d
                rows.append({
                    'fact_id': fact['fact_id'], 'subject_qid': fact['subject_qid'],
                    'relation': fact['relation'], 'language': lang, 'template': tid,
                    'u': pair['U'], 'd': pair['D'],
                    'base_length': base_length(pair['U'], lang),
                    'u_subject_tokens': len(u.subject_indices),
                    'd_subject_tokens': len(d.subject_indices),
                    'token_delta': len(d.subject_indices) - len(u.subject_indices),
                    'u_prompt_tokens': len(u.input_ids), 'd_prompt_tokens': len(d.input_ids),
                    'u_boundary_crossing': u.boundary_crossing,
                    'd_boundary_crossing': d.boundary_crossing,
                    'u_last_token_crossing': u.offsets[u.subject_indices[-1]][1] > u.span[1],
                    'd_last_token_crossing': d.offsets[d.subject_indices[-1]][1] > d.span[1],
                })
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / 'reviewed_57_prompt_tokenization.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    relations = defaultdict(list)
    for row in rows:
        groups[row['language']].append(row)
        relations[row['language'] + '/' + row['relation']].append(row)
    report = {
        'status': 'actual_aya_tokenizer_diagnostic_not_factual_accuracy',
        'model_id': snapshot['model_id'], 'revision': snapshot['revision'],
        'tokenizer_class': type(tokenizer).__name__,
        'inputs_sha256': {str(path.relative_to(ROOT)): sha(path)
                          for path in (FACTS, TEMPLATES, SNAPSHOT)},
        'overall': summary(rows),
        'by_language': {key: summary(value) for key, value in groups.items()},
        'by_language_relation': {key: summary(value) for key, value in relations.items()},
        'same_string_plus_one_token_coverage_t1': coverage,
        'csv': str(csv_path.relative_to(ROOT)),
    }
    (OUT / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

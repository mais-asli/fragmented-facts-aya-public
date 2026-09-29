"""Tokenizer-only feasibility of natural question-word marks outside the name.

No new prompts are approved or run through Aya by this script. A context mark
is a negative control candidate, not an equivalent-length causal control.
"""

from collections import Counter
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.prompts import encode_prompt  # noqa: E402

FACTS = ROOT / 'data/curated/aya-independent-test-v1.jsonl'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
OUT = ROOT / 'results/context-mark-control-feasibility-20260927.json'
MARKS = {'he': ('היכן', 'הֵיכָן'), 'ar': ('أين', 'أَيْنَ')}


def main():
    facts = [json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines()]
    templates = json.loads(TEMPLATES.read_text(encoding='utf-8'))
    if len(facts) != 40:
        raise ValueError('Expected frozen 40-subject test')
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'data/raw/aya_23_8b'),
                                             local_files_only=True, use_fast=True)
    rows = []
    for fact in facts:
        for language in ('he', 'ar'):
            template = templates['evaluation'][fact['relation']][language]['t1']['text']
            old_word, new_word = MARKS[language]
            if template.count(old_word) != 1 or old_word in fact['pairs'][language]['U']:
                raise ValueError('Context-word replacement is not isolated')
            marked_context = template.replace(old_word, new_word)
            subject_u = fact['pairs'][language]['U']
            subject_d = fact['pairs'][language]['D']
            u = encode_prompt(tokenizer, template, subject_u, language)
            d = encode_prompt(tokenizer, template, subject_d, language)
            c = encode_prompt(tokenizer, marked_context, subject_u, language)
            if u.subject != c.subject or u.text == c.text:
                raise ValueError('Invalid context-only candidate')
            rows.append({'fact_id': fact['fact_id'], 'relation': fact['relation'],
                         'language': language, 'context_word_u': old_word,
                         'context_word_marked_candidate': new_word,
                         'u_tokens': len(u.input_ids), 'd_subject_tokens': len(d.input_ids),
                         'context_mark_tokens': len(c.input_ids),
                         'subject_mark_total_token_delta': len(d.input_ids) - len(u.input_ids),
                         'context_mark_total_token_delta': len(c.input_ids) - len(u.input_ids),
                         'whole_prompt_delta_equal': len(d.input_ids) == len(c.input_ids)})
    counts = {}
    for lang in ('he', 'ar'):
        subset = [r for r in rows if r['language'] == lang]
        counts[lang] = {'facts': len(subset),
                        'whole_prompt_length_matched': sum(r['whole_prompt_delta_equal'] for r in subset),
                        'context_delta_distribution': dict(sorted(Counter(r['context_mark_total_token_delta'] for r in subset).items())),
                        'subject_delta_distribution': dict(sorted(Counter(r['subject_mark_total_token_delta'] for r in subset).items()))}
    result = {'scope': 'Unreviewed natural question-word marking candidate, tokenizer-only; no Aya inference',
              'template': 't1', 'candidate_replacements': MARKS, 'counts': counts, 'rows': rows,
              'decision_rule': 'If few whole-prompt lengths match, do not describe this as a length-matched control. New natural-language human review required before any model run.'}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

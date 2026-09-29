"""Find feasible same-text outside-subject BPE controls without Aya outcomes."""

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
OUT = ROOT / 'results/e3-outside-control-position-audit-20260927.json'


def candidates(tokenizer, prompt):
    special = set(tokenizer.all_special_ids)
    found = []
    for index, token_id in enumerate(prompt.input_ids):
        if token_id in special:
            continue
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(piece, str) or len(piece) < 2:
            continue
        for cut in range(1, len(piece)):
            halves = (piece[:cut], piece[cut:])
            ids = [tokenizer.convert_tokens_to_ids(half) for half in halves]
            if any(not isinstance(value, int) or value in special or
                   tokenizer.convert_ids_to_tokens(value) != half
                   for value, half in zip(ids, halves)):
                continue
            alt = prompt.input_ids[:index] + ids + prompt.input_ids[index + 1:]
            if tokenizer.decode(alt, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) == prompt.text:
                found.append({'index': index, 'original_token_id': token_id,
                              'replacement_token_ids': ids,
                              'region': ('before' if index < prompt.subject_indices[0]
                                         else 'after' if index > prompt.subject_indices[-1]
                                         else 'subject'),
                              'distance_to_subject': (prompt.subject_indices[0] - index if index < prompt.subject_indices[0]
                                                      else index - prompt.subject_indices[-1]
                                                      if index > prompt.subject_indices[-1] else 0)})
                break
    return found


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'data/raw/aya_23_8b'),
                                              local_files_only=True, use_fast=True)
    facts = [json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines()]
    templates = json.loads(TEMPLATES.read_text(encoding='utf-8'))
    rows = []
    for fact in facts:
        for language in ('he', 'ar'):
            template = templates['evaluation'][fact['relation']][language]['t1']['text']
            prompt = encode_prompt(tokenizer, template, fact['pairs'][language]['U'], language)
            options = candidates(tokenizer, prompt)
            before = sorted((c for c in options if c['region'] == 'before'),
                            key=lambda c: (c['distance_to_subject'], c['index']))
            after = sorted((c for c in options if c['region'] == 'after'),
                           key=lambda c: (c['distance_to_subject'], c['index']))
            rows.append({'fact_id': fact['fact_id'], 'relation': fact['relation'],
                         'language': language,
                         'earliest_before': min(before, key=lambda c: c['index']) if before else None,
                         'nearest_before': before[0] if before else None,
                         'nearest_after': after[0] if after else None,
                         'outside_candidate_count': len(before) + len(after)})
    result = {'scope': 'Tokenizer-only audit after the first byte-level E3 result; no further Aya outcomes inspected',
              'rows': rows, 'counts': {}}
    for language in ('he', 'ar'):
        subset = [r for r in rows if r['language'] == language]
        result['counts'][language] = {
            'nearest_before_available': sum(r['nearest_before'] is not None for r in subset),
            'nearest_after_available': sum(r['nearest_after'] is not None for r in subset),
            'nearest_before_ids': Counter(str(r['nearest_before']['original_token_id'])
                                          for r in subset if r['nearest_before']),
            'nearest_after_ids': Counter(str(r['nearest_after']['original_token_id'])
                                         for r in subset if r['nearest_after']),
        }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': result['counts'], 'output': str(OUT)}, default=dict))


if __name__ == '__main__':
    main()

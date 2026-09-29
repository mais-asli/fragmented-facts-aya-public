"""Scan exact same-text two-piece BPE alternatives, including byte fragments.

This is a method-development feasibility scan, not an Aya model run. It uses
only existing tokenizer IDs and accepts a candidate only when the complete
decoded chat prompt remains byte-for-byte identical.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.prompts import encode_prompt  # noqa: E402

FACTS = ROOT / 'data/curated/aya-independent-test-v1.jsonl'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
TOKENIZER = ROOT / 'data/raw/aya_23_8b'
OUT = ROOT / 'results/e3-byte-level-eligibility-feasibility-20260927.json'


def first_exact_split(tokenizer, prompt, inside):
    special = set(tokenizer.all_special_ids)
    positions = prompt.subject_indices if inside else range(prompt.subject_indices[0])
    for index in positions:
        if inside:
            a, b = prompt.offsets[index]
            if a < prompt.span[0] or b > prompt.span[1]:
                continue
        token_id = prompt.input_ids[index]
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
            alternative = prompt.input_ids[:index] + ids + prompt.input_ids[index + 1:]
            if tokenizer.decode(alternative, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != prompt.text:
                continue
            if len(alternative) != len(prompt.input_ids) + 1:
                raise ValueError('Expected exactly one extra token')
            return {'replaced_index': index, 'original_token_id': token_id,
                    'replacement_token_ids': ids, 'token_string_cut': cut,
                    'alternative_input_ids_sha256': hashlib.sha256(
                        json.dumps(alternative, separators=(',', ':')).encode()).hexdigest()}
    return None


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    facts = [json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines() if line.strip()]
    templates = json.loads(TEMPLATES.read_text(encoding='utf-8'))
    rows = []
    if len(facts) != 40:
        raise ValueError('Expected 40 frozen facts')
    for fact in facts:
        for language in ('he', 'ar', 'en'):
            template = templates['evaluation'][fact['relation']][language]['t1']['text']
            name = fact['pairs'][language]['U'] if language != 'en' else fact['subject_labels']['en']
            prompt = encode_prompt(tokenizer, template, name, language)
            inside = first_exact_split(tokenizer, prompt, True)
            outside = first_exact_split(tokenizer, prompt, False)
            rows.append({'fact_id': fact['fact_id'], 'subject_qid': fact['subject_qid'],
                         'relation': fact['relation'], 'language': language,
                         'canonical_input_ids_sha256': hashlib.sha256(json.dumps(
                             prompt.input_ids, separators=(',', ':')).encode()).hexdigest(),
                         'inside': inside, 'outside': outside,
                         'matched': inside is not None and outside is not None})
    result = {'scope': 'Tokenizer-only post-E3 method feasibility; no Aya outcomes read',
              'rule': 'Replace the earliest eligible BPE token by its earliest pair of existing token IDs whose concatenated token strings match it; full decoded chat prompt must remain identical; exactly one extra token in either arm',
              'input_sha256': {'facts': hashlib.sha256(FACTS.read_bytes()).hexdigest(),
                               'templates': hashlib.sha256(TEMPLATES.read_bytes()).hexdigest(),
                               'tokenizer_json': hashlib.sha256((TOKENIZER / 'tokenizer.json').read_bytes()).hexdigest()},
              'rows': rows, 'counts': {}}
    for language in ('en', 'he', 'ar'):
        subset = [r for r in rows if r['language'] == language]
        result['counts'][language] = {
            'inside': sum(r['inside'] is not None for r in subset),
            'outside': sum(r['outside'] is not None for r in subset),
            'matched': sum(r['matched'] for r in subset),
            'matched_by_relation': dict(Counter(r['relation'] for r in subset if r['matched']))}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'counts': result['counts']}, indent=2))


if __name__ == '__main__':
    main()

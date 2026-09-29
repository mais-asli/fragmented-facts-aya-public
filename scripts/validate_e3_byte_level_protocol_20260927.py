"""Run the frozen E3 prompt invariants locally without loading Aya weights."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.prompts import encode_prompt, template_for  # noqa: E402
from fragmented_facts.io import read_json, read_jsonl  # noqa: E402
sys.path.insert(0, str(ROOT / 'project_plan/feasibility'))
from e3_byte_level_run_20260927 import alternative, verify_files  # noqa: E402


def main():
    protocol = read_json(ROOT / 'configs/e3_byte_level_protocol_20260927.json')
    verify_files(protocol)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / 'data/raw/aya_23_8b'), local_files_only=True, use_fast=True)
    if hashlib.sha256((ROOT / 'data/raw/aya_23_8b/tokenizer.json').read_bytes()).hexdigest() != protocol['tokenizer_sha256']:
        raise ValueError('Tokenizer SHA differs')
    facts = read_jsonl(ROOT / 'data/curated/aya-independent-test-v1-screened.jsonl')
    templates = read_json(ROOT / 'data/review/imported_ReviewerA_20260926/templates.json')
    eligibility = read_json(ROOT / 'results/e3-byte-level-eligibility-feasibility-20260927.json')
    saved = {(r['fact_id'], r['language']): r for r in eligibility['rows']}
    counts = Counter()
    for fact in facts:
        for language in ('he', 'ar'):
            template = template_for(templates, fact['relation'], language, 't1')
            original = encode_prompt(tokenizer, template,
                                     fact['subject_labels'][language], language)
            row = saved[(fact['fact_id'], language)]
            for location in ('inside', 'outside'):
                changed = alternative(tokenizer, original, row[location])
                if changed.text != original.text or len(changed.input_ids) != len(original.input_ids) + 1:
                    raise ValueError('Visible text or added token differs')
                counts[(language, location)] += 1
    if set(counts.values()) != {40} or len(counts) != 4:
        raise ValueError('Incomplete matched contrasts')
    print(json.dumps({'validated': {str(k): v for k, v in counts.items()},
                      'model_loaded': False}))


if __name__ == '__main__':
    main()

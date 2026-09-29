"""Freeze two outside-subject byte-split controls before their Aya outcomes.

This is an exploratory sensitivity analysis designed after the first E3 result.
It cannot identify token count independently of token identity and position.
"""

import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.prompts import EncodedPrompt, encode_prompt, template_for  # noqa: E402

AUDIT = ROOT / 'results/e3-outside-control-position-audit-20260927.json'
FACTS = ROOT / 'data/curated/aya-independent-test-v1-screened.jsonl'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
OLD_PROTOCOL = ROOT / 'configs/e3_byte_level_protocol_20260927.json'
OLD_ELIGIBILITY = ROOT / 'results/e3-byte-level-eligibility-feasibility-20260927.json'
OUT = ROOT / 'configs/e3_position_controls_protocol_20260927.json'
CODE = ROOT / 'project_plan/feasibility/e3_position_controls_run_20260927.py'
SLURM = ROOT / 'slurm/e3-position-controls-20260927.slurm'
WORKFLOW = ROOT / 'project_plan/feasibility/e3_position_controls_workflow_tau_20260927.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def prompt_for_local_tokenizer(tokenizer, template, subject, language):
    """Use the rendered-text path when local Transformers 5 changes chat-template IDs.

    The result is accepted only if it reproduces the original frozen E3 ID hash.
    Aya inference itself uses the original pinned TAU environment.
    """
    try:
        return encode_prompt(tokenizer, template, subject, language)
    except ValueError as error:
        if 'Chat-template token IDs disagree' not in str(error):
            raise
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': template.replace('{subject}', subject)}],
                                         tokenize=False, add_generation_prompt=True)
    if text.count(subject) != 1:
        raise ValueError('Ambiguous subject span')
    start, end = text.index(subject), text.index(subject) + len(subject)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(x) for x in encoded['offset_mapping']]
    indices = [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]
    delimiter = next((i for i, (a, b) in enumerate(offsets)
                      if i > indices[-1] and b > a and a >= end), None)
    if not indices or delimiter is None:
        raise ValueError('Subject or delimiter not tokenized')
    return EncodedPrompt(text, encoded['input_ids'], subject, language, (start, end), offsets,
                         indices, delimiter, any(offsets[i][0] < start or offsets[i][1] > end
                                                 for i in indices))


def main():
    if OUT.exists():
        raise FileExistsError('E3 positional controls already frozen')
    prior = json.loads(OLD_PROTOCOL.read_text(encoding='utf-8'))
    original_eligibility = json.loads(OLD_ELIGIBILITY.read_text(encoding='utf-8'))
    original_ids = {(r['fact_id'], r['language']): r['canonical_input_ids_sha256']
                    for r in original_eligibility['rows']}
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    facts = {f['fact_id']: f for f in
             (json.loads(line) for line in FACTS.read_text(encoding='utf-8').splitlines())}
    templates = json.loads(TEMPLATES.read_text(encoding='utf-8'))
    tokenizer_path = ROOT / 'data/raw/aya_23_8b'
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True,
                                               use_fast=True)
    if sha(tokenizer_path / 'tokenizer.json') != prior['tokenizer_sha256']:
        raise ValueError('Aya tokenizer changed')
    if set(facts) != set(prior['selected_fact_ids']) or len(audit['rows']) != 80:
        raise ValueError('E3 held-out cohort changed')
    rows = []
    for row in audit['rows']:
        fact = facts[row['fact_id']]
        language = row['language']
        template = template_for(templates, fact['relation'], language, 't1')
        prompt = prompt_for_local_tokenizer(tokenizer, template,
                                            fact['subject_labels'][language], language)
        if ids_hash(prompt.input_ids) != original_ids[(row['fact_id'], language)]:
            raise ValueError('Local token IDs differ from original E3 freeze')
        record = {'fact_id': row['fact_id'], 'language': language,
                  'relation': fact['relation'], 'canonical_input_ids_sha256': ids_hash(prompt.input_ids),
                  'controls': {}}
        for name in ('nearest_before', 'nearest_after'):
            saved = row[name]
            index, old, parts = saved['index'], saved['original_token_id'], saved['replacement_token_ids']
            if (saved['region'] != ('before' if name == 'nearest_before' else 'after')
                    or prompt.input_ids[index] != old or len(parts) != 2 or
                    (index >= prompt.subject_indices[0] if name == 'nearest_before'
                     else index <= prompt.subject_indices[-1])):
                raise ValueError('Outside-subject control does not match audit')
            tokens = [tokenizer.convert_ids_to_tokens(i) for i in [old, *parts]]
            alt = prompt.input_ids[:index] + parts + prompt.input_ids[index + 1:]
            if (not all(isinstance(t, str) for t in tokens) or
                    tokens[0] != tokens[1] + tokens[2] or
                    len(alt) != len(prompt.input_ids) + 1 or
                    tokenizer.decode(alt, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False) != prompt.text):
                raise ValueError('Control changes decoded prompt or token count')
            record['controls'][name] = {
                'replaced_index': index, 'original_token_id': old,
                'replacement_token_ids': parts,
                'alternative_input_ids_sha256': ids_hash(alt),
                'distance_to_subject': saved['distance_to_subject']}
        rows.append(record)
    if len({(r['fact_id'], r['language']) for r in rows}) != 80:
        raise ValueError('Duplicate or missing E3 positional pair')
    inputs = [AUDIT, FACTS, TEMPLATES, OLD_PROTOCOL, OLD_ELIGIBILITY,
              CODE, SLURM, WORKFLOW,
              ROOT / 'configs/study_independent_test_v1.json',
              ROOT / 'src/fragmented_facts/experiments.py',
              ROOT / 'src/fragmented_facts/model.py',
              ROOT / 'src/fragmented_facts/prompts.py',
              ROOT / 'src/fragmented_facts/scoring.py']
    payload = {
        'schema_version': 1,
        'method_status': 'post-original-E3 and post-byte-E3 exploratory sensitivity',
        'model_id': prior['model_id'], 'model_revision': prior['model_revision'],
        'precision': prior['precision'], 'tokenizer_sha256': prior['tokenizer_sha256'],
        'languages': ['he', 'ar'], 'template': 't1', 'shards': 2,
        'conditions': ['nearest_before', 'nearest_after'],
        'purpose': 'Check whether outside-subject control placement changes the inside-versus-outside contrast; do not select the favorable control post hoc.',
        'interpretation_limit': 'Both controls add one token while preserving visible text but change token identity and position.',
        'selected_fact_ids': prior['selected_fact_ids'],
        'rows': sorted(rows, key=lambda r: (r['fact_id'], r['language'])),
        'file_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in inputs}}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'pairs': len(rows), 'expected_aya_records': 160,
                      'sha256': sha(OUT)}))


if __name__ == '__main__':
    main()

"""Score frozen nearest-before/after E3 controls with the pinned Aya checkpoint."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.experiments import fact_record, observation  # noqa: E402
from fragmented_facts.io import ResultStore, digest, read_json, read_jsonl, write_json  # noqa: E402
from fragmented_facts.model import load_runner  # noqa: E402
from fragmented_facts.prompts import encode_prompt, template_for  # noqa: E402

PROTOCOL = ROOT / 'configs/e3_position_controls_protocol_20260927.json'
FACTS = ROOT / 'data/curated/aya-independent-test-v1-screened.jsonl'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
CONFIG = ROOT / 'configs/study_independent_test_v1.json'
SNAPSHOT = ROOT / 'model_snapshot.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def alternative(tokenizer, prompt, saved):
    index = saved['replaced_index']
    old, parts = saved['original_token_id'], saved['replacement_token_ids']
    if prompt.input_ids[index] != old or len(parts) != 2:
        raise ValueError('Frozen outside control changed')
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in [old, *parts]]
    if tokens[0] != tokens[1] + tokens[2]:
        raise ValueError('Byte-string split changed')
    ids = prompt.input_ids[:index] + parts + prompt.input_ids[index + 1:]
    if (ids_hash(ids) != saved['alternative_input_ids_sha256'] or
            len(ids) != len(prompt.input_ids) + 1 or
            tokenizer.decode(ids, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False) != prompt.text):
        raise ValueError('Outside control changes visible text or token count')
    a, b = prompt.offsets[index]
    offsets = prompt.offsets[:index] + [(a, b), (a, b)] + prompt.offsets[index + 1:]
    subject_indices = [i + (i > index) for i in prompt.subject_indices]
    return replace(prompt, input_ids=ids, offsets=offsets,
                   subject_indices=subject_indices,
                   delimiter_index=prompt.delimiter_index + (prompt.delimiter_index > index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True, choices=[0, 1])
    parser.add_argument('--num-shards', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.num_shards != 2:
        raise ValueError('Frozen positional E3 layout uses two shards')
    output = (ROOT / args.output).resolve()
    if output != ROOT / f'results/e3-position-controls-20260927-shard{args.shard}':
        raise ValueError('Unexpected E3 positional-control output path')
    protocol = read_json(PROTOCOL)
    for relative, expected in protocol['file_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen E3 positional input differs: ' + relative)
    if (protocol['model_id'] != 'CohereLabs/aya-23-8B' or
            protocol['precision'] != 'nf4' or len(protocol['rows']) != 80):
        raise ValueError('Wrong E3 positional protocol')
    facts = read_jsonl(FACTS)
    if sorted(f['fact_id'] for f in facts) != protocol['selected_fact_ids']:
        raise ValueError('E3 positional fact identities changed')
    frozen = {(r['fact_id'], r['language']): r for r in protocol['rows']}
    if len(frozen) != 80:
        raise ValueError('E3 positional pairs missing')
    config, templates = read_json(CONFIG), read_json(TEMPLATES)
    runner = load_runner(SNAPSHOT, config)
    if (runner.identity['revision'] != protocol['model_revision'] or
            any(f['screen_model_hash'] != digest(runner.identity) for f in facts)):
        raise ValueError('Model differs from pinned Aya screen')
    manifest = {'schema_version': 1, 'experiment': 'E3_position_controls',
                'split': 'test', 'method_status': protocol['method_status'],
                'model': runner.identity, 'protocol_sha256': sha(PROTOCOL),
                'facts_sha256': sha(FACTS), 'shard': args.shard,
                'num_shards': args.num_shards, 'data_kind': 'research'}
    completed = resumed = 0
    expected = []
    with ResultStore(output, manifest) as store:
        for fact in sorted(facts, key=lambda item: item['fact_id']):
            if int(digest(fact['subject_qid']), 16) % 2 != args.shard:
                continue
            for language in ('he', 'ar'):
                template = template_for(templates, fact['relation'], language, 't1')
                prompt = encode_prompt(runner.tokenizer, template,
                                       fact['subject_labels'][language], language)
                saved = frozen[(fact['fact_id'], language)]
                if ids_hash(prompt.input_ids) != saved['canonical_input_ids_sha256']:
                    raise ValueError('Canonical prompt changed after freeze')
                for variant in protocol['conditions']:
                    control = alternative(runner.tokenizer, prompt, saved['controls'][variant])
                    key = {'experiment': 'E3_position_controls',
                           'fact_id': fact['fact_id'], 'language': language,
                           'template': 't1', 'seed': 17, 'variant': variant,
                           'extra_tokens': 1}
                    expected.append(digest(key))
                    if store.get(key):
                        resumed += 1
                        continue
                    value = observation(runner, control, fact, None,
                                        fact['object_labels'][language], generate=False)
                    store.put(key, {'fact': fact_record(fact), **value,
                                    'method_status': protocol['method_status'],
                                    'runtime': runner.runtime()})
                    completed += 1
        summary = {'run_id': store.run_id, 'run_path': str(store.path),
                   'experiment': 'E3_position_controls', 'split': 'test',
                   'status': 'complete', 'data_kind': 'research',
                   'completed_now': completed, 'resumed_items': resumed,
                   'total_items': len(store.rows()),
                   'expected_count': len(set(expected)),
                   'expected_keys': sorted(set(expected))}
        if completed + resumed != len(expected) or len(store.rows()) != len(expected):
            raise ValueError('Incomplete E3 positional controls')
        write_json(store.path / 'completion.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'expected_keys'}), flush=True)


if __name__ == '__main__':
    main()

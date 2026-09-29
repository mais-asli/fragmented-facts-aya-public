"""Versioned Aya E3 byte-level same-text control, separate from frozen E1-E5.

Run only with the SHA-frozen protocol and reviewed 40-subject test cohort.
The post-E3 method extension is exploratory; it does not overwrite old E3.
"""

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

PROTOCOL = ROOT / 'configs/e3_byte_level_protocol_20260927.json'
ELIGIBILITY = ROOT / 'results/e3-byte-level-eligibility-feasibility-20260927.json'
FACTS = ROOT / 'data/curated/aya-independent-test-v1-screened.jsonl'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
CONFIG = ROOT / 'configs/study_independent_test_v1.json'
SNAPSHOT = ROOT / 'model_snapshot.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(protocol):
    for relative, expected in protocol['file_sha256'].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Frozen file differs: ' + relative)


def alternative(tokenizer, prompt, saved):
    index = saved['replaced_index']
    old_id = saved['original_token_id']
    parts = saved['replacement_token_ids']
    if (prompt.input_ids[index] != old_id or len(parts) != 2 or
            any(tokenizer.convert_ids_to_tokens(part) is None for part in parts)):
        raise ValueError('Frozen token replacement differs')
    old_piece = tokenizer.convert_ids_to_tokens(old_id)
    new_piece = ''.join(tokenizer.convert_ids_to_tokens(part) for part in parts)
    if old_piece != new_piece or old_piece[:saved['token_string_cut']] != tokenizer.convert_ids_to_tokens(parts[0]):
        raise ValueError('BPE byte-string split differs')
    ids = prompt.input_ids[:index] + parts + prompt.input_ids[index + 1:]
    if (len(ids) != len(prompt.input_ids) + 1 or
            hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            != saved['alternative_input_ids_sha256'] or
            tokenizer.decode(ids, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False) != prompt.text):
        raise ValueError('Alternative changes the full decoded prompt')
    a, b = prompt.offsets[index]
    offsets = prompt.offsets[:index] + [(a, b), (a, b)] + prompt.offsets[index + 1:]
    subject_indices = []
    for old in prompt.subject_indices:
        subject_indices.extend([old, old + 1] if old == index else [old + (old > index)])
    return replace(prompt, input_ids=ids, offsets=offsets,
                   subject_indices=subject_indices,
                   delimiter_index=prompt.delimiter_index + (prompt.delimiter_index > index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--num-shards', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.num_shards != 2 or args.shard not in (0, 1):
        raise ValueError('Frozen run requires exactly two shards')
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Output must be inside results')
    protocol = read_json(PROTOCOL)
    verify_files(protocol)
    if protocol['model_id'] != 'CohereLabs/aya-23-8B' or protocol['precision'] != 'nf4':
        raise ValueError('Wrong Aya protocol identity')
    eligibility = read_json(ELIGIBILITY)
    facts = read_jsonl(FACTS)
    templates = read_json(TEMPLATES)
    config = read_json(CONFIG)
    if (len(facts) != 40 or
            sorted(f['fact_id'] for f in facts) != protocol['selected_fact_ids'] or
            eligibility['input_sha256']['tokenizer_json'] != protocol['tokenizer_sha256']):
        raise ValueError('Frozen cohort or tokenizer eligibility differs')
    eligible = {(r['fact_id'], r['language']): r for r in eligibility['rows']}
    if len(eligible) != 120 or any(not eligible[(f['fact_id'], language)]['matched']
                                   for f in facts for language in ('he', 'ar')):
        raise ValueError('Frozen Hebrew/Arabic matched coverage differs')
    runner = load_runner(SNAPSHOT, config)
    if (runner.identity['revision'] != protocol['model_revision'] or
            any(f.get('screen_model_hash') != digest(runner.identity) for f in facts)):
        raise ValueError('Model differs from original English screen')
    manifest = {'schema_version': 1, 'experiment': 'E3_byte_level', 'split': 'test',
                'method_status': 'post-E3 exploratory extension',
                'model': runner.identity, 'protocol_sha256': sha(PROTOCOL),
                'eligibility_sha256': sha(ELIGIBILITY), 'facts_sha256': sha(FACTS),
                'templates_sha256': sha(TEMPLATES), 'shard': args.shard,
                'num_shards': args.num_shards, 'data_kind': 'research'}
    expected = []
    completed = 0
    resumed = 0
    with ResultStore(output, manifest) as store:
        for fact in sorted(facts, key=lambda item: item['fact_id']):
            if int(digest(fact['subject_qid']), 16) % args.num_shards != args.shard:
                continue
            for language in ('he', 'ar'):
                template = template_for(templates, fact['relation'], language, 't1')
                prompt = encode_prompt(runner.tokenizer, template,
                                       fact['subject_labels'][language], language)
                frozen = eligible[(fact['fact_id'], language)]
                if hashlib.sha256(json.dumps(prompt.input_ids, separators=(',', ':')).encode()).hexdigest() != frozen['canonical_input_ids_sha256']:
                    raise ValueError('Canonical prompt differs from frozen tokenizer scan')
                conditions = [('canonical', prompt),
                              ('subject_split', alternative(runner.tokenizer, prompt, frozen['inside'])),
                              ('outside_split', alternative(runner.tokenizer, prompt, frozen['outside']))]
                if len({condition.text for _, condition in conditions}) != 1:
                    raise ValueError('Visible E3 text differs across conditions')
                gold = fact['object_labels'][language]
                for variant, condition in conditions:
                    key = {'experiment': 'E3_byte_level', 'fact_id': fact['fact_id'],
                           'language': language, 'template': 't1', 'seed': 17,
                           'variant': variant, 'extra_tokens': 0 if variant == 'canonical' else 1}
                    expected.append(digest(key))
                    if store.get(key):
                        resumed += 1
                        continue
                    value = observation(runner, condition, fact, None, gold, generate=False)
                    store.put(key, {'fact': fact_record(fact), **value,
                                    'method_status': 'post-E3 exploratory extension',
                                    'runtime': runner.runtime()})
                    completed += 1
        summary = {'run_id': store.run_id, 'run_path': str(store.path),
                   'experiment': 'E3_byte_level', 'split': 'test', 'status': 'complete',
                   'data_kind': 'research', 'completed_now': completed,
                   'resumed_items': resumed,
                   'total_items': len(store.rows()), 'expected_count': len(set(expected)),
                   'expected_keys': sorted(set(expected))}
        if completed + resumed != len(expected) or completed + resumed != len(store.rows()):
            raise ValueError('Incomplete new E3 run')
        write_json(store.path / 'completion.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'expected_keys'}), flush=True)


if __name__ == '__main__':
    main()

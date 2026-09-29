"""Retrospective Aya E4 localization on already reviewed, screened cohorts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.experiments import run_experiment  # noqa: E402
from fragmented_facts.io import digest, read_json, read_jsonl  # noqa: E402
from fragmented_facts.model import load_runner  # noqa: E402

PROTOCOL = ROOT / 'configs/e4_reviewed_cohorts_protocol_20260927.json'
CONFIG = ROOT / 'configs/e4_reviewed_cohorts_20260927.json'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
SNAPSHOT = ROOT / 'model_snapshot.json'
COHORTS = {
    'pilot': ROOT / 'data/curated/pilot-screened-20260926.jsonl',
    'test': ROOT / 'data/curated/aya-independent-test-v1-screened.jsonl',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=COHORTS, required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--num-shards', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.num_shards != 2:
        raise ValueError('Frozen E4 layout uses exactly two shards per cohort')
    output = (ROOT / args.output).resolve()
    expected_prefix = f'e4-reviewed-{args.cohort}-20260927-shard{args.shard}'
    if output != ROOT / 'results' / expected_prefix:
        raise ValueError('Unexpected E4 output path')
    protocol = read_json(PROTOCOL)
    for relative, expected in protocol['file_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen E4 reviewed cohort input differs: ' + relative)
    config, templates = read_json(CONFIG), read_json(TEMPLATES)
    if (config['languages'] != ['en'] or config['population'] != 'core'
            or not config['readout_layers_frozen'] or
            config['readout_layers'] != protocol['layers']):
        raise ValueError('E4 reviewed-cohort configuration changed')
    facts = read_jsonl(COHORTS[args.cohort])
    expected = protocol['cohorts'][args.cohort]
    core = [f for f in facts if f.get('core_member')]
    if (len(facts) != expected['reviewed_count'] or len(core) != expected['core_count']
            or sorted(f['fact_id'] for f in core) != expected['core_fact_ids']
            or any(f['split'] != args.cohort or f['review']['reviewer'] != 'Reviewer A'
                   or not f['review']['pre_release_verified'] for f in facts)):
        raise ValueError('Reviewed English E4 cohort changed')
    for relation in ('P19', 'P20', 'P159', 'P740'):
        if len({f['object_qid'] for f in core if f['relation'] == relation}) < 2:
            raise ValueError('No E4 donor with a distinct same-relation answer')
    runner = load_runner(SNAPSHOT, config)
    if (runner.identity['revision'] != protocol['model_revision'] or
            any(f['screen_model_hash'] != digest(runner.identity) for f in facts)):
        raise ValueError('E4 cohort or screen checkpoint differs from pinned Aya')
    result = run_experiment(runner, facts, templates, config,
                            {'protocol_hash': sha(PROTOCOL), 'fixture': False},
                            'E4', args.cohort, output, shard=args.shard, num_shards=2)
    print(json.dumps({'cohort': args.cohort, **result}), flush=True)


if __name__ == '__main__':
    main()

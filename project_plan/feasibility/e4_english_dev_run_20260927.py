"""Versioned English-only Aya E4 screening and localization on reviewed dev facts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.experiments import run_experiment, screen  # noqa: E402
from fragmented_facts.io import digest, read_json, read_jsonl  # noqa: E402
from fragmented_facts.model import load_runner  # noqa: E402

PROTOCOL = ROOT / 'configs/e4_english_dev_protocol_20260927.json'
FACTS = ROOT / 'data/curated/aya-e4-english-dev-ReviewerA-reviewed-20260927.jsonl'
SCREENED = ROOT / 'data/curated/aya-e4-english-dev-screened-20260927.jsonl'
TEMPLATES = ROOT / 'data/review/imported_ReviewerA_20260926/templates.json'
CONFIG = ROOT / 'configs/e4_english_dev_20260927.json'
SNAPSHOT = ROOT / 'model_snapshot.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['screen', 'e4'], required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if (args.mode == 'screen' and (args.shard, args.num_shards) != (0, 1)) or (
            args.mode == 'e4' and (args.num_shards != 2 or args.shard not in (0, 1))):
        raise ValueError('Unexpected frozen Slurm shard layout')
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Output must remain inside results')
    protocol = read_json(PROTOCOL)
    for relative, expected in protocol['file_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('E4 frozen source differs: ' + relative)
    if protocol['model_revision'] != '89da1a0ed02d6130f93ae0ffdbedb63b760c0471':
        raise ValueError('Wrong Aya revision')
    config, templates = read_json(CONFIG), read_json(TEMPLATES)
    if config['languages'] != ['en'] or config['population'] != 'core':
        raise ValueError('E4 must be English only on the screened core')
    original = read_jsonl(FACTS)
    if (len(original) < 10 or sorted(f['fact_id'] for f in original) != protocol['selected_fact_ids']
            or any(f['split'] != 'dev' or f['review']['reviewer'] != 'Reviewer A'
                   or not f['review']['pre_release_verified'] for f in original)):
        raise ValueError('Invalid personally reviewed English development cohort')
    runner = load_runner(SNAPSHOT, config)
    if runner.identity['revision'] != protocol['model_revision']:
        raise ValueError('Aya model identity differs')
    if args.mode == 'screen':
        path = screen(runner, original, templates, output, seed=17)
        print(json.dumps({'mode': 'screen', 'run_path': path}), flush=True)
        return
    if not SCREENED.is_file():
        raise FileNotFoundError('The English screen has not completed')
    facts = read_jsonl(SCREENED)
    if (sorted(f['fact_id'] for f in facts) != protocol['selected_fact_ids'] or
            any(f['screen_model_hash'] != digest(runner.identity) for f in facts)):
        raise ValueError('Screened facts or model differ')
    eligible = [f for f in facts if f['english_eligible'] and f['core_member']]
    if len(eligible) < 10 or any(len({f['object_qid'] for f in eligible
                                     if f['relation'] == relation}) < 2
                                 for relation in ('P19', 'P20', 'P159', 'P740')):
        raise ValueError('English E4 development eligibility gate failed')
    result = run_experiment(runner, facts, templates, config,
                            {'protocol_hash': sha(PROTOCOL), 'fixture': False},
                            'E4', 'dev', output, shard=args.shard, num_shards=args.num_shards)
    print(json.dumps({'mode': 'e4', **result}), flush=True)


if __name__ == '__main__':
    main()

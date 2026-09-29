"""Independent-of-entity output-script diagnostic for held-out E2 generations.

This measures Unicode letter script, not grammatical language correctness.
It must not be substituted for a native-speaker output audit.
"""

from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/heldout'
ANALYSIS = BASE / 'results/analysis-aya-independent-test-v1.json'
OUT = ROOT / 'results/heldout-output-script-diagnostic-20260927.json'
AR_PREFIX = re.compile(r'^\s*(?:الإجابة|الجواب)\s*:\s*')


def script_counts(text):
    counts = Counter()
    for char in text:
        if not unicodedata.category(char).startswith('L'):
            continue
        name = unicodedata.name(char, '')
        if name.startswith('HEBREW'):
            counts['hebrew'] += 1
        elif name.startswith('ARABIC'):
            counts['arabic'] += 1
        elif name.startswith('LATIN'):
            counts['latin'] += 1
        else:
            counts['other'] += 1
    return counts


def dominant(text):
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return 'no_letters'
    script, count = counts.most_common(1)[0]
    return script if count / total >= 0.8 else 'mixed'


def main():
    analysis = json.loads(ANALYSIS.read_text(encoding='utf-8'))
    if analysis['data_kind'] != 'research' or analysis['model']['model_id'] != 'CohereLabs/aya-23-8B':
        raise ValueError('Pinned Aya research result required')
    tallies = Counter()
    examples = []
    keys = set()
    for run in analysis['run_paths']:
        if not run.startswith('results/e2-'):
            continue
        for path in (BASE / run / 'items').glob('*.json'):
            row = json.loads(path.read_text(encoding='utf-8'))
            key = row['key']
            if key['language'] not in ('he', 'ar'):
                raise ValueError('Unexpected language in E2')
            identity = (key['fact_id'], key['language'], key['template'], key['variant'])
            if identity in keys:
                raise ValueError('Duplicate E2 row')
            keys.add(identity)
            output = row['generation']['text']
            raw_script = dominant(output)
            stripped_script = dominant(AR_PREFIX.sub('', output)) if key['language'] == 'ar' else raw_script
            entity = bool(row['evaluation']['entity_correct'])
            tallies[(key['language'], key['variant'], 'entity_correct' if entity else 'entity_wrong',
                     'raw', raw_script)] += 1
            tallies[(key['language'], key['variant'], 'entity_correct' if entity else 'entity_wrong',
                     'arabic_prefix_stripped', stripped_script)] += 1
            wanted = {'he': 'hebrew', 'ar': 'arabic'}[key['language']]
            if not entity and stripped_script == wanted and len(examples) < 12:
                examples.append({'key': key, 'output': output, 'classification': stripped_script})
    if len(keys) != 480:
        raise ValueError('Incomplete held-out E2 coverage')
    groups = []
    for language in ('he', 'ar'):
        for variant in ('U', 'D'):
            for entity in ('entity_correct', 'entity_wrong'):
                for condition in ('raw', 'arabic_prefix_stripped'):
                    counts = {script: count for (la, va, en, co, script), count in tallies.items()
                              if (la, va, en, co) == (language, variant, entity, condition)}
                    groups.append({'language': language, 'variant': variant,
                                   'entity_status': entity, 'text_condition': condition,
                                   'n': sum(counts.values()), 'dominant_script_counts': counts,
                                   'requested_script_count': counts.get({'he': 'hebrew', 'ar': 'arabic'}[language], 0)})
    result = {'scope': 'Script diagnostic only; not a native-speaker language judgment and not a replacement score',
              'e2_rows': len(keys), 'method': 'At least 80% of Unicode letter codepoints belong to one script; strip only exact Arabic answer prefix for second view',
              'groups': groups, 'wrong_entity_requested_script_examples': examples,
              'warning': 'A response can use the requested script but the wrong language or entity; Arabic answer prefixes can dominate a short Latin-script answer. Machine requested_language_correct is joint with entity correctness.'}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'e2_rows': len(keys)}, indent=2))


if __name__ == '__main__':
    main()

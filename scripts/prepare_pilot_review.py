"""Prepare an outcome-independent review batch from pinned pilot subjects."""
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fragmented_facts.data import apply_split_manifest, csv_rows, write_csv
from fragmented_facts.io import digest, file_hash, read_json, write_json

out = ROOT / 'data/review/pilot_batch_v1'
if out.exists():
    raise FileExistsError('Pilot review batch exists; preserve reviewer work')
pool = apply_split_manifest(ROOT / 'data/raw/candidates_final.jsonl', ROOT / 'data/review/subject_splits.json')
relations = ['P19', 'P20', 'P159', 'P740']
groups = {r: sorted([f for f in pool if f['split'] == 'pilot' and f['relation'] == r
                    and f['wikidata']['single_answer_agrees'] is True],
                   key=lambda f: digest([17, 'pilot-review-order-v1', f['subject_qid']])) for r in relations}
selected = []
while len(selected) < 70 and any(groups.values()):
    for relation in relations:
        if groups[relation] and len(selected) < 70:
            selected.append(groups[relation].pop(0))
assert len(selected) == 70 and len({f['subject_qid'] for f in selected}) == 70
facts_by_id = {r['fact_id']: r for r in csv_rows(ROOT / 'data/review/facts.csv')}
pairs_by_id = {(r['fact_id'], r['language']): r for r in csv_rows(ROOT / 'data/review/pairs.csv')}
facts = [facts_by_id[f['fact_id']] for f in selected]
pairs = [pairs_by_id[f['fact_id'], lang] for f in selected for lang in ('he', 'ar')]
assert all(r['decision'] == 'pending' for r in facts + pairs), 'Existing human work requires a reviewed-copy workflow'
out.mkdir(parents=True)
write_csv(out / 'facts.csv', facts, list(facts[0]))
write_csv(out / 'pairs.csv', pairs, list(pairs[0]))
templates = read_json(ROOT / 'configs/templates.json')
template_rows = []
for stage, by_relation in templates.items():
    if stage not in ('evaluation', 'screen'):
        continue
    for relation, languages in by_relation.items():
        for language, items in languages.items():
            for template_id, item in items.items():
                template_rows.append({'key': '/'.join([stage, relation, language, template_id]),
                                      'stage': stage, 'relation': relation, 'language': language,
                                      'template_id': template_id, **item})
smoke = read_json(ROOT / 'results/aya-smoke-nf4-890456.json')
manifest = {'version': 1, 'seed': 17, 'source_sha256': file_hash(ROOT / 'data/raw/candidates_final.jsonl'),
            'split_manifest_sha256': file_hash(ROOT / 'data/review/subject_splits.json'),
            'selection': 'Round-robin relations within fixed pilot split; source agreement true; hash order; no model outcomes',
            'primary_subjects': 50, 'reserve_subjects': 20,
            'facts': [{'fact_id': f['fact_id'], 'subject_qid': f['subject_qid'],
                       'relation': f['relation'], 'priority': 'Primary' if i < 50 else 'Reserve'}
                      for i, f in enumerate(selected)],
            'template_keys': [x['key'] for x in template_rows],
            'smoke_report_sha256': file_hash(ROOT / 'results/aya-smoke-nf4-890456.json')}
manifest['manifest_hash'] = digest(manifest)
write_json(out / 'manifest.json', manifest)
payload = {'manifest': manifest, 'facts': facts, 'pairs': pairs, 'templates': template_rows,
           'smoke': smoke['examples'], 'original_facts': selected,
           'relation_names': {'P19': 'Place of birth', 'P20': 'Place of death',
                              'P159': 'Headquarters location', 'P740': 'Location of formation'}}
write_json(out / 'workbook_input.json', payload)
print(json.dumps({'batch': str(out), 'primary': 50, 'reserves': 20,
                  'primary_relations': dict(collections.Counter(f['relation'] for f in selected[:50])),
                  'templates': len(template_rows), 'pair_rows': len(pairs)}, indent=2))

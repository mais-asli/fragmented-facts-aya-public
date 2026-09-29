"""Consolidate the per-prompt Aya records of the held-out, E4 and external runs into paper_v2/_work/*.jsonl.

Run from anywhere:  python "nlp final/paper_v2/scripts/consolidate_v2.py"
Reads <nlp final>/results/<run dirs>/.../items/*.json (unchanged originals) and keeps only the
fields the paper uses. The pilot records (_work/e1.jsonl ... screen.jsonl) are copied from
paper_refined/_work, which were produced by paper_refined/scripts/consolidate_records.py.
"""
import glob, json, os, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('FF_ROOT', HERE.parent.parent))
OUT = HERE.parent / '_work'; OUT.mkdir(exist_ok=True)
GROUPS = {
    'ho_screen': 'heldout/results/screen-*/*/items/*.json', 'ho_e1': 'heldout/results/e1-*/*/items/*.json',
    'ho_e2': 'heldout/results/e2-*/*/items/*.json', 'ho_e3': 'h27/results/e3-*/*/items/*.json',
    'ho_e5': 'h27/results/e5-*/*/items/*.json', 'ho_e3b': 'b27/results/*/*/items/*.json',
    'ho_e3p': 'e3p27/results/*/*/items/*.json', 'e4c': 'e4c27/results/*/*/items/*.json',
    'e4dev': 'e427/results/*/*/items/*.json', 'gre_screen': 'gre27/results/screen-*/*/items/*.json',
    'gre_e1': 'gre27/results/e1-*/*/items/*.json', 'gre_e2': 'gre27/results/e2-*/*/items/*.json'}
def compact(d, f):
    p = d.get('prompt') or {}
    r = {'key': d.get('key'), 'run': Path(f).parts[-4] if len(Path(f).parts) >= 4 else '', 'relation': (d.get('fact') or {}).get('relation'),
         'subject': p.get('subject'), 'subject_tokens': p.get('subject_tokens'), 'base_characters': p.get('base_characters'),
         'sitelinks': (d.get('popularity') or {}).get('sitelinks')}
    g = d.get('gold_score')
    if g: r.update(gold_sum=g.get('sum_logprob'), gold_answer=g.get('answer'), answer_tokens=g.get('answer_tokens'))
    gen = d.get('generation')
    if gen: r.update(gen=gen.get('text'), truncated=gen.get('truncated'), gen_mean_lp=gen.get('answer_mean_logprob'))
    ev = d.get('evaluation')
    if ev: r.update(correct=ev.get('entity_correct'), category=ev.get('category'), lang_ok=ev.get('requested_language_correct'))
    ea = d.get('evaluation_all_aliases')
    if ea: r['correct_all_alias'] = ea.get('entity_correct')
    if 'patch' in d and isinstance(d['patch'], dict): r['patch'] = {k: v for k, v in d['patch'].items() if not isinstance(v, (list, dict))}
    for k in ('method_status', 'donor_fact_id', 'corrupt_score', 'restored_score', 'restoration_gain', 'likelihood_margin'):
        if k in d: r[k] = d[k] if not isinstance(d[k], dict) else {kk: vv for kk, vv in d[k].items() if not isinstance(vv, list)}
    if isinstance(d.get('readout'), dict): r['readout'] = {kk: vv for kk, vv in d['readout'].items() if not isinstance(vv, list)}
    return r
if __name__ == '__main__':
    for name, pat in GROUPS.items():
        files = sorted(glob.glob(str(ROOT / 'results' / pat)))
        with open(OUT / f'{name}.jsonl', 'w', encoding='utf-8') as fh:
            for f in files:
                fh.write(json.dumps(compact(json.load(open(f, encoding='utf-8')), f), ensure_ascii=False) + '\n')
        print(name, len(files))
    for n in ('e1', 'e2', 'e3', 'e5a', 'e5b', 'screen', 'computed'):
        src = ROOT / 'paper_refined' / '_work' / (n + ('.json' if n == 'computed' else '.jsonl'))
        if src.exists(): shutil.copy2(src, OUT / src.name)

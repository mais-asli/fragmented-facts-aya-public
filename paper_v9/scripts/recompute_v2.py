"""Independent recomputation of the held-out, external, E3, E4 and E5 point estimates from per-prompt records."""
import json, collections, statistics, sys
from pathlib import Path
W = Path(__file__).resolve().parent.parent / '_work'
def load(n): return [json.loads(l) for l in open(W/f'{n}.jsonl', encoding='utf-8')]
def fid(r): return r['key']['fact_id']
def rel(r): return fid(r).split('-')[1]
def macro(per_subject):  # {fact_id: value}
    by = collections.defaultdict(list)
    for f, v in per_subject.items(): by[f.split('-')[1]].append(v)
    return sum(statistics.mean(v) for v in by.values()) / len(by), {k: len(v) for k, v in by.items()}
def subj_mean(rows, val):
    d = collections.defaultdict(list)
    for r in rows: d[fid(r)].append(val(r))
    return {k: statistics.mean(v) for k, v in d.items()}
out = {}
for coh, pre in (('heldout', 'ho'), ('external', 'gre')):
    e1 = load(f'{pre}_e1'); e2 = load(f'{pre}_e2'); sc = load(f'{pre}_screen')
    for lang in ('en', 'he', 'ar'):
        rows = [r for r in e1 if r['key']['language'] == lang]
        m, cnt = macro(subj_mean(rows, lambda r: float(bool(r['correct']))))
        out[f'{coh}/E1/{lang}'] = (round(m, 4), sum(bool(r['correct']) for r in rows), len(rows), cnt)
    # screen
    s = collections.defaultdict(list)
    for r in sc: s[fid(r)].append(bool(r['correct']))
    out[f'{coh}/screen'] = (sum(all(v) for v in s.values()), sum(any(v) for v in s.values()), len(s))
    for lang in ('he', 'ar'):
        pairs = collections.defaultdict(dict)
        for r in e2:
            if r['key']['language'] != lang: continue
            pairs[(fid(r), r['key']['template'])][r['key']['variant']] = r
        dacc = {}; dll = {}; tr = collections.Counter(); ut = []; dt = []
        per = collections.defaultdict(lambda: ([], []))
        for (f, t), p in pairs.items():
            u, d = p['U'], p['D']
            per[f][0].append(float(bool(d['correct'])) - float(bool(u['correct'])))
            per[f][1].append(d['gold_sum'] - u['gold_sum'])
            tr[f"{int(bool(u['correct']))}->{int(bool(d['correct']))}"] += 1
            ut.append(u['subject_tokens']); dt.append(d['subject_tokens'])
        a, _ = macro({f: statistics.mean(v[0]) for f, v in per.items()})
        l, _ = macro({f: statistics.mean(v[1]) for f, v in per.items()})
        out[f'{coh}/E2/{lang}'] = dict(n_pairs=len(pairs), dacc=round(a, 4), dS=round(l, 3), transitions=dict(tr),
                                      tok_U=round(statistics.mean(ut), 2), tok_D=round(statistics.mean(dt), 2),
                                      all_increase=all(d > u for u, d in zip(ut, dt)), min_add=min(d-u for u, d in zip(ut, dt)), max_add=max(d-u for u, d in zip(ut, dt)))
# E5 held-out
e5 = load('ho_e5')
base = {}
for r in e5:
    k = r['key']
    if k['variant'] in ('baseline_D', 'baseline_U', 'D', 'U', 'unpatched_D', 'unpatched_U'):
        base[(fid(r), k['language'], k['variant'])] = r
print('E5 variants', collections.Counter(r['key']['variant'] for r in e5))
print('E5 windows', collections.Counter(r['key'].get('window') for r in e5))
json.dump(out, sys.stdout, ensure_ascii=False, indent=1)

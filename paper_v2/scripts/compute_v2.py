"""Compute every new-cohort number used in the v2 paper from per-prompt records (paper_v2/_work)
and write _work/computed_v2.json. Bootstrap = project analysis.cluster_effect (seed 17, 10,000)."""
import json, collections, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stats_util import cluster_effect
W = HERE.parent / '_work'
def load(n): return [json.loads(l) for l in open(W / f'{n}.jsonl', encoding='utf-8')]
def fid(r): return r['key']['fact_id']
def signflip(values, subjects, relations, repeats=10000, seed=17):
    cells = collections.defaultdict(list)
    for v, s, r in zip(values, subjects, relations): cells[s, r].append(v)
    rn = collections.Counter(r for s, r in cells); w = collections.defaultdict(float)
    for (s, r), vs in cells.items(): w[s] += float(np.mean(vs)) / rn[r] / len(rn)
    v = np.array(list(w.values())); obs = abs(float(v.sum())); rng = np.random.default_rng(seed)
    ex = sum(abs(float(np.dot(v, rng.choice([-1, 1], size=len(v))))) >= obs - 1e-12 for _ in range(repeats))
    return (ex + 1) / (repeats + 1)
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None]*len(ps); prev = 0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, ps[i]*(len(ps)-rank))); out[i] = prev
    return out
def eff(values, facts):
    p, ci, nb, n = cluster_effect(values, [f.split('-')[0] for f in facts], [f.split('-')[1] for f in facts])
    return {'estimate': p, 'ci95': ci, 'n_subjects': n, 'replicates': nb}
C = {}
for coh, pre in (('heldout', 'ho'), ('external', 'gre')):
    e2 = load(f'{pre}_e2'); e1 = load(f'{pre}_e1')
    for lang in ('en', 'he', 'ar'):
        rows = [r for r in e1 if r['key']['language'] == lang]
        C[f'{coh}_E1_{lang}_U'] = eff([float(bool(r['correct'])) for r in rows], [fid(r) for r in rows])
    ps_acc, ps_ll = [], []
    for lang in ('he', 'ar'):
        pairs = collections.defaultdict(dict)
        for r in e2:
            if r['key']['language'] == lang: pairs[(fid(r), r['key']['template'])][r['key']['variant']] = r
        keys = sorted(pairs); facts = [k[0] for k in keys]
        U = [float(bool(pairs[k]['U']['correct'])) for k in keys]; D = [float(bool(pairs[k]['D']['correct'])) for k in keys]
        da = [d - u for u, d in zip(U, D)]; dl = [pairs[k]['D']['gold_sum'] - pairs[k]['U']['gold_sum'] for k in keys]
        C[f'{coh}_E2_{lang}_U'] = eff(U, facts); C[f'{coh}_E2_{lang}_D'] = eff(D, facts)
        a = eff(da, facts); l = eff(dl, facts)
        a['p'] = signflip(da, [f.split('-')[0] for f in facts], [f.split('-')[1] for f in facts])
        l['p'] = signflip(dl, [f.split('-')[0] for f in facts], [f.split('-')[1] for f in facts])
        C[f'{coh}_E2_{lang}_delta_acc'] = a; C[f'{coh}_E2_{lang}_delta_ll'] = l
        tr = collections.Counter(f"{int(u)}->{int(d)}" for u, d in zip(U, D)); C[f'{coh}_E2_{lang}_transitions'] = dict(tr)
        C[f'{coh}_E2_{lang}_correct_prompts'] = [int(sum(U)), int(sum(D)), len(keys)]
        ps_acc.append(a['p']); ps_ll.append(l['p'])
    for lang, h in zip(('he', 'ar'), holm(ps_acc)): C[f'{coh}_E2_{lang}_delta_acc']['holm_p'] = h
    for lang, h in zip(('he', 'ar'), holm(ps_ll)): C[f'{coh}_E2_{lang}_delta_ll']['holm_p'] = h
    sc = load(f'{pre}_screen'); s = collections.defaultdict(list)
    for r in sc: s[fid(r)].append(bool(r['correct']))
    C[f'{coh}_screen'] = {'both': sum(all(v) for v in s.values()), 'any': sum(any(v) for v in s.values()), 'n': len(s)}
json.dump(C, open(W / 'computed_v2.json', 'w'), indent=1)
for k, v in C.items():
    if isinstance(v, dict) and 'estimate' in v:
        print(k, round(v['estimate'], 4), [round(x, 4) for x in v['ci95']], {kk: round(vv, 4) for kk, vv in v.items() if kk in ('p', 'holm_p')})
    else: print(k, v)

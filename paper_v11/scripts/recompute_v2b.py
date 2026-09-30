"""Recompute held-out/external E2 robustness rows and verify E3/E4/E5 point estimates from per-prompt records.
Bootstrap replicates the project's analysis.cluster_effect exactly (seed 17, 10,000 subject resamples)."""
import json, collections
import numpy as np
from pathlib import Path
W = Path(__file__).resolve().parent.parent / '_work'
def load(n): return [json.loads(l) for l in open(W/f'{n}.jsonl', encoding='utf-8')]
def fid(r): return r['key']['fact_id']
def cluster_effect(values, subjects, relations, repeats=10000, seed=17):
    ids = sorted(set(subjects)); rels = sorted(set(relations))
    cells = collections.defaultdict(list)
    for v, s, r in zip(values, subjects, relations): cells[s, r].append(v)
    m = np.full((len(ids), len(rels)), np.nan)
    for i, s in enumerate(ids):
        for j, r in enumerate(rels):
            if (s, r) in cells: m[i, j] = np.mean(cells[s, r])
    point = float(np.nanmean(np.nanmean(m, axis=0)))
    rng = np.random.default_rng(seed); boot = []
    for _ in range(repeats):
        smp = m[rng.integers(0, len(ids), size=len(ids))]; c = np.isfinite(smp).sum(axis=0)
        if np.all(c > 0): boot.append(float(np.mean(np.nansum(smp, axis=0) / c)))
    return point, np.quantile(boot, [0.025, 0.975]).tolist(), len(boot), len(ids)
def e2_effect(rows, lang, keep=lambda f, t: True):
    pairs = collections.defaultdict(dict)
    for r in rows:
        k = r['key']
        if k['language'] == lang and keep(fid(r), k['template']): pairs[(fid(r), k['template'])][k['variant']] = r
    va, vl, s, rl = [], [], [], []
    for (f, t), p in sorted(pairs.items()):
        va.append(float(bool(p['D']['correct'])) - float(bool(p['U']['correct'])))
        vl.append(p['D']['gold_sum'] - p['U']['gold_sum']); s.append(f.split('-')[0]); rl.append(f.split('-')[1])
    return cluster_effect(va, s, rl), cluster_effect(vl, s, rl)
out = {}
for coh, pre in (('heldout', 'ho'), ('external', 'gre')):
    e2 = load(f'{pre}_e2'); sc = load(f'{pre}_screen')
    scr = collections.defaultdict(list)
    for r in sc: scr[fid(r)].append(bool(r['correct']))
    both = {f for f, v in scr.items() if all(v)}; one = {f for f, v in scr.items() if any(v)}
    for lang in ('he', 'ar'):
        rows = {'primary': lambda f, t: True, 't1': lambda f, t: t == 't1', 't2': lambda f, t: t == 't2', 't3': lambda f, t: t == 't3',
                f'recall2of2_n{len(both)}': lambda f, t: f in both, f'recall1of2_n{len(one)}': lambda f, t: f in one}
        for name, keep in rows.items():
            try:
                a, l = e2_effect(e2, lang, keep)
                out[f'{coh}/{lang}/{name}'] = {'acc': [round(a[0]*100, 2)] + [round(x*100, 2) for x in a[1]] + [a[2], a[3]],
                                               'll': [round(l[0], 3)] + [round(x, 3) for x in l[1]] + [l[2]]}
            except Exception as ex:
                out[f'{coh}/{lang}/{name}'] = str(ex)
# E5 held-out verification: same-entity etc. minus baseline_D (reverse vs baseline_U), per window
e5 = load('ho_e5'); base = {}
for r in e5:
    k = r['key']
    if k['variant'] in ('baseline_D', 'baseline_U'): base[(fid(r), k['language'], k['variant'])] = r
for lang in ('he', 'ar'):
    for w in ('early', 'lower_middle', 'upper_middle', 'late'):
        for cond in ('same_entity', 'unrelated', 'identity', 'first_subject', 'delimiter', 'reverse'):
            v, s, rl = [], [], []
            for r in e5:
                k = r['key']
                if k['language'] == lang and k.get('window') == w and k['variant'] == cond:
                    b = base[(fid(r), lang, 'baseline_U' if cond == 'reverse' else 'baseline_D')]
                    v.append(r['gold_sum'] - b['gold_sum']); s.append(fid(r).split('-')[0]); rl.append(fid(r).split('-')[1])
            p, ci, nb, n = cluster_effect(v, s, rl)
            out[f'E5/{lang}/{w}/{cond}'] = [round(p, 3)] + [round(x, 3) for x in ci] + [n]
    lp = [r for r in e5 if r['key']['language'] == lang and r['key']['variant'] == 'late_prediction']
    if lp:
        v = [r['gold_sum'] - base[(fid(r), lang, 'baseline_D')]['gold_sum'] for r in lp]
        p, ci, nb, n = cluster_effect(v, [fid(r).split('-')[0] for r in lp], [fid(r).split('-')[1] for r in lp])
        out[f'E5/{lang}/late_prediction'] = [round(p, 3)] + [round(x, 3) for x in ci] + [n]
    bD = [r for (f, l, vv), r in base.items() if l == lang and vv == 'baseline_D']; bU = [r for (f, l, vv), r in base.items() if l == lang and vv == 'baseline_U']
    out[f'E5/{lang}/baseline_correct_D_U'] = [sum(bool(r['correct']) for r in bD), sum(bool(r['correct']) for r in bU), len(bD)]
# E5 baselines vs E2 t1
e2 = load('ho_e2'); e2k = {(fid(r), r['key']['language'], r['key']['variant']): r for r in e2 if r['key']['template'] == 't1'}
diff = 0; maxd = 0.0
for (f, l, vv), r in base.items():
    o = e2k[(f, l, vv[-1])]
    maxd = max(maxd, abs(o['gold_sum'] - r['gold_sum'])); diff += (o['gen'] != r['gen']) + (bool(o['correct']) != bool(r['correct']))
out['E5_vs_E2_t1'] = {'n': len(base), 'gen_or_correct_diffs': diff, 'max_abs_gold_diff': maxd}
# E4 cells
e4 = load('e4c')
cells = collections.defaultdict(list)
for r in e4:
    k = r['key']; cells[(r['run'], k['site'], k['layer'])].append((fid(r), r['restoration_gain']))
for (run, site, layer), v in sorted(cells.items()):
    if layer == 12 or (site in ('prediction', 'last_subject') and layer == 24):
        vals = [x[1] for x in v]
        out[f'E4/{run[:12]}/{site}/{layer}'] = [round(float(np.mean(vals)), 3), len(vals), sum(x > 0 for x in vals)]
print(json.dumps(out, indent=0, ensure_ascii=False))

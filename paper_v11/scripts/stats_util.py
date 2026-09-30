import collections
import numpy as np
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

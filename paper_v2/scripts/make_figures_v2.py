"""Figures for the v2 paper. Reads result JSONs from <nlp final>/results and records from paper_v2/_work.
Set FF_ROOT to the project root if this folder is not <nlp final>/paper_v2."""
import json, os, collections, numpy as np, matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('FF_ROOT', HERE.parent.parent))
R = ROOT / 'results'; W = HERE.parent / '_work'; OUT = HERE.parent / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
  'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.8, 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
  'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e',
  'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b', 'text.color': '#0b0b0b'})
HE = '#2a78d6'; AR = '#eb6834'; AQ = '#1baf7a'; VI = '#4a3aa7'; GRID = '#e6e5e1'; MUTED = '#8a8984'; INK2 = '#52514e'
J = lambda p: json.load(open(p, encoding='utf-8'))
cp = J(W / 'computed.json'); c2 = J(W / 'computed_v2.json')

# ---------------- Figure 1 (right): U -> D accuracy dumbbells per cohort and language
rows = [('Pilot', 'he', cp['E2_he_U'], cp['E2_he_D']), ('Pilot', 'ar', cp['E2_ar_U'], cp['E2_ar_D']),
        ('Held-out', 'he', c2['heldout_E2_he_U'], c2['heldout_E2_he_D']), ('Held-out', 'ar', c2['heldout_E2_ar_U'], c2['heldout_E2_ar_D']),
        ('External', 'he', c2['external_E2_he_U'], c2['external_E2_he_D']), ('External', 'ar', c2['external_E2_ar_U'], c2['external_E2_ar_D'])]
eng = {'Pilot': cp['E1_en_U']['estimate'], 'Held-out': c2['heldout_E1_en_U']['estimate'], 'External': c2['external_E1_en_U']['estimate']}
fig, ax = plt.subplots(figsize=(2.62, 2.05))
ys = [5.2, 4.5, 3.0, 2.3, 0.8, 0.1]
for y, (coh, lang, u, d) in zip(ys, rows):
    col = HE if lang == 'he' else AR; mk = 'o' if lang == 'he' else 's'
    uu, dd = 100*u['estimate'], 100*d['estimate']
    ax.plot([dd, uu], [y, y], color=col, lw=1.6, solid_capstyle='round', zorder=2, alpha=0.55)
    ax.plot(uu, y, marker=mk, ms=5.6, color=col, mec='white', mew=0.6, zorder=3, ls='none')
    ax.plot(dd, y, marker=mk, ms=5.2, mfc='white', mec=col, mew=1.2, zorder=3, ls='none')
    ax.text(max(uu, dd) + 2.2, y, f'{uu:.1f}$\\rightarrow${dd:.1f}', va='center', fontsize=7.2, color='#0b0b0b')
for yc, coh, n in [(4.85, 'Pilot', 57), (2.65, 'Held-out', 40), (0.45, 'External', 26)]:
    ax.text(-1.5, yc + 0.78, f'{coh} ($n$={n}); English {100*eng[coh]:.1f}%', fontsize=7.4, color=INK2, va='bottom')
ax.set_xlim(-2, 50); ax.set_ylim(-0.45, 6.15); ax.set_yticks([])
ax.set_xlabel('Complete-answer accuracy (%)')
ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
h = [Line2D([], [], color=HE, marker='o', ls='none', ms=5, label='Hebrew'), Line2D([], [], color=AR, marker='s', ls='none', ms=5, label='Arabic'),
     Line2D([], [], color=INK2, marker='o', ls='none', ms=5, label='unmarked'), Line2D([], [], color=INK2, marker='o', mfc='white', ls='none', ms=5, label='marked')]
ax.legend(handles=h, loc='lower center', frameon=False, fontsize=7.0, ncol=4, handletextpad=0.1, columnspacing=0.7, bbox_to_anchor=(0.47, 1.0))
fig.savefig(OUT / 'fig1_accuracy.pdf', bbox_inches='tight', pad_inches=0.02)

# ---------------- Figure 2: forest of D - U effects across cohorts and sensitivity analyses
pil = J(R / 'tau_research_20260926/results/analysis-reviewed-pilot-20260926.json')
scr = J(R / 'english-screen-subgroups-20260926.json'); fmt = J(R / 'answer-format-sensitivity-pilot-20260926.json')
fmt_h = J(R / 'answer-format-sensitivity-heldout-20260927.json'); fmt_e = J(R / 'answer-format-sensitivity-google-re-external-20260927.json')
hq = J(R / 'heldout-strict-hq-sensitivity-20260927.json'); cue = J(R / 'answer-cue-exclusion-sensitivity-20260927.json')
sys_path = str(HERE); import sys; sys.path.insert(0, sys_path)
from stats_util import cluster_effect
def load(n): return [json.loads(l) for l in open(W / f'{n}.jsonl', encoding='utf-8')]
def e2_subset(pre, lang, keep):
    rows = load(f'{pre}_e2'); pairs = collections.defaultdict(dict)
    for r in rows:
        k = r['key']
        if k['language'] == lang and keep(k['fact_id']): pairs[(k['fact_id'], k['template'])][k['variant']] = r
    keys = sorted(pairs); f = [k[0] for k in keys]
    va = [float(bool(pairs[k]['D']['correct'])) - float(bool(pairs[k]['U']['correct'])) for k in keys]
    vl = [pairs[k]['D']['gold_sum'] - pairs[k]['U']['gold_sum'] for k in keys]
    a = cluster_effect(va, [x.split('-')[0] for x in f], [x.split('-')[1] for x in f])
    l = cluster_effect(vl, [x.split('-')[0] for x in f], [x.split('-')[1] for x in f])
    return {'estimate': a[0], 'ci95': a[1]}, {'estimate': l[0], 'ci95': l[1]}
sc = collections.defaultdict(list)
for r in load('ho_screen'): sc[r['key']['fact_id']].append(bool(r['correct']))
core = {f for f, v in sc.items() if all(v)}
ho_core = {l: e2_subset('ho', l, lambda f: f in core) for l in ('he', 'ar')}
def po(l): return [o for o in pil['orthography'] if o['language'] == l][0]
def sg(l): return [o for o in [s for s in scr['subgroups'] if s['name'] == 'strict_two_of_two'][0]['orthography'] if o['language'] == l][0]
def fm(d, l, key='prefix_exact'): return [o for o in d['orthography'] if o['language'] == l][0][key]['effect']
def hql(l): return [o for o in hq['languages'] if o['language'] == l][0]
def cuel(split, l):
    c = [c for c in cue['cohorts'] if c['split'] == split][0]; x = [o for o in c['languages'] if o['language'] == l][0]['without_answer_cues']
    return x['accuracy_D_minus_U'], x['gold_sum_loglik_D_minus_U']
def c2e(coh, l, what): return c2[f'{coh}_E2_{l}_delta_{what}']
F = [('Pilot', 'All 57 subjects (primary)', lambda l: po(l)['accuracy'], lambda l: po(l)['log_likelihood'], True),
     ('Pilot', 'English-recallable ($n$=17)', lambda l: sg(l)['accuracy'], lambda l: sg(l)['log_likelihood'], False),
     ('Pilot', 'Answer-format rule (post hoc)', lambda l: fm(fmt, l, 'with_he_terminal'), None, False),
     ('Held-out', 'All 40 subjects (primary)', lambda l: c2e('heldout', l, 'acc'), lambda l: c2e('heldout', l, 'll'), True),
     ('Held-out', 'English-recallable ($n$=16)', lambda l: ho_core[l][0], lambda l: ho_core[l][1], False),
     ('Held-out', 'Answer-prefix rule (fixed before test)', lambda l: fm(fmt_h, l), None, False),
     ('Held-out', 'Strict headquarters source ($n$=39)', lambda l: hql(l)['accuracy'], lambda l: hql(l)['log_likelihood'], False),
     ('Held-out', 'No answer cue in name ($n$=39)', lambda l: cuel('test', l)[0], lambda l: cuel('test', l)[1], False),
     ('External', 'All 26 subjects (primary)', lambda l: c2e('external', l, 'acc'), lambda l: c2e('external', l, 'll'), True),
     ('External', 'Answer-prefix rule (post hoc)', lambda l: fm(fmt_e, l), None, False)]
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True, gridspec_kw={'wspace': 0.05})
ys = []; y = 0; prev = None; seps = []
for coh, *_ in F[::-1]:
    if prev is not None and coh != prev: y += 0.55; seps.append(y - 0.78)
    ys.append(y); y += 1; prev = coh
ys = ys[::-1]
forest = []
for ax, unit in zip(axes, ['acc', 'll']):
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    for s in seps: ax.axhline(s, color=GRID, lw=0.8)
    for (coh, lab, fa, fl, primary), yy in zip(F, ys):
        f = fa if unit == 'acc' else fl
        if f is None: continue
        for lang, col, mk, off in [('he', HE, 'o', 0.17), ('ar', AR, 's', -0.17)]:
            e = f(lang); k = 100 if unit == 'acc' else 1
            est = e['estimate']*k; lo, hi = [v*k for v in e['ci95']]
            forest.append((coh, lab, unit, lang, round(est, 3), round(lo, 3), round(hi, 3)))
            ax.plot([lo, hi], [yy+off, yy+off], color=col, lw=1.4 if primary else 1.0, solid_capstyle='butt', zorder=2)
            ax.plot(est, yy+off, marker=mk, ms=5.2 if primary else 4.3, mfc=col if primary else 'white', mec=col, mew=1.1, zorder=3, ls='none')
    ax.set_xlabel('Marked $-$ unmarked accuracy (points)' if unit == 'acc' else 'Marked $-$ unmarked answer log-likelihood (nats)')
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
axes[0].set_yticks(ys); axes[0].set_yticklabels([f'{c}: {l}' if p else l for c, l, _, _, p in F])
for t, (_, _, _, _, p) in zip(axes[0].get_yticklabels(), F):
    if p: t.set_fontweight('bold')
axes[0].set_xlim(-45, 20); axes[1].set_xlim(-3.6, 1.2)
axes[1].text(0.5, ys[2], 'Scoring rules change accuracy only;\nthe likelihood target is fixed.', transform=axes[1].get_yaxis_transform(), ha='center', va='center', color=MUTED, fontsize=7.2, style='italic')
h = [Line2D([], [], color=HE, marker='o', ls='-', lw=1.2, ms=5, label='Hebrew'), Line2D([], [], color=AR, marker='s', ls='-', lw=1.2, ms=5, label='Arabic')]
fig.legend(handles=h, loc='upper center', frameon=False, ncol=2, handlelength=1.6, bbox_to_anchor=(0.62, 1.04))
axes[0].set_title('(a) Complete-answer accuracy', loc='left', fontsize=8.8, pad=4)
axes[1].set_title('(b) Canonical-answer log-likelihood', loc='left', fontsize=8.8, pad=4)
fig.savefig(OUT / 'forest_effects.pdf', bbox_inches='tight', pad_inches=0.02)
json.dump(forest, open(W / 'forest_v2.json', 'w'), indent=0)

# ---------------- Figure 3: mechanism. (a,b) E5 by window, pilot vs held-out; (c) English E4 restoration by layer
mech = J(R / 'tau_mechanism_20260926/results/analysis-reviewed-pilot-with-mechanism-20260926.json')
ho5 = J(R / 'h27/results/analysis-e5-aya-heldout-strengthening-20260927.json')
e4 = J(R / 'e4-reviewed-cohorts-analysis-20260927.json')
wins = ['early', 'lower_middle', 'upper_middle', 'late']; wl = ['4–5', '12–13', '20–21', '28–29']
conds = [('same_entity', 'Same entity (U→D)', HE), ('unrelated', 'Unrelated donor', AR), ('reverse', 'Reverse (D→U)', AQ)]
fig = plt.figure(figsize=(6.3, 2.08))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.12], wspace=0.28)
axs = [fig.add_subplot(gs[0, 0])]; axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0])); axs.append(fig.add_subplot(gs[0, 2]))
e5out = []
for ax, lang, title in zip(axs[:2], ['he', 'ar'], ['(a) E5, Hebrew', '(b) E5, Arabic']):
    ax.axhline(0, color=INK2, lw=0.7, zorder=1)
    ax.axvspan(0.62, 1.38, color='#f4f3f0', zorder=0, lw=0)
    x = np.arange(4)
    for ci, (c, lab, col) in enumerate(conds):
        for src, name, off, filled, ls in [(mech, 'pilot', -0.13, True, '-'), (ho5, 'held-out', 0.13, False, (0, (2.5, 1.5)))]:
            est, lo, hi = [], [], []
            for w in wins:
                r = [p for p in src['patch_controls'] if p['language'] == lang and p['window'] == w and p['condition'] == c][0]['score_gain']
                est.append(r['estimate']); lo.append(r['ci95'][0]); hi.append(r['ci95'][1])
                e5out.append((name, lang, w, c, round(r['estimate'], 3), round(r['ci95'][0], 3), round(r['ci95'][1], 3)))
            est, lo, hi = map(np.array, (est, lo, hi))
            xx = x + off + (ci - 1)*0.035
            ax.errorbar(xx, est, yerr=[est-lo, hi-est], fmt='o' if filled else 's', color=col, mfc=col if filled else 'white', mec=col,
                        ms=3.9, elinewidth=0.9, capsize=0, lw=0, zorder=3)
            ax.plot(xx, est, color=col, lw=0.9, ls=ls, alpha=0.7, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(wl); ax.set_xlabel('Patched layers')
    ax.set_title(title, loc='left', fontsize=8.8)
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
axs[0].set_ylabel('Change in answer log-lik. (nats)')
plt.setp(axs[1].get_yticklabels(), visible=False)
ax = axs[2]
sites = [('last_subject', 'Last subject token', HE), ('prediction', 'Prediction position', VI), ('first_subject', 'First subject token', MUTED)]
for site, lab, col in sites:
    for coh, filled, ls in [('pilot', True, '-'), ('test', False, (0, (2.5, 1.5)))]:
        cells = sorted([c for c in e4['cohorts'][coh]['cells'] if c['site'] == site], key=lambda c: c['layer'])
        L = [c['layer'] for c in cells]; m = np.array([c['gain_mean_nats'] for c in cells])
        lo = np.array([c['gain_subject_bootstrap_95_interval'][0] for c in cells]); hi = np.array([c['gain_subject_bootstrap_95_interval'][1] for c in cells])
        ax.fill_between(L, lo, hi, color=col, alpha=0.10 if filled else 0.07, lw=0)
        ax.plot(L, m, color=col, lw=1.2, ls=ls, marker='o' if filled else 's', ms=3.2, mfc=col if filled else 'white', mec=col)
ax.axvspan(11.3, 13.7, color='#f4f3f0', zorder=0, lw=0)
ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28]); ax.set_xlabel('Restored layer (English, E4)')
ax.set_ylabel('Restoration gain (nats)'); ax.set_title('(c) E4, English', loc='left', fontsize=8.8)
ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
h1 = [Line2D([], [], color=col, marker='o', lw=1.0, ms=4, label=lab) for c, lab, col in conds]
h2 = [Line2D([], [], color=col, lw=1.2, label=lab + ' (E4)') for s, lab, col in sites]
h3 = [Line2D([], [], color=INK2, marker='o', lw=1.0, ms=4, label='pilot cohort'),
      Line2D([], [], color=INK2, marker='s', mfc='white', lw=1.0, ls=(0, (2.5, 1.5)), ms=4, label='held-out cohort')]
hh = [h1[0], h2[0], h1[1], h2[1], h1[2], h2[2], h3[0], h3[1]]
fig.legend(handles=hh, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.27), handletextpad=0.3, columnspacing=1.0, fontsize=7.4)
fig.savefig(OUT / 'mechanism.pdf', bbox_inches='tight', pad_inches=0.02)
json.dump(e5out, open(W / 'e5_v2.json', 'w'), indent=0)

# ---------------- Appendix: tokenization per subject, three cohorts
fig, ax = plt.subplots(figsize=(3.05, 2.4))
ax.plot([0, 56], [0, 56], color=MUTED, lw=0.7, zorder=1); ax.text(12.6, 9.2, 'D = U', color=MUTED, fontsize=7.5)
for pre, name, mk in [('e2', 'pilot', 'o'), ('ho_e2', 'held-out', 's'), ('gre_e2', 'external', '^')]:
    rows = load(pre)
    for lang, col in [('he', HE), ('ar', AR)]:
        d = collections.defaultdict(lambda: {'U': [], 'D': []})
        for r in rows:
            if r['key']['language'] == lang: d[r['key']['fact_id']][r['key']['variant']].append(r['subject_tokens'])
        u = np.array([np.mean(v['U']) for v in d.values()]); dd = np.array([np.mean(v['D']) for v in d.values()])
        ax.scatter(u, dd, s=11, marker=mk, facecolor=col if name == 'pilot' else 'white', edgecolor=col, linewidth=0.7, zorder=3, alpha=0.9)
ax.set_xlim(0, 16); ax.set_ylim(0, 56)
ax.set_xlabel('Subject tokens, unmarked (U)'); ax.set_ylabel('Subject tokens, marked (D)')
hh = [Line2D([], [], color=HE, marker='o', ls='none', ms=4, label='Hebrew'), Line2D([], [], color=AR, marker='o', ls='none', ms=4, label='Arabic'),
      Line2D([], [], color=INK2, marker='o', ls='none', ms=4, label='pilot'), Line2D([], [], color=INK2, marker='s', mfc='white', ls='none', ms=4, label='held-out'),
      Line2D([], [], color=INK2, marker='^', mfc='white', ls='none', ms=4, label='external')]
ax.legend(handles=hh, frameon=False, loc='upper left', fontsize=7, handletextpad=0.2, ncol=2)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.savefig(OUT / 'tokenization.pdf', bbox_inches='tight', pad_inches=0.02)
print('figures ok')

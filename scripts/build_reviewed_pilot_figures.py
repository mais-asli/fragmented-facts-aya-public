"""Build ACL-ready figures and tables from saved real Aya pilot analysis only."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/tau_research_20260926/results/analysis-reviewed-pilot-20260926.json'
OUT = ROOT / 'paper/generated'


def point_and_interval(ax, row, y, metric, scale, color):
    effect = row[metric]
    value = effect['estimate'] * scale
    interval = effect['ci95']
    if interval is not None:
        low, high = [v * scale for v in interval]
        ax.plot([low, high], [y, y], color=color, linewidth=2.2, solid_capstyle='round')
        ax.plot([low, low], [y - 0.07, y + 0.07], color=color, linewidth=1.1)
        ax.plot([high, high], [y - 0.07, y + 0.07], color=color, linewidth=1.1)
    ax.scatter([value], [y], color=color, s=38, zorder=3)


def main():
    report = json.loads(SOURCE.read_text(encoding='utf-8'))
    if report['data_kind'] != 'research' or report['model']['model_id'] != 'CohereLabs/aya-23-8B':
        raise ValueError('The input is not an actual Aya research analysis')
    ortho = {entry['language']: entry for entry in report['orthography']
             if entry['split'] == 'pilot'}
    if set(ortho) != {'he', 'ar'} or any(ortho[lang]['n_pairs'] != 171 for lang in ortho):
        raise ValueError('Expected complete 57-subject, three-template paired effects')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    figure, axes = plt.subplots(2, 1, figsize=(3.25, 4.0), constrained_layout=True)
    colors = {'he': '#315c9d', 'ar': '#bd5e39'}
    for ax, metric, scale, label in (
        (axes[0], 'accuracy', 100, 'Entity accuracy: D − U (points)'),
        (axes[1], 'log_likelihood', 1, 'Answer log likelihood: D − U (nats)'),
    ):
        for y, language in ((1, 'he'), (0, 'ar')):
            point_and_interval(ax, ortho[language], y, metric, scale, colors[language])
        ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--')
        ax.set_yticks([0, 1], ['Arabic', 'Hebrew'])
        ax.set_ylim(-0.4, 1.4)
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / 'paired_effects_pilot.pdf')
    figure.savefig(OUT / 'paired_effects_pilot.png', dpi=220)
    plt.close(figure)

    baseline = {entry['language']: entry for entry in report['baseline']
                if entry['split'] == 'pilot'}
    if set(baseline) != {'en', 'he', 'ar'}:
        raise ValueError('Expected English, Hebrew and Arabic pilot baselines')
    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Language & Entity & Entity + language \\', r'\midrule']
    for language, label in (('en', 'English'), ('he', 'Hebrew'), ('ar', 'Arabic')):
        entry = baseline[language]
        entity = 100 * entry['entity_correct']['estimate']
        requested = 100 * entry['requested_language_correct']['estimate']
        if entry['entity_correct']['n_subjects'] != 57:
            raise ValueError('Unexpected baseline subject count')
        lines.append(f'{label} & {entity:.1f}\\% & {requested:.1f}\\% ' + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (OUT / 'baseline_table_pilot.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'figure_pdf': str(OUT / 'paired_effects_pilot.pdf'),
                      'table_tex': str(OUT / 'baseline_table_pilot.tex')}, indent=2))


if __name__ == '__main__':
    main()

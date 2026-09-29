"""Plot actual Aya exploratory activation-control specificity with intervals."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / 'tmp/matplotlib'
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_CACHE))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = ROOT / 'results/tau_mechanism_20260926/results/analysis-reviewed-pilot-with-mechanism-20260926.json'
OUT = ROOT / 'paper/generated'
WINDOWS = [('early', 'Early 4–5'), ('lower_middle', 'Mid 12–13'),
           ('upper_middle', 'Mid 20–21'), ('late', 'Late 28–29')]


def main():
    report = json.loads(SOURCE.read_text(encoding='utf-8'))
    if report['data_kind'] != 'research' or report['model']['model_id'] != 'CohereLabs/aya-23-8B':
        raise ValueError('Expected actual Aya research results')
    rows = {(entry['window'], entry['language']): entry
            for entry in report['patch_specificity']
            if entry['split'] == 'pilot' and entry['component'] == 'residual'}
    if set(rows) != {(w, l) for w, _ in WINDOWS for l in ('he', 'ar')}:
        raise ValueError('Missing a predeclared pilot window/language comparison')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(3.35, 3.25), constrained_layout=True)
    for language, shift, color, label in (
        ('he', 0.11, '#315c9d', 'Hebrew'),
        ('ar', -0.11, '#bd5e39', 'Arabic')):
        for index, (window, _) in enumerate(WINDOWS):
            entry = rows[window, language]
            value = entry['specificity']['estimate']
            interval = entry['specificity']['ci95']
            y = len(WINDOWS) - 1 - index + shift
            if interval is not None:
                ax.plot(interval, [y, y], color=color, linewidth=1.8)
            ax.scatter([value], [y], color=color, s=26, zorder=3,
                       label=label if index == 0 else None)
    ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(len(WINDOWS)), [label for _, label in reversed(WINDOWS)])
    ax.set_xlabel('Same-entity minus unrelated donor\nanswer log likelihood (nats)')
    ax.grid(axis='x', alpha=0.2)
    ax.legend(frameon=False, loc='best')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'activation_specificity_pilot.pdf')
    fig.savefig(OUT / 'activation_specificity_pilot.png', dpi=220)
    plt.close(fig)
    print(OUT / 'activation_specificity_pilot.pdf')


if __name__ == '__main__':
    main()

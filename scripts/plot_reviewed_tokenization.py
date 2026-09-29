"""Publication-ready descriptive plot from the saved Aya tokenization CSV."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/01a09b51/reviewed_tokenization_20260926/reviewed_57_prompt_tokenization.csv'
OUT = ROOT / 'paper/generated'


def main():
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 342:
        raise ValueError('Expected 342 reviewed paired prompts')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['language'], row['subject_qid']].append(row)
    if len(grouped) != 114 or any(len(group) != 3 for group in grouped.values()):
        raise ValueError('Expected three templates for each of 57 subjects in two languages')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, ax = plt.subplots(figsize=(5.7, 3.5), constrained_layout=True)
    styles = {'he': ('Hebrew', '#315c9d', 'o'), 'ar': ('Arabic', '#bd5e39', '^')}
    for language in ('he', 'ar'):
        points = []
        for (lang, _), group in grouped.items():
            if lang != language:
                continue
            u = sum(int(row['u_subject_tokens']) for row in group) / 3
            d = sum(int(row['d_subject_tokens']) for row in group) / 3
            points.append((u, d))
        label, color, marker = styles[language]
        ax.scatter([p[0] for p in points], [p[1] for p in points],
                   label=f'{label} (n=57)', color=color, marker=marker,
                   s=29, alpha=0.78, edgecolors='white', linewidths=0.35)
    ax.plot([0, 20], [0, 20], linestyle='--', color='#666666', linewidth=1,
            label='Equal token count')
    ax.set_xlabel('Unmarked subject tokens in prompt')
    ax.set_ylabel('Marked subject tokens in prompt')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, max(float(row['d_subject_tokens']) for row in rows) + 2)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc='upper left')
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / 'reviewed_tokenization.pdf', metadata={
        'Title': 'Aya-23-8B tokenization of reviewed Hebrew and Arabic name pairs'})
    figure.savefig(OUT / 'reviewed_tokenization.png', dpi=220)
    plt.close(figure)
    print(OUT / 'reviewed_tokenization.pdf')


if __name__ == '__main__':
    main()

"""Publication-ready exploratory figure from the verified E6 analysis JSON."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/aya-e6-position-decomposition-analysis-20260928.json"
OUT = ROOT / "results/figures"
PARTS = (
    ("A_to_B", "A→B  same text, new tokens", "#7051A1"),
    ("B_to_C", "B→C  position gaps", "#D28E28"),
    ("C_to_D", "C→D  mark context", "#198C86"),
    ("A_to_D", "A→D  total", "#222222"),
)


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    if report["total_fact_language_pairs"] != 246:
        raise ValueError("Expected the verified 246-row E6 analysis")
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.0), sharex=True, sharey=True)
    for row_index, cohort in enumerate(("pilot", "heldout", "external_google_re")):
        for col_index, language in enumerate(("ar", "he")):
            ax = axes[row_index, col_index]
            cell = report["cells"][f"{cohort}_{language}"]
            for index, (part, label, color) in enumerate(PARTS):
                item = cell["components"][part]
                value = item["relation_macro_mean_nats"]
                lo, hi = item["answer_cluster_bootstrap_ci95_nats"]
                y = len(PARTS) - 1 - index
                ax.errorbar(value, y, xerr=[[value - lo], [hi - value]],
                            fmt="o", color=color, ecolor=color, markersize=6,
                            elinewidth=1.8, capsize=3, zorder=3)
            ax.axvline(0, color="#777777", lw=0.9, ls="--", zorder=1)
            ax.set_xlim(-4.15, 4.15)
            ax.set_ylim(-0.5, 3.5)
            ax.set_yticks(range(4), [item[1] for item in reversed(PARTS)])
            ax.grid(axis="x", alpha=0.18, lw=0.6)
            cohort_label = {"pilot": "Pilot", "heldout": "Held-out",
                            "external_google_re": "Google-RE external"}[cohort]
            ax.set_title(f"{cohort_label} · {'Arabic' if language == 'ar' else 'Hebrew'}"
                         f"  (n={cell['subject_count']})", fontsize=10, loc="left")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=8.5)
    fig.supxlabel("Change in canonical-answer log likelihood (nats)", fontsize=10, y=0.095)
    fig.suptitle("Aya-23-8B E6: exploratory position decomposition", fontsize=13, y=0.98)
    fig.text(0.5, 0.04,
             "Points: relation-macro means; bars: 95% answer-cluster bootstrap CIs. "
             "B/C are counterfactual; A→B changes token identity and count together; no pure-count contrast.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.21, right=0.98, top=0.92, bottom=0.15,
                        hspace=0.38, wspace=0.12)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "e6_position_decomposition_20260928.png", dpi=220)
    fig.savefig(OUT / "e6_position_decomposition_20260928.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

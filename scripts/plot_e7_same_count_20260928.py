"""Static publication-style forest plot for exploratory E7 controls."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/aya-e7-same-count-analysis-20260928.json"
PNG = ROOT / "results/figures/e7_same_count_20260928.png"
SVG = ROOT / "results/figures/e7_same_count_20260928.svg"
ROWS = (
    ("pilot_he", "Pilot · Hebrew"),
    ("pilot_ar", "Pilot · Arabic"),
    ("heldout_he", "Held-out · Hebrew"),
    ("heldout_ar", "Held-out · Arabic"),
    ("external_google_re_he", "Google-RE · Hebrew"),
    ("external_google_re_ar", "Google-RE · Arabic"),
)
PANELS = (
    ("D_minus_full_mean", "Marked minus mean same-count unmarked"),
    ("early_minus_late", "Two same-count unmarked split patterns"),
)


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    if report["pairs"] != 246 or set(dict(ROWS)) != set(report["cells"]):
        raise ValueError("E7 result grid missing a figure cell")
    y = np.arange(len(ROWS))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True,
                             constrained_layout=True)
    for axis, (contrast, title) in zip(axes, PANELS):
        means = []
        lo = []
        hi = []
        for cell, _ in ROWS:
            value = report["cells"][cell]["contrasts"][contrast]
            means.append(value["relation_macro_mean_nats"])
            left, right = value["answer_cluster_bootstrap_ci95_nats"]
            lo.append(left)
            hi.append(right)
        colors = ["#2877A8" if cell.endswith("_he") else "#C87838" for cell, _ in ROWS]
        axis.axvline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=0)
        for i, (point, left, right, color) in enumerate(zip(means, lo, hi, colors)):
            axis.plot([left, right], [y[i], y[i]], color=color, linewidth=2.1)
            axis.plot(point, y[i], marker="o", color=color, markersize=5.5)
        axis.set_title(title, fontsize=10, pad=10)
        axis.set_xlabel("Canonical-answer log-likelihood difference (nats)", fontsize=8)
        axis.grid(axis="x", alpha=0.16)
        axis.set_ylim(-0.6, len(ROWS) - 0.4)
        axis.invert_yaxis()
        axis.tick_params(labelsize=8)
    axes[0].set_yticks(y, [label for _, label in ROWS])
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle("Aya-23-8B · exploratory same-count controls", fontsize=12)
    fig.supxlabel("Points: relation-macro means · bars: 95% answer-cluster bootstrap intervals. Forced splits are off-manifold; count is not causally isolated.",
                  fontsize=8, y=-0.012)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(PNG), "svg": str(SVG)}))


if __name__ == "__main__":
    main()

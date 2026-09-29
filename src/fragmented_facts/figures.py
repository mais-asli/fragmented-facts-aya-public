"""Regenerate vector PDF figures and LaTeX tables from measured analysis files."""
from pathlib import Path

from .io import atomic_text, read_json


def make_figures(analysis_path, output):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output).resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    report = read_json(analysis_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    fixture = report["data_kind"] == "synthetic_fixture"
    tag = "SYNTHETIC SOFTWARE TEST — NOT AYA RESULTS" if fixture else "Aya-23-8B"
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    files = []
    for section, metric, title, filename in (
        ("orthography", "accuracy", "Orthographic intervention: D minus U accuracy", "orthography_accuracy"),
        ("patch_specificity", "specificity", "Same-entity minus unrelated-donor likelihood", "patch_specificity")):
        entries = [e for e in report[section] if e["split"] == "test" and e[metric]["estimate"] is not None]
        if not entries:
            continue
        fig, ax = plt.subplots(figsize=(6.4, max(2.5, len(entries) * 0.4)))
        labels = [e["language"] + (" / " + e["window"] + " / " + e["component"] if "window" in e else "") for e in entries]
        x = np.array([e[metric]["estimate"] for e in entries])
        for i, entry in enumerate(entries):
            ci = entry[metric]["ci95"]
            if ci is not None:
                ax.plot(ci, [i, i], color="#177e89", linewidth=1.8)
        ax.scatter(x, np.arange(len(entries)), color="#17324d", zorder=3)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_yticks(np.arange(len(entries)), labels)
        ax.set_xlabel(title + " (95% subject-bootstrap intervals where available)")
        ax.set_title(tag)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = out / f"{filename}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            files.append(str(path))
        plt.close(fig)
    table = ["% Generated from saved observations. " + tag, r"\begin{tabular}{llrr}", r"\toprule",
             r"Language & Split & Entity accuracy & Subjects \\", r"\midrule"]
    for e in report["baseline"]:
        value = e["entity_correct"]
        table.append(f"{e['language']} & {e['split']} & {100 * value['estimate']:.1f} & {value['n_subjects']} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    atomic_text(out / "baseline_table.tex", "\n".join(table) + "\n")
    files.append(str(out / "baseline_table.tex"))
    return files

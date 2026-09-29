"""Choose sites from English development localization, never held-out effects."""
import collections
from pathlib import Path
import numpy as np

from .analysis import load_runs
from .io import file_hash, read_json, write_json


def select_sites(run_paths, config_path, output_config, output_evidence, layer_count):
    rows, manifests = load_runs(run_paths)
    if any(m["experiment"] != "E4" or m["split"] != "dev" for m in manifests):
        raise ValueError("Site selection accepts only E4 development runs")
    selected = [r for r in rows if r["key"]["language"] == "en" and r["key"]["site"] == "last_subject"]
    if len({r["fact"]["subject_qid"] for r in selected}) < 10:
        raise ValueError("At least ten independent English development subjects are required for this selector")
    by_layer = collections.defaultdict(list)
    for row in selected:
        by_layer[row["key"]["layer"]].append(row["restoration_gain"])
    if len(by_layer) < 4 or layer_count < 8:
        raise ValueError("Need four scanned layers and an eight-or-more-layer model")
    # Four depth bands avoid choosing all windows around one development peak.
    groups = [[] for _ in range(4)]
    for layer in by_layer:
        if layer + 1 < layer_count:
            groups[min(3, 4 * layer // layer_count)].append(layer)
    if any(not g for g in groups):
        raise ValueError("The development scan must cover all four depth bands")
    windows = []
    for band, layers in enumerate(groups):
        best = min(layers, key=lambda l: (-float(np.mean(by_layer[l])), l))
        windows.append({"name": f"band{band + 1}", "layers": [best, best + 1]})
    primary = min(windows, key=lambda w: (-float(np.mean(by_layer[w["layers"][0]])), w["layers"][0]))["name"]
    config = read_json(config_path)
    config.update(windows=windows, primary_window=primary,
        window_selection_reason="Predeclared selector: greatest mean English-dev last-subject restoration gain in each depth quartile; primary is greatest of the four. Consecutive two-layer windows.")
    if Path(output_config).exists() or Path(output_evidence).exists():
        raise FileExistsError("Do not overwrite site choices; use a new explicit version")
    write_json(output_config, config)
    evidence = {"experiment": "E4", "split": "dev", "layer_count": layer_count,
        "subjects": len({r["fact"]["subject_qid"] for r in selected}),
        "mean_gain_by_layer": {str(l): float(np.mean(v)) for l, v in by_layer.items()},
        "windows": windows, "primary_window": primary,
        "sources": [{"path": str(p), "manifest_sha256": file_hash(Path(p) / "manifest.json"),
                     "completion_sha256": file_hash(Path(p) / "completion.json")} for p in run_paths]}
    write_json(output_evidence, evidence)
    return evidence

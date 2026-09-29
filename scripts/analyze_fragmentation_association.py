"""Exploratory subject-level association of added tokens with Aya response."""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.analysis import paired_rows

BASE = ROOT / "results/tau_research_20260926"
ANALYSIS = BASE / "results/analysis-reviewed-pilot-20260926.json"
TOKENS = ROOT / "outputs/01a09b51/reviewed_tokenization_20260926/reviewed_57_prompt_tokenization.csv"
OUT = ROOT / "results/fragmentation-association-pilot-20260926.json"


def main():
    report = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    if report["data_kind"] != "research":
        raise ValueError("Actual Aya study results required")
    token_rows = {}
    with TOKENS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            token_rows[row["fact_id"], row["language"], row["template"]] = row
    if len(token_rows) != 342:
        raise ValueError("Expected tokenizer records for every E2 pair")
    rows = []
    for relative in report["run_paths"]:
        for path in (BASE / relative / "items").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["key"]["experiment"] == "E2":
                rows.append(row)
    output = {"scope": "Exploratory within-language subject-level association; not causal", "languages": []}
    for language in ("he", "ar"):
        pairs, missing = paired_rows(rows, "E2", "U", "D", language)
        if len(pairs) != 171 or missing:
            raise ValueError("Incomplete E2 paired observations")
        cells = defaultdict(list)
        for u, d in pairs:
            key = d["key"]
            token = token_rows[key["fact_id"], language, key["template"]]
            cells[key["fact_id"]].append({
                "relation": d["fact"]["relation"],
                "token_delta": int(token["token_delta"]),
                "base_length": int(token["base_length"]),
                "sitelinks": d.get("popularity", {}).get("sitelinks") or 0,
                "accuracy_delta": int(d["evaluation"]["entity_correct"])
                                  - int(u["evaluation"]["entity_correct"]),
                "likelihood_delta": d["gold_score"]["sum_logprob"]
                                    - u["gold_score"]["sum_logprob"],
            })
        if len(cells) != 57 or any(len(v) != 3 for v in cells.values()):
            raise ValueError("Subject-level template coverage differs from protocol")
        subjects = []
        for fact_id, prompts in sorted(cells.items()):
            subjects.append({
                "fact_id": fact_id,
                "relation": prompts[0]["relation"],
                "token_delta": float(np.mean([p["token_delta"] for p in prompts])),
                "base_length": prompts[0]["base_length"],
                "log_sitelinks": math.log1p(prompts[0]["sitelinks"]),
                "accuracy_delta": float(np.mean([p["accuracy_delta"] for p in prompts])),
                "likelihood_delta": float(np.mean([p["likelihood_delta"] for p in prompts])),
            })
        predictor = np.asarray([r["token_delta"] for r in subjects], dtype=float)
        language_result = {"language": language, "n_subjects": len(subjects),
                           "token_delta_range": [float(predictor.min()), float(predictor.max())],
                           "outcomes": {}}
        # P19 is the reference category. Four relation cells are fixed in the
        # frozen cohort; the model is descriptive and has no held-out check.
        X = np.asarray([[1.0, r["token_delta"], r["base_length"], r["log_sitelinks"],
                         *[float(r["relation"] == rel) for rel in ("P20", "P159", "P740")]]
                        for r in subjects], dtype=float)
        if np.linalg.matrix_rank(X) != X.shape[1]:
            raise ValueError("Singular covariate design")
        for outcome in ("accuracy_delta", "likelihood_delta"):
            y = np.asarray([r[outcome] for r in subjects], dtype=float)
            spearman = spearmanr(predictor, y)
            fit = sm.OLS(y, X).fit(cov_type="HC3")
            language_result["outcomes"][outcome] = {
                "spearman_rho": float(spearman.statistic),
                "spearman_p_unadjusted": float(spearman.pvalue),
                "adjusted_token_delta_coefficient": float(fit.params[1]),
                "adjusted_token_delta_hc3_ci95": fit.conf_int()[1].tolist(),
                "adjusted_token_delta_p_unadjusted": float(fit.pvalues[1]),
                "adjusted_r_squared": float(fit.rsquared_adj),
                "adjustment": "base characters, log(1+sitelinks), relation indicators",
            }
        output["languages"].append(language_result)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

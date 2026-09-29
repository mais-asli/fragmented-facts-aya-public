"""Every 28 September number in the paper, recomputed from the per-prompt records in _work/.

Point estimates are recomputed here and must equal the saved analysis files exactly; intervals
and p-values are also recomputed (same estimators, independent resampling streams), and the
paper reports the saved official intervals and p-values, which this script also copies into
_work/computed_v3.json under "official". Run consolidate_v3.py first.
Set FF_ROOT to the project root if this folder is not <nlp final>/paper_v3.
"""
import json, os, itertools, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FF_ROOT", HERE.parent.parent)); R = ROOT / "results"
W = HERE.parent / "_work"
load = lambda n: [json.loads(l) for l in open(W / f"{n}.jsonl", encoding="utf-8")]
OUT = {}

def hier_boot(groups, B=10000, seed=1729):
    """Answer strata with replacement, then subjects within each drawn stratum; macro mean of stratum means."""
    rng = np.random.default_rng(seed); arrs = [np.array(v) for v in groups.values()]; k = len(arrs); out = np.empty(B)
    for b in range(B):
        out[b] = np.mean([arrs[i][rng.integers(0, len(arrs[i]), len(arrs[i]))].mean() for i in rng.integers(0, k, k)])
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
def sign_flip(groups, seed=1729, mc=200000):
    m = np.array([np.mean(v) for v in groups.values()]); obs = abs(m.mean()); k = len(m)
    if k <= 18:
        return float(np.mean([abs((m * np.array(s)).mean()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=k)]))
    s = np.random.default_rng(seed).choice([1, -1], size=(mc, k))
    return (int((np.abs((s * m).mean(1)) >= obs - 1e-12).sum()) + 1) / (mc + 1)
def macro(groups): return float(np.mean([np.mean(v) for v in groups.values()]))
def holm(ps):
    order = sorted(ps, key=ps.get); out = {}; run = 0
    for i, k in enumerate(order): run = max(run, min(1, (len(ps) - i) * ps[k])); out[k] = run
    return out
def stat(groups, test=False):
    d = {"strata": len(groups), "subjects": sum(len(v) for v in groups.values()), "macro": macro(groups),
         "subject_mean": float(np.mean([x for v in groups.values() for x in v])), "ci": hier_boot(groups)}
    if test: d["p"] = sign_flip(groups)
    return d

# ---------------- source-checked cohorts: E2-style U/D
sv = load("sv_ud"); OUT["sv"] = {}
stratum = {"arabic_geonames": lambda r: r["fact"]["country_iso2"], "hebrew_cbs": lambda r: r["fact"]["answer_he"]}
for coh in stratum:
    rows = [r for r in sv if r["key"]["cohort"] == coh]; res = {"n": len(rows)}
    for name, fn, test in [("t1_margin", lambda s: s["t1"]["D"]["gold_minus_distractor"] - s["t1"]["U"]["gold_minus_distractor"], True),
                           ("t2_margin", lambda s: s["t2"]["D"]["gold_minus_distractor"] - s["t2"]["U"]["gold_minus_distractor"], True),
                           ("t1_gold", lambda s: s["t1"]["D"]["gold"]["sum_logprob"] - s["t1"]["U"]["gold"]["sum_logprob"], True),
                           ("t1_distractor", lambda s: s["t1"]["D"]["distractor"]["sum_logprob"] - s["t1"]["U"]["distractor"]["sum_logprob"], False)]:
        g = collections.defaultdict(list)
        for r in rows: g[stratum[coh](r)].append(fn(r["scores"]))
        res[name] = stat(g, test)
    t = lambda v: [r["scores"]["t1"][v]["subject_tokens"] for r in rows]
    res["tokens"] = {"U_mean": float(np.mean(t("U"))), "D_mean": float(np.mean(t("D"))),
                     "delta_median": float(np.median(np.array(t("D")) - np.array(t("U")))), "all_increase": bool(all(np.array(t("D")) > np.array(t("U"))))}
    res["distinct_answers"] = len({stratum[coh](r) for r in rows})
    OUT["sv"][coh] = res
OUT["sv_holm_recomputed"] = holm({c: OUT["sv"][c]["t1_margin"]["p"] for c in stratum})

# ---------------- source-disjoint E5 at the fixed site
e5 = load("sv_e5"); OUT["sv_e5"] = {}
for coh in stratum:
    rows = [r for r in e5 if r["key"]["cohort"] == coh]; res = {"n": len(rows)}
    def grp(fn):
        g = collections.defaultdict(list)
        for r in rows: g[r["fact"]["answer_stratum"]].append(fn(r["scores"]))
        return g
    for cond in ["same_entity_U_to_D", "unrelated_U_to_D", "identity_D_to_D", "first_subject_U_to_D"]:
        res[cond] = stat(grp(lambda s: s[cond]["gold_minus_distractor"] - s["baseline_D"]["gold_minus_distractor"]), cond == "same_entity_U_to_D")
    res["identity_maxabs"] = float(max(abs(r["scores"]["identity_D_to_D"]["gold_minus_distractor"] - r["scores"]["baseline_D"]["gold_minus_distractor"]) for r in rows))
    res["same_minus_unrelated"] = stat(grp(lambda s: s["same_entity_U_to_D"]["gold_minus_distractor"] - s["unrelated_U_to_D"]["gold_minus_distractor"]))
    res["marking_margin_in_subset"] = stat(grp(lambda s: s["baseline_D"]["gold_minus_distractor"] - s["baseline_U"]["gold_minus_distractor"]))
    res["gold_only_gain"] = stat(grp(lambda s: s["same_entity_U_to_D"]["gold"]["sum_logprob"] - s["baseline_D"]["gold"]["sum_logprob"]))
    OUT["sv_e5"][coh] = res
OUT["sv_e5_holm_recomputed"] = holm({c: OUT["sv_e5"][c]["same_entity_U_to_D"]["p"] for c in stratum})
svk = {(r["key"]["cohort"], r["fact"]["fact_id"]): r["scores"]["t1"] for r in sv}
OUT["sv_e5_baseline_maxdev"] = max(abs(r["scores"]["baseline_" + v][a]["sum_logprob"] - svk[(r["key"]["cohort"], r["fact"]["fact_id"])][v][a]["sum_logprob"])
                                   for r in e5 for v in "UD" for a in ("gold", "distractor"))

# ---------------- E6 and E7 (template 1, the 246 earlier fact-language pairs)
def rel_boot(rows, fn, B=10000, seed=17):
    """Relation-macro mean; answer-QID clusters resampled within relation."""
    byrel = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows: byrel[r["rel"]][r["obj"]].append(fn(r))
    point = float(np.mean([np.mean([x for v in cl.values() for x in v]) for cl in byrel.values()]))
    rels = [[np.array(v) for v in cl.values()] for cl in byrel.values()]; rng = np.random.default_rng(seed); out = np.empty(B)
    for b in range(B):
        out[b] = np.mean([np.concatenate([cls[i] for i in rng.integers(0, len(cls), len(cls))]).mean() for cls in rels])
    return {"mean": point, "ci": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]}
key = lambda r: (r["key"]["cohort"], r["key"]["language"], r["fact"]["fact_id"])
e6 = {key(r): r for r in load("e6")}; e7 = {key(r): r for r in load("e7")}
assert set(e6) == set(e7) and len(e6) == 246
OUT["e6"] = {}; OUT["e7"] = {}; OUT["e6_e7_levels"] = {}; OUT["e6_tokens"] = {}; maxdev = 0
for coh, lang in itertools.product(["pilot", "heldout", "external_google_re"], ["he", "ar"]):
    rows = []
    for k, r in e6.items():
        if k[:2] != (coh, lang): continue
        s, s7 = r["sum_logprob"], e7[k]["sum_logprob"]
        maxdev = max(maxdev, abs(s["A"] - r["e2_baseline"]["A"]), abs(s["D"] - r["e2_baseline"]["D"]))
        rows.append({"rel": r["fact"]["relation"], "obj": r["fact"]["object_qid"], **s, **s7, "full": (s7["early"] + s7["late"]) / 2,
                     "tA": r["subject_tokens"]["A"], "tB": r["subject_tokens"]["B"], "tD": r["subject_tokens"]["D"],
                     "tM": e7[k]["subject_tokens"]["mid"], "tE": e7[k]["subject_tokens"]["early"], "tL": e7[k]["subject_tokens"]["late"]})
    cell = f"{coh}/{lang}"
    OUT["e6"][cell] = {"n": len(rows), **{n: rel_boot(rows, f) for n, f in [("A_B", lambda r: r["B"] - r["A"]), ("B_C", lambda r: r["C"] - r["B"]),
                                                                            ("C_D", lambda r: r["D"] - r["C"]), ("A_D", lambda r: r["D"] - r["A"])]}}
    OUT["e7"][cell] = {n: rel_boot(rows, f) for n, f in [("D_minus_full", lambda r: r["D"] - r["full"]), ("early_minus_late", lambda r: r["early"] - r["late"]),
                                                          ("mid_minus_B", lambda r: r["mid"] - r["B"]), ("full_minus_mid", lambda r: r["full"] - r["mid"]),
                                                          ("D_minus_B", lambda r: r["D"] - r["B"])]}
    OUT["e6_e7_levels"][cell] = {c: rel_boot(rows, lambda r, c=c: r[c] - r["A"]) for c in ("B", "mid", "full", "D")}
    OUT["e6_tokens"][cell] = {"A": float(np.mean([r["tA"] for r in rows])), "B": float(np.mean([r["tB"] for r in rows])),
                              "mid": float(np.mean([r["tM"] for r in rows])), "D": float(np.mean([r["tD"] for r in rows])),
                              "full_equals_D": all(r["tE"] == r["tD"] == r["tL"] for r in rows), "A_equals_B": sum(r["tA"] == r["tB"] for r in rows)}
OUT["e6_baseline_maxdev"] = maxdev

# ---------------- official saved values (reported in the paper), with exact point-estimate checks
J = lambda n: json.load(open(R / n, encoding="utf-8"))
off = {"sv": J("aya-source-verified-external-analysis-20260928.json"), "sv_e5": J("aya-source-verified-e5-analysis-20260928.json"),
       "e6": J("aya-e6-position-decomposition-analysis-20260928.json"), "e7": J("aya-e7-same-count-analysis-20260928.json"),
       "margin_old": J("posthoc_gold_distractor_margin_20260928.json")}
for coh in stratum:
    o = off["sv"]["cohorts"][coh]
    for mine, theirs in [("t1_margin", "primary_t1_paired_margin"), ("t2_margin", "secondary_t2_paired_margin"), ("t1_gold", "secondary_t1_gold_loglik_change")]:
        assert abs(OUT["sv"][coh][mine]["macro"] - o[theirs]["answer_macro_mean"]) < 1e-9, (coh, mine)
    o5 = off["sv_e5"]["cohorts"][coh]
    assert abs(OUT["sv_e5"][coh]["same_entity_U_to_D"]["macro"] - o5["primary_same_entity_margin_gain"]["answer_macro_mean"]) < 1e-9
    assert abs(OUT["sv_e5"][coh]["gold_only_gain"]["macro"] - o5["same_entity_gold_only_gain"]["answer_macro_mean"]) < 1e-9
for cell in OUT["e6"]:
    c = cell.replace("/", "_")
    for comp in ("A_B", "B_C", "C_D", "A_D"):
        t = comp.replace("_", "_to_")
        assert abs(OUT["e6"][cell][comp]["mean"] - off["e6"]["cells"][c]["components"][t]["relation_macro_mean_nats"]) < 1e-9, (cell, comp)
    for comp, t in [("D_minus_full", "D_minus_full_mean"), ("early_minus_late", "early_minus_late"), ("mid_minus_B", "mid_minus_B"),
                    ("full_minus_mid", "full_mean_minus_mid"), ("D_minus_B", "D_minus_B")]:
        assert abs(OUT["e7"][cell][comp]["mean"] - off["e7"]["cells"][c]["contrasts"][t]["relation_macro_mean_nats"]) < 1e-9, (cell, comp)
OUT["official"] = {
    "sv": {c: {**{k: {"macro": v["answer_macro_mean"], "ci": v["hierarchical_bootstrap_ci95"], "p": v.get("two_sided_p")}
                  for k, v in off["sv"]["cohorts"][c].items() if isinstance(v, dict) and "answer_macro_mean" in v},
               "generation": off["sv"]["cohorts"][c]["generation"], "paired_generation": off["sv"]["cohorts"][c]["paired_generation"],
               "english_control": off["sv"]["cohorts"][c]["english_control"]} for c in stratum},
    "sv_holm": off["sv"]["holm_adjusted_primary_p"],
    "sv_e5": {c: {"same_entity": off["sv_e5"]["cohorts"][c]["primary_same_entity_margin_gain"],
                  "same_minus_unrelated": off["sv_e5"]["cohorts"][c]["same_entity_minus_unrelated_margin"],
                  "gold_only": off["sv_e5"]["cohorts"][c]["same_entity_gold_only_gain"],
                  "controls": off["sv_e5"]["cohorts"][c]["controls_vs_baseline_D"],
                  "identity_maxabs": off["sv_e5"]["cohorts"][c]["identity_max_absolute_margin_deviation_nats"]} for c in stratum},
    "sv_e5_holm": off["sv_e5"]["holm_adjusted_primary_p"],
    "e6": {c: {k: {"mean": v["relation_macro_mean_nats"], "ci": v["answer_cluster_bootstrap_ci95_nats"]} for k, v in x["components"].items()}
           for c, x in off["e6"]["cells"].items()},
    "e7": {c: {k: {"mean": v["relation_macro_mean_nats"], "ci": v["answer_cluster_bootstrap_ci95_nats"]} for k, v in x["contrasts"].items()}
           for c, x in off["e7"]["cells"].items()},
    "margin_old": {c: {l: {"margin": v["gold_minus_distractor_delta"]["relation_macro_mean"], "ci": v["gold_minus_distractor_delta"]["subject_cluster_bootstrap_ci95"],
                           "holm": v["gold_minus_distractor_delta"].get("holm_p_two_languages"), "distractor": v["distractor_delta"]["relation_macro_mean"]}
                       for l, v in x.items()} for c, x in off["margin_old"]["cohorts"].items()}}
json.dump(OUT, open(W / "computed_v3.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("computed_v3 ok; all point estimates equal the saved analyses")

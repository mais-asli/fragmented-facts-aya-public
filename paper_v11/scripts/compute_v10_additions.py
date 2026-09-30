"""Numbers added in paper version 10, recomputed from the raw per-prompt records.

1. E5 last-layer control (layer 31, prediction position; pilot and held-out): relation-macro gain over
   the unpatched marked run, with 95% subject-cluster intervals (seed 17, 10,000 resamples), and the
   first answer token's share of the unmarked-vs-marked loss, which the control recovers by construction.
2. Split of the layers 12-13 same-entity gain into the first answer token and the later answer tokens.
3. Majority-answer baselines of the source-checked cohorts (always giving the most frequent answer).

Reads results/tau_mechanism_20260926 and results/h27 (complete archive) and _work/sv_ud.jsonl; writes
_work/computed_v10.json and checks every value (including Table 12's intervals) against the rounded
number printed in the paper.
Set FF_ROOT to the project root if this folder is not <root>/paper_v11.
"""
import collections, glob, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FF_ROOT", HERE.parent.parent))
W = HERE.parent / "_work"
sys.path.insert(0, str(HERE))
from stats_util import cluster_effect  # noqa: E402

RUNS = {"pilot": "results/tau_mechanism_20260926/results/e5-*/*/items/*.json",
        "heldout": "results/h27/results/e5-*/*/items/*.json"}
OUT = {"e5_last_layer": {}, "e5_subject_patch_token_split": {}, "majority_baseline": {}}


def records(pattern):
    by = collections.defaultdict(dict)
    for path in glob.glob(str(ROOT / pattern)):
        d = json.load(open(path, encoding="utf-8"))
        k = d["key"]
        if k["variant"] in ("baseline_U", "baseline_D", "late_prediction"):
            name = k["variant"]
        elif k["variant"] == "same_entity" and k.get("window") == "lower_middle":
            name = "same_12_13"
        else:
            continue
        by[(k["fact_id"], k["language"])][name] = d
    return by


def effect(values, keys):
    point, ci, _, n = cluster_effect(values, [k[0] for k in keys], [k[0].split("-")[1] for k in keys])
    return {"estimate": point, "ci95": ci, "n_subjects": n}


for cohort, pattern in RUNS.items():
    by = records(pattern)
    if not by:
        raise FileNotFoundError("No E5 records under " + str(ROOT / pattern))
    for lang in ("he", "ar"):
        keys = sorted(k for k in by if k[1] == lang)
        lp = {name: [by[k][name]["gold_score"]["token_logprobs"] for k in keys]
              for name in ("baseline_U", "baseline_D", "late_prediction", "same_12_13")}
        U, D, L, S = lp["baseline_U"], lp["baseline_D"], lp["late_prediction"], lp["same_12_13"]
        later_unchanged = max(abs(a - b) for l, d in zip(L, D) for a, b in zip(l[1:], d[1:]))
        first_vs_U = max(abs(l[0] - u[0]) for l, u in zip(L, U))
        OUT["e5_last_layer"][f"{cohort}/{lang}"] = {
            "gain": effect([sum(l) - sum(d) for l, d in zip(L, D)], keys),
            "first_token_loss_share": effect([u[0] - d[0] for u, d in zip(U, D)], keys)["estimate"],
            "total_loss": effect([sum(u) - sum(d) for u, d in zip(U, D)], keys)["estimate"],
            "max_abs_later_token_change": later_unchanged, "max_abs_first_token_minus_unmarked": first_vs_U,
            "correct_generations": sum(bool(by[k]["late_prediction"]["evaluation"]["entity_correct"]) for k in keys),
            "n": len(keys)}
        OUT["e5_subject_patch_token_split"][f"{cohort}/{lang}"] = {
            "first_token": effect([s[0] - d[0] for s, d in zip(S, D)], keys),
            "later_tokens": effect([sum(s[1:]) - sum(d[1:]) for s, d in zip(S, D)], keys)}

rows = [json.loads(line) for line in open(W / "sv_ud.jsonl", encoding="utf-8")]
for cohort, field in (("arabic_geonames", "answer_ar"), ("hebrew_cbs", "answer_he")):
    gold = [r["fact"][field] for r in rows if r["key"]["cohort"] == cohort]
    top, count = collections.Counter(gold).most_common(1)[0]
    OUT["majority_baseline"][cohort] = {"most_frequent_answer": top, "count": count, "n": len(gold),
                                        "accuracy": count / len(gold)}

# Values printed in the paper (Section 5.1, Section 5.4 and Table 12).
paper = {"pilot/he": 0.67, "pilot/ar": 0.91, "heldout/he": 0.61, "heldout/ar": 1.42}
for cell, value in paper.items():
    item = OUT["e5_last_layer"][cell]
    assert round(item["gain"]["estimate"], 2) == value, (cell, item["gain"]["estimate"])
    assert abs(item["gain"]["estimate"] - item["first_token_loss_share"]) < 0.01, cell
    assert item["max_abs_later_token_change"] == 0.0, cell
# Table 12: point estimates and 95% intervals, rounded as printed.
table12 = {"pilot/he": ((0.67, 0.06, 1.32), (0.62, 0.10, 1.19), (0.31, -0.05, 0.73)),
           "pilot/ar": ((0.91, 0.39, 1.45), (0.70, 0.36, 1.05), (0.30, -0.08, 0.68)),
           "heldout/he": ((0.61, 0.00, 1.28), (0.75, 0.27, 1.30), (0.59, 0.25, 0.96)),
           "heldout/ar": ((1.42, 0.50, 2.48), (1.08, 0.34, 1.93), (0.26, -0.19, 0.75))}
for cell, rows in table12.items():
    split = OUT["e5_subject_patch_token_split"][cell]
    for printed, item in zip(rows, (OUT["e5_last_layer"][cell]["gain"], split["first_token"], split["later_tokens"])):
        got = tuple(round(x + 0.0, 2) for x in [item["estimate"]] + item["ci95"])
        assert got == printed, (cell, printed, got)
    assert OUT["e5_last_layer"][cell]["max_abs_first_token_minus_unmarked"] < 0.04, cell  # "within 0.04 nats"
later = [OUT["e5_subject_patch_token_split"][c]["later_tokens"]["estimate"] for c in paper]
assert (round(min(later), 2), round(max(later), 2)) == (0.26, 0.59), later
assert round(100 * OUT["majority_baseline"]["arabic_geonames"]["accuracy"], 1) == 11.4
assert round(100 * OUT["majority_baseline"]["hebrew_cbs"]["accuracy"], 1) == 7.9

# V11 preserves the small positive held-out Hebrew lower endpoint.
assert tuple(round(x, 3) for x in OUT["e5_last_layer"]["heldout/he"]["gain"]["ci95"]) == (0.005, 1.279)

json.dump(OUT, open(W / "computed_v10.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("computed_v10 ok; last-layer gains",
      {c: round(OUT["e5_last_layer"][c]["gain"]["estimate"], 3) for c in paper},
      "later-token range", [round(min(later), 3), round(max(later), 3)])

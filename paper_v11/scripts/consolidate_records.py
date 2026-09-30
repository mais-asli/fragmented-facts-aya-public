"""Consolidate the raw per-prompt Aya records into compact JSONL files in ../_work/.

Inputs (read only): ../../results/tau_research_20260926/results/{e1,e2,e3,screen}-*/*/items/*.json
and ../../results/tau_mechanism_20260926/results/e5-*/*/items/*.json.
Outputs: ../_work/{e1,e2,e3,e5a,e5b,screen}.jsonl (one record per prompt/condition).
"""
import glob, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
W = HERE.parent / "_work"
W.mkdir(exist_ok=True)
RUNS = {
    "e1.jsonl": "results/tau_research_20260926/results/e1-*/*/items/*.json",
    "e2.jsonl": "results/tau_research_20260926/results/e2-*/*/items/*.json",
    "e3.jsonl": "results/tau_research_20260926/results/e3-*/*/items/*.json",
    "screen.jsonl": "results/tau_research_20260926/results/screen-*/*/items/*.json",
    "e5a.jsonl": "results/tau_mechanism_20260926/results/e5-*shard0/*/items/*.json",
    "e5b.jsonl": "results/tau_mechanism_20260926/results/e5-*shard1/*/items/*.json",
}


def compact(d):
    p = d.get("prompt", {})
    gs, ds = d.get("gold_score") or {}, d.get("distractor_score") or {}
    r = {"key": d["key"], "status": d.get("status"), "relation": d["fact"]["relation"],
         "subject_qid": d["fact"]["subject_qid"], "object_qid": d["fact"].get("object_qid"),
         "sitelinks": (d.get("popularity") or {}).get("sitelinks"), "subject": p.get("subject"),
         "subject_tokens": p.get("subject_tokens"), "base_characters": p.get("base_characters"),
         "subject_words": p.get("subject_words"), "n_input_ids": len(p.get("input_ids", [])),
         "user_text": p.get("text"), "subject_indices": p.get("subject_indices"),
         "gold_answer": gs.get("answer"), "gold_sum": gs.get("sum_logprob"), "gold_mean": gs.get("mean_logprob"),
         "gold_tokens": gs.get("answer_tokens"), "distractor": ds.get("answer"),
         "distractor_sum": ds.get("sum_logprob"), "margin": d.get("likelihood_margin")}
    g = d.get("generation")
    if g:
        r.update({"gen_text": g.get("text"), "gen_mean_lp": g.get("answer_mean_logprob"),
                  "gen_tokens": g.get("answer_token_count"), "truncated": g.get("truncated"),
                  "gen_first_lp": (g.get("generated_token_logprobs") or [None])[0]})
    e = d.get("evaluation")
    if e:
        r.update({"entity_correct": e.get("entity_correct"), "lang_correct": e.get("requested_language_correct"),
                  "category": e.get("category")})
    ea = d.get("evaluation_all_aliases")
    if ea:
        r["entity_correct_all"] = ea.get("entity_correct")
    return r


for name, pattern in RUNS.items():
    files = sorted(glob.glob(str(ROOT / pattern)))
    with open(W / name, "w", encoding="utf-8") as out:
        for f in files:
            with open(f, encoding="utf-8") as fh:
                out.write(json.dumps(compact(json.load(fh)), ensure_ascii=False) + "\n")
    print(name, len(files))

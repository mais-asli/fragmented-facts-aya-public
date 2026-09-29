"""Command-line workflow. Use --help on any subcommand for inputs and outputs."""
import argparse
import json
from pathlib import Path

from .io import code_hash, digest, read_json, read_jsonl, write_json


def parser():
    p = argparse.ArgumentParser(description="Fragmented Facts: Aya-23-8B research pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("fetch", help="Download the original public mLAMA archive")
    a.add_argument("--output", default="data/raw/mlama1.1.zip")
    a.add_argument("--url", default="http://cistern.cis.lmu.de/mlama/mlama1.1.zip")
    a = sub.add_parser("import", help="Create provenance-preserving candidate facts")
    a.add_argument("--archive", default="data/raw/mlama1.1.zip")
    a.add_argument("--output", default="data/raw/candidates.jsonl")
    a.add_argument("--limit", type=int, default=1800)
    a.add_argument("--seed", type=int, default=17)
    a.add_argument("--allow-missing-labels", action="store_true")
    a.add_argument("--relations", nargs="+", default=["P19", "P20", "P159", "P740"])
    a = sub.add_parser("enrich", help="Cache Wikidata labels, aliases, and statements")
    a.add_argument("--facts", required=True)
    a.add_argument("--cache", default="data/raw/wikidata")
    a.add_argument("--output", default="data/raw/candidates_enriched.jsonl")
    a = sub.add_parser("pageviews", help="Optional cached historical pageview proxy")
    a.add_argument("--facts", required=True)
    a.add_argument("--cache", default="data/raw/pageviews")
    a.add_argument("--output", required=True)
    a = sub.add_parser("review-export", help="Create human fact/pair review sheets without overwriting")
    a.add_argument("--facts", required=True)
    a.add_argument("--output", default="data/review")
    a = sub.add_parser("split-assign", help="Pin candidate-pool subject splits before review or inference")
    a.add_argument("--facts", required=True)
    a.add_argument("--output", default="data/review/subject_splits.json")
    a.add_argument("--seed", type=int, default=17)
    a = sub.add_parser("prepare", help="Ingest approved facts/pairs and assign entity-disjoint splits")
    a.add_argument("--facts", required=True)
    a.add_argument("--reviews", default="data/review/facts.csv")
    a.add_argument("--pairs", default="data/review/pairs.csv")
    a.add_argument("--output", default="data/curated/reviewed.jsonl")
    a.add_argument("--seed", type=int, default=17)
    a.add_argument("--split-manifest", default="data/review/subject_splits.json")
    a = sub.add_parser("validate", help="Check schema, Unicode, aliases, and split invariants")
    a.add_argument("--facts", required=True)
    a.add_argument("--templates", default="configs/templates.json")
    a.add_argument("--require-review", action="store_true")
    a = sub.add_parser("freeze", help="Freeze reviewed inputs after Aya feasibility passes")
    a.add_argument("--config", default="configs/study.json")
    a.add_argument("--facts", required=True)
    a.add_argument("--templates", default="configs/templates.json")
    a.add_argument("--snapshot")
    a.add_argument("--smoke")
    a.add_argument("--stage", choices=["behavior", "mechanism"], default="behavior")
    a.add_argument("--development-evidence")
    a.add_argument("--output", required=True)
    for name in ("screen", "run", "workload"):
        a = sub.add_parser(name, help={"screen": "Two independent English eligibility prompts", "run": "Execute one frozen experiment", "workload": "Measure real Aya workload and prompt/score invariants"}[name])
        a.add_argument("--snapshot", required=True)
        a.add_argument("--config", default="configs/study.json")
        a.add_argument("--facts", required=True)
        a.add_argument("--templates", default="configs/templates.json")
        a.add_argument("--output", default="results")
        if name != "workload":
            a.add_argument("--shard", type=int, default=0)
            a.add_argument("--num-shards", type=int, default=1)
        if name == "run":
            a.add_argument("--experiment", choices=["E1", "E2", "E3", "E4", "E5", "diagnostics"], required=True)
            a.add_argument("--split", choices=["pilot", "dev", "test"], required=True)
            a.add_argument("--protocol", required=True)
            a.add_argument("--limit", type=int)
        if name == "workload":
            a.add_argument("--limit", type=int, default=50)
    a = sub.add_parser("attach-screen", help="Attach complete screening outcomes and select core subjects")
    a.add_argument("--facts", required=True)
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--output", default="data/curated/screened.jsonl")
    a = sub.add_parser("select-sites", help="Select four windows from English E4 development data")
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--config", default="configs/study.json")
    a.add_argument("--output-config", default="configs/mechanism.json")
    a.add_argument("--output-evidence", default="results/site_selection.json")
    a.add_argument("--layer-count", type=int, required=True)
    for name in ("analyze", "audit-export"):
        a = sub.add_parser(name)
        a.add_argument("--runs", nargs="+", required=True)
        a.add_argument("--output", required=True)
        a.add_argument("--seed", type=int, default=17)
        if name == "analyze":
            a.add_argument("--bootstrap", type=int, default=10000)
    a = sub.add_parser("agreement")
    a.add_argument("--first", required=True)
    a.add_argument("--second", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("figures", help="Regenerate figures/tables from measured analysis")
    a.add_argument("--analysis", required=True)
    a.add_argument("--output", default="paper/generated")
    a = sub.add_parser("doctor", help="Read local readiness without printing credentials")
    a.add_argument("--output", default="results/readiness.json")
    a = sub.add_parser("integration-test", help="CPU test with a tiny random Cohere model; never Aya research results")
    a.add_argument("--output", default="results/integration")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cmd = args.command
    if cmd == "fetch":
        from .sources import fetch_archive
        result = fetch_archive(args.output, args.url)
    elif cmd == "import":
        from .sources import import_mlama
        result = import_mlama(args.archive, args.output, args.limit, args.relations, args.seed, not args.allow_missing_labels)
    elif cmd == "enrich":
        from .sources import enrich
        result = enrich(args.facts, args.cache, args.output)
    elif cmd == "pageviews":
        from .sources import pageviews
        result = pageviews(args.facts, args.output, args.cache)
    elif cmd == "review-export":
        from .data import review_export
        result = review_export(args.facts, args.output)
    elif cmd == "prepare":
        from .data import prepare
        result = prepare(args.facts, args.reviews, args.pairs, args.output, args.seed, args.split_manifest)
    elif cmd == "split-assign":
        from .data import create_split_manifest
        result = create_split_manifest(args.facts, args.output, args.seed)
    elif cmd == "validate":
        from .data import validate_facts
        from .prompts import validate_templates
        result = validate_facts(read_jsonl(args.facts), args.require_review)
        validate_templates(read_json(args.templates), args.require_review)
    elif cmd == "freeze":
        from .protocol import freeze
        result = freeze(args.config, args.facts, args.templates, args.output, args.snapshot, args.smoke,
                        args.stage, args.development_evidence)
    elif cmd in ("screen", "run", "workload"):
        from .data import validate_facts
        from .model import load_runner
        from .prompts import validate_templates
        config, facts, templates = read_json(args.config), read_jsonl(args.facts), read_json(args.templates)
        validate_facts(facts, require_review=True)
        validate_templates(templates, require_review=True)
        if cmd == "run":
            from .protocol import verify
            frozen = verify(args.protocol, args.config, args.facts, args.templates)
        runner = load_runner(args.snapshot, config)
        if cmd == "screen":
            from .experiments import screen
            result = screen(runner, facts, templates, args.output, config["seed"], args.num_shards, args.shard)
        elif cmd == "workload":
            from .verification import workload
            result = workload(runner, facts, templates, args.output, args.limit)
        else:
            from .experiments import run_experiment
            if any(runner.identity[k] != frozen["model"][k] for k in ("model_id", "revision", "precision", "compute_dtype")):
                raise ValueError("Loaded Aya differs from frozen model identity")
            result = run_experiment(runner, facts, templates, config, frozen, args.experiment, args.split,
                                    args.output, args.shard, args.num_shards, args.limit)
    elif cmd == "attach-screen":
        from .experiments import attach_screen
        result = attach_screen(args.facts, args.runs, args.output)
    elif cmd == "select-sites":
        from .localization import select_sites
        result = select_sites(args.runs, args.config, args.output_config, args.output_evidence, args.layer_count)
    elif cmd == "analyze":
        from .analysis import analyze_runs, load_runs
        from .baselines import failure_predictors, majority_baseline, matched_comparison, risk_coverage, clustered_associations
        report = analyze_runs(args.runs, args.output, args.bootstrap, args.seed)
        rows, _ = load_runs(args.runs)
        report["failure_predictors"] = failure_predictors(rows, args.bootstrap, args.seed)
        report["majority_baseline"] = majority_baseline(rows)
        report["matched_comparison"] = matched_comparison(rows, args.seed)
        report["risk_coverage"] = risk_coverage(rows)
        report["clustered_associations"] = clustered_associations(rows)
        write_json(args.output, report)
        result = {"output": args.output, "observation_counts": report["observation_counts"]}
    elif cmd == "audit-export":
        from .analysis import load_runs
        from .audit import export_audit
        rows, _ = load_runs(args.runs)
        result = export_audit(rows, args.output, args.seed)
    elif cmd == "agreement":
        from .audit import agreement
        result = agreement(args.first, args.second)
        write_json(args.output, result)
    elif cmd == "figures":
        from .figures import make_figures
        result = make_figures(args.analysis, args.output)
    elif cmd == "doctor":
        from .verification import doctor
        result = doctor(args.output)
    else:
        from .verification import integration
        result = integration(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

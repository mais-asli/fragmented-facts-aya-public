"""Verify the Aya held-out archive before using its numbers in the paper."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "89da1a0ed02d6130f93ae0ffdbedb63b760c0471"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "results/heldout")
    parser.add_argument("--output", type=Path, default=ROOT / "results/aya-independent-test-v1-local-audit.json")
    args = parser.parse_args()
    base = args.bundle.resolve()
    require(base.is_relative_to((ROOT / "results").resolve()), "Bundle must be inside project results")
    frozen = read_json(ROOT / "data/curated/aya-independent-test-v1.freeze.json")
    n = frozen["reviewed_kept"]
    require(n == 40 and frozen["all_candidates"] == 60, "Unexpected frozen cohort")
    for name, digest in frozen["sha256"].items():
        local = ROOT / name
        require(local.is_file() and sha(local) == digest, "Frozen local input changed: " + name)
        copied = base / name
        if copied.is_file():
            require(sha(copied) == digest, "Transferred input differs: " + name)

    facts = [json.loads(line) for line in (base / "data/curated/aya-independent-test-v1.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(facts) == n and all(f["split"] == "test" for f in facts),
            "Facts or test assignment changed")
    subjects = {f["subject_qid"] for f in facts}
    require(len(subjects) == n, "Duplicate test subject")
    pilot = {json.loads(line)["subject_qid"] for line in
             (ROOT / "data/curated/pilot-reviewed-20260926.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip()}
    require(not subjects & pilot, "Pilot subject overlap")

    screening = read_json(base / "data/curated/aya-independent-test-v1-screened.screening.json")
    require(screening["candidates"] == n, "English screen did not cover all facts")
    analysis = read_json(base / "results/analysis-aya-independent-test-v1.json")
    model = analysis["model"]
    require(model["model_id"] == "CohereLabs/aya-23-8B" and
            model["revision"] == REVISION and model["precision"] == "nf4" and
            model["compute_dtype"] == "fp16", "Model or precision differs")
    require(analysis["data_kind"] == "research", "Non-research analysis")
    require(analysis["observation_counts"] == {"E1": 9 * n, "E2": 12 * n},
            "E1/E2 coverage differs")
    orthography = analysis["orthography"]
    require({x["language"] for x in orthography} == {"he", "ar"},
            "Hebrew/Arabic comparison missing")
    for row in orthography:
        require(row["split"] == "test" and row["missing_pairs"] == 0 and
                row["n_pairs"] == 3 * n and
                row["accuracy"]["n_subjects"] == n and
                analysis["bootstrap_requested"] == 10000 and
                row["accuracy"]["bootstrap_replicates"] >= 8000,
                "Paired estimate incomplete: " + row["language"])
    paths = analysis["run_paths"]
    require(len(paths) == 4, "Expected two E1 and two E2 shards")
    counts = Counter()
    protocol_hashes = set()
    for relative in paths:
        run = (base / relative).resolve()
        require(run.is_relative_to(base) and run.is_dir(), "Unsafe or missing run path")
        completion = read_json(run / "completion.json")
        manifest = read_json(run / "manifest.json")
        experiment = manifest["experiment"]
        require(experiment in {"E1", "E2"} and manifest["split"] == "test" and
                completion["status"] == "complete" and
                completion["total_items"] == len(list((run / "items").glob("*.json"))),
                "Shard completion differs: " + relative)
        require(manifest["model"]["revision"] == REVISION and
                manifest["model"]["precision"] == "nf4", "Shard model differs")
        protocol_hashes.add(manifest["protocol_hash"])
        counts[experiment] += completion["total_items"]
    require(counts == Counter({"E1": 9 * n, "E2": 12 * n}) and
            len(protocol_hashes) == 1, "Shard counts or protocol hashes differ")
    protocol = read_json(base / "results/protocol-aya-independent-test-v1.json")
    require(protocol["protocol_hash"] in protocol_hashes, "Frozen protocol hash differs")
    result = {
        "status": "passed", "cohort": frozen["cohort"], "facts": n,
        "subjects": len(subjects), "pilot_subject_overlap": 0,
        "screened": screening["candidates"], "model": model,
        "protocol_hash": protocol["protocol_hash"],
        "observations": dict(counts),
        "orthography": orthography,
        "analysis_sha256": sha(base / "results/analysis-aya-independent-test-v1.json"),
    }
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / "results").resolve()), "Output must be inside results")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "facts", "subjects", "observations",
                                             "analysis_sha256")}, indent=2))


if __name__ == "__main__":
    main()

"""Source review, deterministic subject splits, alias registry, and pair review."""
import collections
import csv
import io
import json
import re
from pathlib import Path

from .io import atomic_text, digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .unicode import nfc, normalize_answer, strip_marks, validate_pair

LANGUAGES = ("en", "he", "ar")


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, "\ufeff" + stream.getvalue())


def review_export(facts_path, output_dir):
    facts = read_jsonl(facts_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, pairs = [], []
    for f in facts:
        row = {"fact_id": f["fact_id"], "subject_qid": f["subject_qid"], "relation": f["relation"],
               "object_qid": f["object_qid"], "single_answer_agrees": f.get("wikidata", {}).get("single_answer_agrees"),
               "decision": "pending", "reviewer": "", "pre_release_verified": "", "evidence_url": "", "notes": ""}
        for lang in LANGUAGES:
            row[f"subject_{lang}"] = f["subject_labels"][lang]
            row[f"object_{lang}"] = f["object_labels"][lang]
            row[f"aliases_{lang}"] = json.dumps(f.get("object_aliases", {}).get(lang, [f["object_labels"][lang]]), ensure_ascii=False)
            if lang != "en":
                pairs.append({"fact_id": f["fact_id"], "language": lang,
                    "U": strip_marks(f["subject_labels"][lang], lang), "D": "", "P": "",
                    "decision": "pending", "reviewer": "", "notes": ""})
        rows.append(row)
    if rows:
        if (out / "facts.csv").exists() or (out / "pairs.csv").exists():
            raise FileExistsError("Review files already exist; choose a new directory to avoid overwriting human work")
        write_csv(out / "facts.csv", rows, list(rows[0]))
        write_csv(out / "pairs.csv", pairs, list(pairs[0]))
    return {"facts": len(rows), "pair_rows": len(pairs), "directory": str(out)}


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def assign_splits(facts, seed=17, ratios=(1 / 6, 1 / 6, 2 / 3)):
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-9 or min(ratios) < 0:
        raise ValueError("Invalid split ratios")
    groups = collections.defaultdict(list)
    for f in facts:
        groups[f["subject_qid"]].append(f)
    strata = collections.defaultdict(list)
    for subject, rows in groups.items():
        strata[tuple(sorted({r["relation"] for r in rows}))].append(subject)
    split = {}
    for subjects in strata.values():
        subjects.sort(key=lambda q: digest([seed, q]))
        n = len(subjects)
        a, b = round(n * ratios[0]), round(n * (ratios[0] + ratios[1]))
        for i, subject in enumerate(subjects):
            split[subject] = "pilot" if i < a else "dev" if i < b else "test"
    return [{**f, "split": split[f["subject_qid"]]} for f in facts]


def create_split_manifest(facts_path, output, seed=17):
    if Path(output).exists():
        raise FileExistsError('Split assignments already exist; do not replace an inspected split')
    facts = assign_splits(read_jsonl(facts_path), seed)
    record = {'schema_version': 1, 'seed': seed, 'source_sha256': file_hash(facts_path),
              'method': 'subject-grouped relation-stratified hash rank on complete candidate pool',
              'subjects': {f['subject_qid']: f['split'] for f in facts}}
    record['manifest_hash'] = digest(record)
    write_json(output, record)
    return {'subjects': len(record['subjects']), 'counts': counts(facts), 'output': str(output)}


def apply_split_manifest(facts_path, split_manifest, seed=17):
    record = read_json(split_manifest)
    expected_hash = record.pop('manifest_hash')
    if digest(record) != expected_hash or record['seed'] != seed:
        raise ValueError('Split manifest changed or uses a different seed')
    if record['source_sha256'] != file_hash(facts_path):
        raise ValueError('Candidate pool changed after split assignment')
    facts = read_jsonl(facts_path)
    if set(record['subjects']) != {f['subject_qid'] for f in facts}:
        raise ValueError('Split manifest does not cover the exact candidate pool')
    if any(s not in ('pilot', 'dev', 'test') for s in record['subjects'].values()):
        raise ValueError('Invalid split name')
    return [{**f, 'split': record['subjects'][f['subject_qid']]} for f in facts]


def prepare(facts_path, reviews_path, pairs_path, output, seed=17, split_manifest=None):
    # Approval filtering must not move an inspected pilot subject into test.
    facts = (apply_split_manifest(facts_path, split_manifest, seed) if split_manifest
             else assign_splits(read_jsonl(facts_path), seed))
    reviews = csv_rows(reviews_path)
    if len({r["fact_id"] for r in reviews}) != len(reviews):
        raise ValueError("Duplicate review fact IDs")
    by_id = {r["fact_id"]: r for r in reviews}
    accepted = []
    for f in facts:
        r = by_id.get(f["fact_id"], {})
        if r.get("decision") != "approved":
            continue
        if not r.get("reviewer", "").strip() or r.get("pre_release_verified", "").lower() != "true" or not r.get("evidence_url"):
            raise ValueError(f"Approved fact lacks reviewer/date evidence: {f['fact_id']}")
        if f.get("wikidata", {}).get("single_answer_agrees") is not True:
            raise ValueError(f"Resolve conflicting/missing Wikidata statement before approval: {f['fact_id']}")
        f["review"] = {"status": "approved", "reviewer": r["reviewer"], "pre_release_verified": True,
                       "evidence_url": r["evidence_url"], "notes": r.get("notes", "")}
        f["object_aliases"] = {}
        f["original_subject_labels"] = dict(f["subject_labels"])
        for lang in LANGUAGES:
            f["subject_labels"][lang] = nfc(r[f"subject_{lang}"].strip())
            f["object_labels"][lang] = nfc(r[f"object_{lang}"].strip())
            values = json.loads(r[f"aliases_{lang}"])
            if not isinstance(values, list) or not values or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError("Aliases must be a nonempty JSON array of strings")
            f["object_aliases"][lang] = sorted(set([nfc(v) for v in values] + [f["object_labels"][lang]]))
        f["pairs"] = {}
        accepted.append(f)
    if not accepted:
        raise ValueError("No approved facts. Complete data/review/facts.csv first")
    approved = {f["fact_id"]: f for f in accepted}
    seen = set()
    for r in csv_rows(pairs_path):
        key = (r["fact_id"], r["language"])
        if key in seen:
            raise ValueError("Duplicate pair annotation")
        seen.add(key)
        if r.get("decision") != "approved" or r["fact_id"] not in approved:
            continue
        lang = r["language"]
        if lang not in ("he", "ar") or not r.get("reviewer", "").strip():
            raise ValueError("Pair needs a target language and a named qualified reviewer")
        u, d = validate_pair(r["U"], r["D"], lang)
        if u != approved[r["fact_id"]]["subject_labels"][lang]:
            raise ValueError("Reviewed U must equal the canonical subject label; edit the fact review consistently")
        pair = {"U": u, "D": d, "reviewer": r["reviewer"], "status": "approved", "notes": r.get("notes", "")}
        if r.get("P"):
            _, partial = validate_pair(u, r["P"], lang)
            # A reviewed partial form must remove marks only, preserving their order.
            it = iter(d)
            if not all(any(x == c for x in it) for c in partial):
                raise ValueError("P must be a subsequence of D")
            pair["P"] = partial
        approved[r["fact_id"]]["pairs"][lang] = pair
    validate_facts(accepted, require_review=True)
    write_jsonl(output, accepted)
    report = counts(accepted)
    write_json(Path(output).with_suffix(".summary.json"), report)
    return report


def counts(facts):
    return {"facts": len(facts), "subjects": len({f["subject_qid"] for f in facts}),
        "objects": len({f["object_qid"] for f in facts}),
        "by_split": dict(collections.Counter(f.get("split", "unassigned") for f in facts)),
        "by_relation": dict(collections.Counter(f["relation"] for f in facts)),
        "pairs": {l: dict(collections.Counter(f.get("split", "unassigned") for f in facts if l in f.get("pairs", {}))) for l in ("he", "ar")}}


def validate_facts(facts, require_review=False):
    if not facts:
        raise ValueError("Empty dataset")
    ids, subjects, answers = set(), {}, collections.defaultdict(set)
    for f in facts:
        if f["fact_id"] in ids:
            raise ValueError("Duplicate fact ID")
        ids.add(f["fact_id"])
        for field in ("subject_qid", "object_qid"):
            if not re.fullmatch(r"Q[1-9][0-9]*", f[field]):
                raise ValueError("Invalid QID")
        s = f.get("split")
        if s not in ("pilot", "dev", "test"):
            raise ValueError("Missing/invalid split")
        if f["subject_qid"] in subjects and subjects[f["subject_qid"]] != s:
            raise ValueError("Subject leakage across splits")
        subjects[f["subject_qid"]] = s
        answers[f["subject_qid"], f["relation"]].add(f["object_qid"])
        for lang in LANGUAGES:
            for label in ("subject_labels", "object_labels"):
                value = f[label][lang]
                if not value or value != nfc(value) or value != value.strip():
                    raise ValueError("Labels must be nonempty, trimmed NFC")
            if not f.get("object_aliases", {}).get(lang):
                raise ValueError("Missing answer aliases")
        if require_review and (f.get("review", {}).get("status") != "approved" or
                               not f["review"].get("reviewer") or not f["review"].get("pre_release_verified")):
            raise ValueError("Fact is not historically/source reviewed")
        for lang, pair in f.get("pairs", {}).items():
            validate_pair(pair["U"], pair["D"], lang)
            if pair["U"] != f["subject_labels"][lang] or pair.get("status") != "approved" or not pair.get("reviewer"):
                raise ValueError("Pair approval/canonical U mismatch")
    if any(len(v) != 1 for v in answers.values()):
        raise ValueError("Multiple gold objects for a subject/relation")
    return counts(facts)


def alias_registry(facts, mode="all"):
    if mode not in ("all", "canonical"):
        raise ValueError("Unknown answer registry mode")
    registry = collections.defaultdict(lambda: collections.defaultdict(set))
    for f in facts:
        for lang, values in f["object_aliases"].items():
            if mode == "canonical":
                values = [f["object_labels"][lang]]
            for alias in values:
                registry[normalize_answer(alias)][f["object_qid"]].add(lang)
    return {alias: {q: sorted(langs) for q, langs in entities.items()} for alias, entities in registry.items()}

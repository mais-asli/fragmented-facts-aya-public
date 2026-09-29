"""Read original mLAMA, cache Wikidata, and retain all source provenance."""
import collections
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .io import digest, file_hash, now, read_json, read_jsonl, write_json, write_jsonl

MLAMA_URL = "https://cistern.cis.lmu.de/mlama/mlama1.1.zip"
LANGUAGES = ("en", "he", "ar")
RELATIONS = ("P19", "P20", "P159", "P740")
USER_AGENT = "FragmentedFactsCourseResearch/0.1 (educational dataset audit; low-rate cached requests)"


def open_url(request, timeout=90):
    # Use a current trust store when present, always retaining certificate verification.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def fetch_archive(target, url=MLAMA_URL, max_bytes=700_000_000):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = target.with_suffix(target.suffix + ".source.json")
    if target.exists():
        if not meta.exists() or read_json(meta)["sha256"] != file_hash(target):
            raise ValueError("Existing archive lacks valid provenance/checksum")
        return read_json(meta)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = target.with_suffix(target.suffix + ".partial")
    with open_url(req) as response, partial.open("wb") as stream:
        total = 0
        for block in iter(lambda: response.read(1024 * 1024), b""):
            total += len(block)
            if total > max_bytes:
                raise ValueError("Download exceeded the configured size limit")
            stream.write(block)
        resolved_url = response.url
    if not zipfile.is_zipfile(partial):
        raise ValueError("Downloaded content is not a ZIP archive")
    partial.replace(target)
    result = {"url": url, "resolved_url": resolved_url, "retrieved_at": now(),
              "sha256": file_hash(target), "bytes": total, "license": "mLAMA CC-BY-NC-4.0"}
    write_json(meta, result)
    return result


def qid(value):
    value = str(value).rsplit("/", 1)[-1]
    if not value.startswith("Q") or not value[1:].isdigit():
        raise ValueError(f"Not a Wikidata QID: {value!r}")
    return value


def import_mlama(archive, output, limit=1800, relations=RELATIONS, seed=17, require_aligned=True):
    """Join by actual QIDs, never assume line numbers align across languages."""
    joined = {}
    counts = collections.Counter()
    archive_sha = file_hash(archive)
    with zipfile.ZipFile(archive) as z:
        for lang in LANGUAGES:
            for relation in relations:
                paths = [name for name in z.namelist() if name.endswith(f"/{lang}/{relation}.jsonl")]
                if len(paths) != 1:
                    counts[f"missing_or_duplicate_file:{lang}:{relation}"] += 1
                    continue
                with z.open(paths[0]) as stream:
                    for raw in stream:
                        row = json.loads(raw)
                        try:
                            s = qid(row.get("sub_uri", row.get("sub_id", "")))
                            o = qid(row.get("obj_uri", row.get("obj_id", "")))
                        except ValueError:
                            counts["invalid_qid"] += 1
                            continue
                        key = (s, relation, o)
                        fact = joined.setdefault(key, {"fact_id": "-".join(key), "subject_qid": s,
                            "relation": relation, "object_qid": o, "subject_labels": {}, "object_labels": {},
                            "source": {"dataset": "mLAMA-1.1", "archive_sha256": archive_sha, "records": []},
                            "review": {"status": "pending", "reviewer": "", "pre_release_verified": False},
                            "data_kind": "research"})
                        fact["subject_labels"][lang] = row["sub_label"]
                        fact["object_labels"][lang] = row["obj_label"]
                        fact["source"]["records"].append({"path": paths[0], "lineid": row.get("lineid")})
                        counts[f"rows:{lang}:{relation}"] += 1
    objects = collections.defaultdict(set)
    for s, r, o in joined:
        objects[s, r].add(o)
    valid = []
    for (s, r, _), f in joined.items():
        if len(objects[s, r]) != 1:
            counts["multiple_objects"] += 1
        elif not all(f["subject_labels"].get(l) and f["object_labels"].get(l) for l in (LANGUAGES if require_aligned else ("en",))):
            counts["missing_language_label"] += 1
        else:
            valid.append(f)
    # Round-robin relation strata, deterministic order, at most one fact per subject.
    by_rel = {r: sorted([f for f in valid if f["relation"] == r],
                        key=lambda f: (all(f["subject_labels"].get(l) and f["object_labels"].get(l) for l in LANGUAGES),
                                       digest([seed, f["fact_id"]]))) for r in relations}
    selected, seen = [], set()
    while len(selected) < limit and any(by_rel.values()):
        for r in relations:
            while by_rel[r]:
                f = by_rel[r].pop()
                if f["subject_qid"] not in seen:
                    seen.add(f["subject_qid"])
                    selected.append(f)
                    break
            if len(selected) == limit:
                break
    write_jsonl(output, selected)
    report = {"counts": dict(counts), "joined_facts": len(joined), "eligible_facts": len(valid),
              "selected": len(selected), "selected_by_relation": dict(collections.Counter(f["relation"] for f in selected)),
              "seed": seed, "require_aligned": require_aligned,
              "selection_policy": "relation round-robin; prioritize already aligned labels, then seeded order; one fact per subject",
              "archive_sha256": archive_sha, "status": "source_candidates_require_review"}
    write_json(Path(output).with_suffix(".flow.json"), report)
    return report


class WikidataCache:
    def __init__(self, root, pause=1.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pause = pause

    def fetch(self, ids):
        ids = sorted(set(qid(i) for i in ids))
        missing = [i for i in ids if not (self.root / f"{i}.json").exists()]
        for start in range(0, len(missing), 40):
            batch = missing[start:start + 40]
            query = urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(batch),
                "props": "labels|aliases|claims|sitelinks", "languages": "en|he|ar", "format": "json", "maxlag": "5"})
            url = "https://www.wikidata.org/w/api.php?" + query
            for attempt in range(10):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with open_url(req) as response:
                        payload = json.load(response)
                    if "error" in payload:
                        if payload["error"].get("code") == "maxlag" and attempt < 9:
                            print("Wikidata requested a pause due to server lag; waiting 60s before retry", flush=True)
                            time.sleep(60)
                            continue
                        raise RuntimeError(str(payload["error"]))
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 403:
                        # Documented public export, not an authentication workaround.
                        return self.fetch_exports(ids)
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                        raise
                    delay = max(60 if exc.code == 429 else 2 ** attempt,
                                int(exc.headers.get("Retry-After", "0")) if exc.headers.get("Retry-After", "0").isdigit() else 0)
                    print(f"Wikidata HTTP {exc.code}; respecting {delay}s backoff before retry", flush=True)
                    time.sleep(delay)
                except (urllib.error.URLError, RuntimeError):
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 8))
            for entity_id, entity in payload["entities"].items():
                write_json(self.root / f"{entity_id}.json", {"url": url, "retrieved_at": now(), "entity": entity})
            print(f"Wikidata cached: {min(start + 40, len(missing))}/{len(missing)} missing entities", flush=True)
            time.sleep(self.pause)
        return {i: read_json(self.root / f"{i}.json") for i in ids}

    def fetch_exports(self, ids):
        from concurrent.futures import ThreadPoolExecutor
        missing = [i for i in ids if not (self.root / f"{i}.json").exists()]

        def one(entity_id):
            url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
            for attempt in range(5):
                try:
                    with open_url(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
                        payload = json.load(response)
                    entity = payload["entities"].get(entity_id)
                    if entity is None:
                        # Redirected/merged entities must be manually resolved, not silently reidentified.
                        entity = {"id": entity_id, "missing": True, "redirect_candidates": list(payload["entities"])}
                    write_json(self.root / f"{entity_id}.json", {"url": url, "retrieved_at": now(), "entity": entity})
                    time.sleep(0.35)
                    return entity_id
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        write_json(self.root / f"{entity_id}.json", {"url": url, "retrieved_at": now(), "entity": {"id": entity_id, "missing": True}})
                        return entity_id
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                        raise
                    delay = max(60 if exc.code == 429 else 2 ** attempt,
                                int(exc.headers.get("Retry-After", "0")) if exc.headers.get("Retry-After", "0").isdigit() else 0)
                    time.sleep(delay)
                except urllib.error.URLError:
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 16))
        # Three workers, cached one-time downloads, bounded concurrency. Respect 429 backoff.
        with ThreadPoolExecutor(max_workers=3) as pool:
            for n, _ in enumerate(pool.map(one, missing), 1):
                if n % 25 == 0 or n == len(missing):
                    print(f"Wikidata entity exports cached: {n}/{len(missing)}", flush=True)
        return {i: read_json(self.root / f"{i}.json") for i in ids}


def enrich(facts_path, cache_path, output):
    facts = read_jsonl(facts_path)
    cache = WikidataCache(cache_path)
    all_entities = cache.fetch([f[k] for f in facts for k in ("subject_qid", "object_qid")])
    eligible, excluded = [], []
    for f in facts:
        subj = all_entities[f["subject_qid"]]
        obj = all_entities[f["object_qid"]]
        f["label_origins"] = {}
        for field, record in (("subject_labels", subj), ("object_labels", obj)):
            f["label_origins"][field] = {}
            for lang in LANGUAGES:
                if f[field].get(lang):
                    f["label_origins"][field][lang] = "mLAMA"
                else:
                    value = record["entity"].get("labels", {}).get(lang, {}).get("value")
                    if value:
                        f[field][lang] = value
                        f["label_origins"][field][lang] = "Wikidata_current_snapshot"
        if not all(f[field].get(l) for field in ("subject_labels", "object_labels") for l in LANGUAGES):
            excluded.append({"fact_id": f["fact_id"], "reason": "missing_label_after_enrichment"})
            continue
        statements = [s for s in subj["entity"].get("claims", {}).get(f["relation"], []) if s.get("rank") != "deprecated"]
        answers = {s.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") for s in statements
                   if isinstance(s.get("mainsnak", {}).get("datavalue", {}).get("value"), dict)}
        answers.discard(None)
        f["wikidata"] = {"subject_revision": subj["entity"].get("lastrevid"),
            "object_revision": obj["entity"].get("lastrevid"), "retrieved_at": subj["retrieved_at"],
            "single_answer_agrees": answers == {f["object_qid"]}, "current_answer_qids": sorted(answers),
            "statements": statements}
        f["popularity"] = {"sitelinks": len(subj["entity"].get("sitelinks", {})), "pageviews": {l: None for l in LANGUAGES}}
        f["sitelinks"] = {l: subj["entity"].get("sitelinks", {}).get(l + "wiki", {}).get("title") for l in LANGUAGES}
        f["object_aliases"] = {}
        for lang in LANGUAGES:
            aliases = [f["object_labels"][lang]]
            label = obj["entity"].get("labels", {}).get(lang, {}).get("value")
            if label:
                aliases.append(label)
            aliases += [a["value"] for a in obj["entity"].get("aliases", {}).get(lang, [])]
            f["object_aliases"][lang] = sorted(set(aliases))
        f["review"]["status"] = "pending"  # Public claims do not establish historical validity or native approval.
        eligible.append(f)
    write_jsonl(output, eligible)
    report = {"candidates": len(facts), "facts": len(eligible), "excluded": excluded,
              "single_answer_agrees": sum(f["wikidata"]["single_answer_agrees"] for f in eligible)}
    write_json(Path(output).with_suffix(".enrichment.json"), report)
    return {k: v for k, v in report.items() if k != "excluded"}


def pageviews(facts_path, output, cache_path, start="2023010100", end="2023123100"):
    facts = read_jsonl(facts_path)
    cache = Path(cache_path)
    cache.mkdir(parents=True, exist_ok=True)
    for f in facts:
        for lang in LANGUAGES:
            title = f.get("sitelinks", {}).get(lang)
            if not title:
                continue
            url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia.org/"
                   f"all-access/user/{urllib.parse.quote(title.replace(' ', '_'), safe='')}/monthly/{start}/{end}")
            path = cache / (digest(url) + ".json")
            if not path.exists():
                try:
                    with open_url(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
                        payload = json.load(r)
                    record = {"url": url, "retrieved_at": now(), "payload": payload}
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                    record = {"url": url, "retrieved_at": now(), "missing": True}
                write_json(path, record)
                time.sleep(0.2)
            record = read_json(path)
            f["popularity"]["pageviews"][lang] = (None if record.get("missing") else
                sum(v["views"] for v in record["payload"]["items"]))
            f.setdefault("pageviews_source", {})[lang] = {"cache_sha256": file_hash(path), "start": start, "end": end}
    write_jsonl(output, facts)

"""Deterministic, resumable E1-E5 orchestration with explicit control conditions."""
import collections
from pathlib import Path

from .data import alias_registry
from .hooks import Patch
from .io import ResultStore, code_hash, digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .prompts import encode_prompt, paired_prompts, template_for
from .scoring import evaluate_answer
from .tokenization import split_token_alternative


def stable_order(rows, seed, salt):
    return sorted(rows, key=lambda f: digest([seed, salt, f["fact_id"]]))


def choose_donor(fact, facts, language, seed, tokenizer=None):
    candidates = [f for f in facts if f["relation"] == fact["relation"] and f["object_qid"] != fact["object_qid"]
                  and f["subject_qid"] != fact["subject_qid"] and f["split"] == fact["split"]]
    if not candidates:
        return None
    def size(f):
        text = f["subject_labels"][language]
        return len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer else len(text)
    target_size = size(fact)
    return min(candidates, key=lambda f: (abs(size(f) - target_size), digest([seed, fact["fact_id"], f["fact_id"]])))


def fact_record(f):
    return {k: f[k] for k in ("fact_id", "subject_qid", "object_qid", "relation", "split", "source", "data_kind")}


def observation(runner, prompt, fact, registry, gold, distractor=None, patches=(), generate=True,
                broad_registry=None):
    result = {"prompt": prompt.record(), "gold_score": runner.score(prompt, gold, patches),
              "gold_aliases": fact["object_aliases"]}
    if distractor:
        result["distractor_score"] = runner.score(prompt, distractor, patches)
        result["likelihood_margin"] = result["gold_score"]["sum_logprob"] - result["distractor_score"]["sum_logprob"]
    if generate:
        generated = runner.generate(prompt, patches)
        result["generation"] = generated
        result["evaluation"] = evaluate_answer(generated["text"], fact["object_qid"], prompt.language, registry, generated["truncated"])
        if broad_registry is not None:
            result["evaluation_all_aliases"] = evaluate_answer(
                generated["text"], fact["object_qid"], prompt.language, broad_registry, generated["truncated"])
    return result


def run_experiment(runner, facts, templates, config, protocol, experiment, split, output,
                   shard=0, num_shards=1, limit=None):
    if experiment not in ("E1", "E2", "E3", "E4", "E5", "diagnostics"):
        raise ValueError("Unknown experiment")
    if not 0 <= shard < num_shards or num_shards < 1:
        raise ValueError("Invalid shard configuration")
    if split == "test" and not protocol:
        raise ValueError("Test inference requires a verified frozen protocol")
    if experiment == "E5" and split == "test" and protocol["stage"] != "mechanism":
        raise ValueError("Freeze layer/window choices before E5 test inference")
    if experiment == "E4" and split == "test" and not config.get("readout_layers_frozen"):
        raise ValueError("E4 localization is development-only; freeze readout layers for held-out readout evaluation")
    registry = alias_registry(facts, mode="canonical")
    broad_registry = alias_registry(facts, mode="all")
    population = config.get("population", "core")
    if population not in ("core", "unfiltered", "one_of_two"):
        raise ValueError("Unknown analysis population")
    def in_population(f):
        if population == "unfiltered":
            return f.get("sensitivity_member", False)
        if population == "one_of_two":
            return f.get("english_one_of_two", False)
        return f.get("english_eligible", True) and f.get("core_member", True)
    eligible = [f for f in facts if f["split"] == split and in_population(f)]
    if not protocol.get("fixture") and any("english_eligible" not in f for f in eligible):
        raise ValueError("Run English screening and attach eligibility before research experiments")
    if not protocol.get("fixture") and any(f.get("screen_model_hash") != digest(runner.identity) for f in eligible):
        raise ValueError("Screening used a different model/precision; re-screen or define a documented common subset")
    selected = stable_order(eligible, config["seed"], experiment)
    if limit is not None:
        if split == "test":
            raise ValueError("Ad-hoc --limit is disabled on test; freeze sample limits in the config")
        selected = selected[:limit]
    languages = config.get("languages", ["en", "he", "ar"])
    if experiment in ("E2", "E5"):
        languages = [l for l in languages if l != "en"]
    manifest = {"schema_version": 1, "model": runner.identity, "experiment": experiment, "split": split,
        "protocol_hash": protocol["protocol_hash"], "config": config, "code_hash": code_hash(),
        "data_hash": digest(facts), "templates_hash": digest(templates), "shard": shard,
        "num_shards": num_shards, "limit": limit, "data_kind": "synthetic_fixture" if protocol.get("fixture") else "research",
        "primary_answer_policy": "exact_reviewed_canonical_name_after_conservative_normalization",
        "sensitivity_answer_policy": "all_source_aliases"}
    completed, skipped, ineligible, expected = 0, 0, [], []
    with ResultStore(output, manifest) as store:
        for language in languages:
            language_rows = [f for f in selected if experiment not in ("E2", "E5") or language in f.get("pairs", {})]
            cap = config.get("sample_limits", {}).get(experiment, {}).get(split)
            if cap is not None:
                language_rows = language_rows[:cap]
            for fact in language_rows:
                if int(digest(fact["subject_qid"]), 16) % num_shards != shard:
                    continue
                tids = list(templates["evaluation"][fact["relation"]][language])
                if experiment in ("E3", "E4", "E5", "diagnostics"):
                    tids = [config.get("mechanism_template", "t1")]
                for tid in tids:
                    template = template_for(templates, fact["relation"], language, tid)
                    prompt = encode_prompt(runner.tokenizer, template, fact["subject_labels"][language], language)
                    donor = choose_donor(fact, eligible, language, config["seed"], runner.tokenizer)
                    gold = fact["object_labels"][language]
                    bad = donor["object_labels"][language] if donor else None
                    base_key = {"experiment": experiment, "fact_id": fact["fact_id"], "language": language,
                                "template": tid, "seed": config["seed"]}

                    def save(condition, encoded, patches=(), extra=None, generation=True):
                        nonlocal completed, skipped
                        key = {**base_key, **condition}
                        expected.append(digest(key))
                        if store.get(key):
                            skipped += 1
                            return
                        value = observation(runner, encoded, fact, registry, gold,
                                            None if experiment == "E3" else bad, patches, generation,
                                            broad_registry=broad_registry)
                        payload = {"fact": fact_record(fact), "popularity": fact.get("popularity", {}),
                                   **value, **(extra or {}), "runtime": runner.runtime()}
                        store.put(key, payload)
                        completed += 1

                    if experiment == "E1":
                        save({"variant": "U"}, prompt)
                    elif experiment == "E2":
                        pair = fact["pairs"][language]
                        u, d = paired_prompts(runner.tokenizer, template, pair, language)
                        save({"variant": "U"}, u)
                        save({"variant": "D"}, d)
                        if config.get("partial_diacritics") and "P" in pair:
                            save({"variant": "P"}, encode_prompt(runner.tokenizer, template, pair["P"], language))
                        # Normalization is U exactly, recorded as an invariant, not independent evidence.
                        if encode_prompt(runner.tokenizer, template, pair["U"], language).input_ids != u.input_ids:
                            raise AssertionError("Normalization prompt mismatch")
                    elif experiment == "E3":
                        # This is a teacher-forced same-string likelihood control;
                        # E1/E2 already provide generated-answer behavior.
                        save({"variant": "canonical"}, prompt, generation=False)
                        for extra in config.get("segmentation_extra_tokens", [1, 2]):
                            alt = split_token_alternative(runner.tokenizer, prompt, max_extra=extra)
                            control = split_token_alternative(runner.tokenizer, prompt, outside=True, max_extra=extra)
                            if alt is None:
                                ineligible.append({**base_key, "extra_tokens": extra, "reason": "subject_split_unavailable"})
                            else:
                                save({"variant": "subject_split", "extra_tokens": extra}, alt, generation=False)
                            if control is None:
                                ineligible.append({**base_key, "extra_tokens": extra, "reason": "pre_subject_split_unavailable"})
                            else:
                                save({"variant": "outside_split", "extra_tokens": extra}, control, generation=False)
                    elif experiment == "E4":
                        if not donor:
                            ineligible.append({**base_key, "reason": "no_same_relation_donor"})
                            continue
                        layers = config.get("readout_layers") or list(range(0, len(runner.model.model.layers), 4))
                        corrupt = encode_prompt(runner.tokenizer, template, donor["subject_labels"][language], language)
                        for layer in layers:
                            for site in ("first_subject", "last_subject", "prediction"):
                                key = {**base_key, "variant": "layer_readout", "layer": layer, "site": site}
                                expected.append(digest(key))
                                if store.get(key):
                                    skipped += 1
                                    continue
                                readout = runner.readout(prompt, layer, site, gold, bad)
                                en_readout = runner.readout(prompt, layer, site, fact["object_labels"]["en"], donor["object_labels"]["en"])
                                # Causal localization: restore correct subject vector into an unrelated-subject prompt.
                                request = (layer, "residual", prompt.position(site))
                                vector = runner.capture(prompt, [request])[request]
                                patches = [Patch(layer, "residual", corrupt.position(site), vector)]
                                baseline = runner.score(corrupt, gold)
                                patched = runner.score(corrupt, gold, patches)
                                store.put(key, {"fact": fact_record(fact), "prompt": prompt.record(), "readout": readout,
                                    "english_label_readout": en_readout, "donor_fact_id": donor["fact_id"],
                                    "corrupt_score": baseline, "restored_score": patched,
                                    "restoration_gain": patched["sum_logprob"] - baseline["sum_logprob"], "runtime": runner.runtime()})
                                completed += 1
                    elif experiment == "E5":
                        if not donor:
                            ineligible.append({**base_key, "reason": "no_same_relation_donor"})
                            continue
                        u, d = paired_prompts(runner.tokenizer, template, fact["pairs"][language], language)
                        unrelated = encode_prompt(runner.tokenizer, template, donor["subject_labels"][language], language)
                        save({"variant": "baseline_U"}, u)
                        save({"variant": "baseline_D"}, d)
                        conditions = [("same_entity", u, d, "last_subject"), ("identity", d, d, "last_subject"),
                                      ("unrelated", unrelated, d, "last_subject"), ("first_subject", u, d, "first_subject"),
                                      ("delimiter", u, d, "delimiter"), ("reverse", d, u, "last_subject")]
                        for window in config["windows"]:
                            for component in config.get("patch_components", ["residual"]):
                                for name, source, target, site in conditions:
                                    request = [(l, component, source.position(site)) for l in window["layers"]]
                                    vectors = runner.capture(source, request)
                                    patches = [Patch(l, component, target.position(site), vectors[l, component, source.position(site)]) for l in window["layers"]]
                                    save({"variant": name, "window": window["name"], "component": component}, target, patches,
                                        {"patch": {"layers": window["layers"], "site": site,
                                            "source_fact_id": donor["fact_id"] if name == "unrelated" else fact["fact_id"],
                                            "baseline_variant": "baseline_U" if name == "reverse" else "baseline_D"}},
                                        generation=config.get("patch_generate", True))
                        layer = config.get("late_prediction_layer", len(runner.model.model.layers) - 1)
                        request = (layer, "residual", u.position("prediction"))
                        vector = runner.capture(u, [request])[request]
                        save({"variant": "late_prediction", "layer": layer}, d,
                            [Patch(layer, "residual", d.position("prediction"), vector)])
                    else:
                        # Explicit open-book name translation; never mixed with factual recall accuracy.
                        instruction = {"he": "Translate this place name into Hebrew. Give only its name: ",
                                       "ar": "Translate this place name into Arabic. Give only its name: ",
                                       "en": "Give only this place name in English: "}[language]
                        translation = encode_prompt(runner.tokenizer, instruction + "{subject}", fact["object_labels"]["en"], language)
                        save({"variant": "object_translation_open_book"}, translation,
                             extra={"diagnostic": "object label provided; not closed-book recall"})
                        english_answer = encode_prompt(runner.tokenizer, template + "\nGive the answer in English only.", fact["subject_labels"][language], language)
                        gold, bad = fact["object_labels"]["en"], donor["object_labels"]["en"] if donor else None
                        save({"variant": "request_english_answer"}, english_answer)
        summary = {"run_id": store.run_id, "run_path": str(store.path), "experiment": experiment, "split": split,
                   "completed_now": completed, "resumed_items": skipped, "total_items": len(store.rows()),
                   "expected_keys": sorted(set(expected)), "expected_count": len(set(expected)),
                   "ineligible": ineligible, "status": "complete", "data_kind": manifest["data_kind"]}
        write_json(store.path / "completion.json", summary)
    return {k: v for k, v in summary.items() if k not in ("expected_keys", "ineligible")}


def screen(runner, facts, templates, output, seed=17, num_shards=1, shard=0):
    registry = alias_registry(facts, mode="canonical")
    broad_registry = alias_registry(facts, mode="all")
    manifest = {"experiment": "screen", "model": runner.identity, "facts_hash": digest(facts),
        "templates_hash": digest(templates), "seed": seed, "shard": shard, "num_shards": num_shards,
        "primary_answer_policy": "exact_reviewed_canonical_name_after_conservative_normalization",
        "sensitivity_answer_policy": "all_source_aliases", "code_hash": code_hash()}
    with ResultStore(output, manifest) as store:
        for f in facts:
            if int(digest(f["subject_qid"]), 16) % num_shards != shard:
                continue
            for tid in templates["screen"][f["relation"]]["en"]:
                key = {"fact_id": f["fact_id"], "template": tid}
                if store.get(key):
                    continue
                prompt = encode_prompt(runner.tokenizer, template_for(templates, f["relation"], "en", tid, "screen"), f["subject_labels"]["en"], "en")
                store.put(key, {"fact": fact_record(f), **observation(
                    runner, prompt, f, registry, f["object_labels"]["en"], broad_registry=broad_registry),
                    "runtime": runner.runtime()})
        write_json(store.path / "completion.json", {"status": "complete", "experiment": "screen", "items": len(store.rows())})
    return str(store.path)


def attach_screen(facts_path, run_paths, output):
    facts = read_jsonl(facts_path)
    result, models, seen = collections.defaultdict(list), set(), set()
    for path in run_paths:
        manifest = read_json(Path(path) / "manifest.json")
        if manifest["experiment"] != "screen" or manifest["facts_hash"] != digest(facts):
            raise ValueError("Screening run does not match this dataset")
        models.add(digest(manifest["model"]))
        for p in (Path(path) / "items").glob("*.json"):
            row = read_json(p)
            key = (row["key"]["fact_id"], row["key"]["template"])
            if key in seen:
                raise ValueError("Duplicate screening observation")
            seen.add(key)
            result[key[0]].append(bool(row["evaluation"]["entity_correct"]))
    if len(models) != 1:
        raise ValueError("Screening mixed model/precision identities")
    for f in facts:
        scores = result[f["fact_id"]]
        if len(scores) != 2:
            raise ValueError(f"Incomplete two-template screen for {f['fact_id']}")
        f["english_eligible"] = all(scores)
        f["english_one_of_two"] = any(scores)
        f["screen_model_hash"] = next(iter(models))
        f["core_member"] = False
        f["sensitivity_member"] = False
    # Targets are caps, not claimed achieved sample sizes. Keep pre-screen subject splits.
    for split, cap in (("pilot", 150), ("dev", 150), ("test", 600)):
        candidates = [f for f in facts if f["split"] == split and f["english_eligible"]]
        by_relation = {r: stable_order([f for f in candidates if f["relation"] == r], 17, "core")
                       for r in sorted({f["relation"] for f in candidates})}
        picked, seen_subjects = 0, set()
        while picked < cap and any(by_relation.values()):
            for group in by_relation.values():
                if group and picked < cap:
                    f = group.pop()
                    if f["subject_qid"] not in seen_subjects:
                        seen_subjects.add(f["subject_qid"])
                        f["core_member"] = True
                        picked += 1
    for f in stable_order(facts, 17, "unfiltered_sensitivity")[:200]:
        f["sensitivity_member"] = True
    write_jsonl(output, facts)
    report = {"candidates": len(facts), "strict_eligible": sum(f["english_eligible"] for f in facts),
              "one_of_two": sum(f["english_one_of_two"] for f in facts),
              "core": dict(collections.Counter(f["split"] for f in facts if f["core_member"]))}
    write_json(Path(output).with_suffix(".screening.json"), report)
    return report

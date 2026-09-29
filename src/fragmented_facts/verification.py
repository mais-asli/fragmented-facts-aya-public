"""Actual software/model checks, with synthetic fixtures isolated from research."""
import importlib.util
import platform
import shutil
import time
from pathlib import Path

from .io import digest, file_hash, now, read_json, write_json, write_jsonl


def doctor(output):
    record = {"checked_at": now(), "python": platform.python_version(),
        "packages": {name: importlib.util.find_spec(name) is not None for name in
            ("torch", "transformers", "bitsandbytes", "accelerate", "numpy", "scipy", "sklearn", "matplotlib")},
        "nvidia_smi_available": bool(shutil.which("nvidia-smi")),
        "slurm_command_available": bool(shutil.which("sbatch")),
        "aya_snapshot_manifest_exists": Path("model_snapshot.json").exists(),
        "research_status": "not_run", "credentials_inspected": False}
    write_json(output, record)
    return record


def mechanical_checks(runner, donor, recipient):
    import torch
    from .hooks import Patch, logit_lens
    model = runner.model
    last = len(model.model.layers) - 1
    checks = []
    for component in ("residual", "mlp", "attention"):
        layer = min(1, last)
        pos = recipient.position("last_subject")
        key = (layer, component, pos)
        vector = runner.capture(recipient, [key])[key]
        expected = runner.logits(recipient.input_ids)[:, -1, :].float()
        actual = runner.logits(recipient.input_ids, [Patch(layer, component, pos, vector)])[:, -1, :].float()
        error = float((expected - actual).abs().max())
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
        checks.append({"name": f"identity_{component}", "passed": True, "max_logit_error": error})
    key = (last, "residual", donor.position("prediction"))
    vector = runner.capture(donor, [key])[key]
    donor_logits = runner.logits(donor.input_ids)[:, -1, :].float()
    with torch.inference_mode():
        reconstruction = logit_lens(model, vector)
    torch.testing.assert_close(reconstruction, donor_logits, atol=0.005, rtol=0.005)
    checks.append({"name": "final_norm_logit_scale", "passed": True})
    patch = Patch(last, "residual", recipient.position("prediction"), vector)
    actual = runner.logits(recipient.input_ids, [patch])[:, -1, :].float()
    torch.testing.assert_close(actual, donor_logits, atol=0.005, rtol=0.005)
    checks.append({"name": "final_prediction_positive_control", "passed": True})
    for module in model.modules():
        if module._forward_hooks:
            raise AssertionError("Forward hook leaked after a test")
    checks.append({"name": "all_hooks_removed", "passed": True})
    return checks


def workload(runner, facts, templates, output, limit=50):
    from .prompts import encode_prompt, template_for
    rows, candidates = [], []
    if limit < 50:
        raise ValueError("The real workload acceptance gate requires at least 50 pilot subjects")
    # Only pilot facts: no held-out outcomes can influence settings.
    for f in facts:
        if f["split"] != "pilot":
            continue
        for lang in ("en", "he", "ar"):
            template = template_for(templates, f["relation"], lang, "t1")
            for variant, subject in [("U", f["subject_labels"][lang])] + (
                [("D", f["pairs"][lang]["D"])] if lang in f.get("pairs", {}) else []):
                prompt = encode_prompt(runner.tokenizer, template, subject, lang)
                candidates.append((f, lang, variant, prompt))
    if len({f["subject_qid"] for f, _, _, _ in candidates}) < limit:
        raise ValueError("Workload gate requires 50 reviewed pilot subjects (reduce only in an explicitly revised protocol)")
    # Include the longest real pilot prompt and answer for each language/variant.
    candidates.sort(key=lambda row: -(len(row[3].input_ids) + len(runner.tokenizer.encode(row[0]["object_labels"][row[1]], add_special_tokens=False))))
    ids = []
    for f, _, _, _ in candidates:
        if f["subject_qid"] not in ids and len(ids) < limit:
            ids.append(f["subject_qid"])
    selected = [r for r in candidates if r[0]["subject_qid"] in ids]
    started = time.perf_counter()
    checks = mechanical_checks(runner, selected[0][3], selected[-1][3])
    for f, lang, variant, prompt in selected:
        t = time.perf_counter()
        score = runner.score(prompt, f["object_labels"][lang])
        score_seconds = time.perf_counter() - t
        generation = runner.generate(prompt)
        rows.append({"fact_id": f["fact_id"], "language": lang, "variant": variant,
            "input_tokens": len(prompt.input_ids), "score_seconds": score_seconds, "score": score, "generation": generation})
    report = {"status": "workload_passed", "revision": runner.identity["revision"], "precision": runner.identity["precision"],
        "subject_count": len(ids), "checks": checks, "seconds": time.perf_counter() - started,
        "max_input_tokens": max(len(r[3].input_ids) for r in selected), "examples": rows, "runtime": runner.runtime(),
        "next_action": "Inspect truncations and native output quality before freezing generation length and precision."}
    write_json(output, report)
    return {k: v for k, v in report.items() if k != "examples"}


def fixture_data():
    facts = []
    for i in range(12):
        labels = {"en": f"Entity Alpha{i}", "he": f"אלפא{i}", "ar": f"الفا{i}"}
        objects = {"en": f"Place Beta{i}", "he": f"עיר{i}", "ar": f"مدينة{i}"}
        facts.append({"fact_id": f"fixture-{i}", "subject_qid": f"Q{90000+i}", "object_qid": f"Q{91000+i}",
            "relation": "P19", "subject_labels": labels, "object_labels": objects,
            "object_aliases": {l: [s] for l, s in objects.items()}, "split": ("pilot", "dev", "test")[i // 4],
            "source": {"dataset": "synthetic_software_fixture", "warning": "These are invented strings, not asserted facts."},
            "data_kind": "synthetic_fixture", "english_eligible": True,
            "popularity": {"sitelinks": 10+i, "pageviews": {l: None for l in labels}},
            "review": {"status": "fixture_only", "reviewer": "software_fixture", "pre_release_verified": False},
            "pairs": {"he": {"U": labels["he"], "D": f"אַלְפָא{i}", "status": "approved", "reviewer": "software_fixture"},
                      "ar": {"U": labels["ar"], "D": f"اَلْفَا{i}", "status": "approved", "reviewer": "software_fixture"}}})
    return facts


def tiny_runner(facts, templates):
    """Use the real Transformers Cohere implementation with tiny random weights.

    The purpose is hook/scoring regression testing on a CPU. This is neither Aya
    loading evidence nor an alternative research model.
    """
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import CohereConfig, CohereForCausalLM, PreTrainedTokenizerFast
    from .model import ModelRunner
    torch.set_num_threads(1)
    torch.manual_seed(17)
    texts = []
    for f in facts:
        for lang in ("en", "he", "ar"):
            subjects = [f["subject_labels"][lang]] + ([f["pairs"][lang]["D"]] if lang in f["pairs"] else [])
            for subject in subjects:
                for t in templates["evaluation"][f["relation"]][lang].values():
                    texts.append(t["text"].replace("{subject}", subject))
            texts.append(f["object_labels"][lang])
    raw = Tokenizer(models.BPE(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<unk>", "<pad>", "<bos>", "<eos>", "<user>", "<end>", "<assistant>"])
    raw.train_from_iterator(texts, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
        additional_special_tokens=["<user>", "<end>", "<assistant>"])
    tok.chat_template = "{{ bos_token }}{% for message in messages %}<user>{{ message['content'] }}<end>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    config = CohereConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, attention_dropout=0.0, logit_scale=0.0625)
    config._attn_implementation = "eager"
    model = CohereForCausalLM(config)
    return ModelRunner(model, tok, {"model_id": "tiny_random_cohere", "revision": "seed-17", "precision": "fp32",
                                  "data_kind": "synthetic_fixture"}, max_new_tokens=3)


def integration(output):
    from .analysis import analyze_runs
    from .data import validate_facts
    from .experiments import run_experiment
    from .figures import make_figures
    from .prompts import encode_prompt, template_for
    from .protocol import freeze, verify
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    facts = fixture_data()
    templates = read_json("configs/templates.json")
    templates = {**templates, "evaluation": {"P19": templates["evaluation"]["P19"]}, "screen": {"P19": templates["screen"]["P19"]}}
    config = {"seed": 17, "model_kind": "tiny_random_cohere", "precision": "fp32", "max_new_tokens": 3,
        "languages": ["en", "he", "ar"], "mechanism_template": "t1", "patch_generate": True,
        "patch_components": ["residual", "mlp", "attention"], "sample_limits": {"E5": {"test": 2}},
        "readout_layers": [1, 3], "windows": [{"name": "early", "layers": [0, 1]}, {"name": "late", "layers": [2, 3]}],
        "primary_window": "early", "window_selection_reason": "Fixed mechanical fixture, not a scientific selection"}
    validate_facts(facts)
    runner = tiny_runner(facts, templates)
    donor = encode_prompt(runner.tokenizer, template_for(templates, "P19", "en", "t1"), facts[0]["subject_labels"]["en"], "en")
    recipient = encode_prompt(runner.tokenizer, template_for(templates, "P19", "he", "t1"), facts[1]["subject_labels"]["he"], "he")
    checks = mechanical_checks(runner, donor, recipient)
    write_json(out / "config.json", config)
    write_json(out / "templates.json", templates)
    write_jsonl(out / "facts.jsonl", facts)
    # Timestamped freeze prevents overwriting a previous integration run's protocol.
    stamp = str(time.time_ns())
    protocol_path = out / f"fixture-protocol-{stamp}.json"
    freeze(out / "config.json", out / "facts.jsonl", out / "templates.json", protocol_path, fixture=True)
    frozen = verify(protocol_path)
    runs = []
    for experiment, split in (("E1", "dev"), ("E1", "test"), ("E2", "test"), ("E3", "test"), ("E4", "dev")):
        result = run_experiment(runner, facts, templates, config, frozen, experiment, split, out / "runs")
        runs.append(result)
    mechanism_path = out / f"fixture-mechanism-{stamp}.json"
    freeze(out / "config.json", out / "facts.jsonl", out / "templates.json", mechanism_path,
           stage="mechanism", development_evidence=Path(runs[-1]["run_path"]) / "completion.json", fixture=True)
    result = run_experiment(runner, facts, templates, config, verify(mechanism_path), "E5", "test", out / "runs")
    runs.append(result)
    repeated = run_experiment(runner, facts, templates, config, verify(mechanism_path), "E5", "test", out / "runs")
    if repeated["completed_now"] != 0 or repeated["resumed_items"] != result["total_items"]:
        raise AssertionError("Resume did not skip completed observations")
    checks.append({"name": "resume_without_duplicate_results", "passed": True, "skipped": repeated["resumed_items"]})
    report = analyze_runs([r["run_path"] for r in runs], out / "analysis.json", repeats=200, allow_fixture=True)
    figures = make_figures(out / "analysis.json", out / "figures")
    verification = {"status": "passed", "data_kind": "synthetic_fixture", "warning": "SOFTWARE VERIFICATION ONLY: no Aya weights or research outcomes.",
        "checks": checks, "runs": runs, "figures": figures, "seconds": time.perf_counter() - started, "runtime": runner.runtime()}
    write_json(out / "verification.json", verification)
    return verification

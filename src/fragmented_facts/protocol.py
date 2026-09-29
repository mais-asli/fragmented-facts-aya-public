"""Freeze exact protocol inputs and reject changed test-time inputs."""
import re
from pathlib import Path

from .data import validate_facts
from .io import code_hash, digest, file_hash, now, read_json, read_jsonl, write_json
from .prompts import validate_templates


def freeze(config_path, facts_path, templates_path, output, snapshot_path=None, smoke_path=None,
           stage="behavior", development_evidence=None, fixture=False):
    config, facts, templates = read_json(config_path), read_jsonl(facts_path), read_json(templates_path)
    validate_facts(facts, require_review=not fixture)
    validate_templates(templates, require_review=not fixture)
    if fixture:
        if not all(f.get("data_kind") == "synthetic_fixture" for f in facts) or config.get("model_kind") != "tiny_random_cohere":
            raise ValueError("Fixture mode requires explicitly synthetic data and a tiny random model")
        identity = {"model_id": "tiny_random_cohere", "revision": "seed-17", "precision": "fp32"}
        evidence = {"status": "synthetic_software_test_only"}
    else:
        if any(f.get("data_kind") != "research" for f in facts):
            raise ValueError("Fixture records cannot enter a research protocol")
        if not snapshot_path or not smoke_path:
            raise ValueError("Freeze requires model snapshot and a successful Aya feasibility report")
        snapshot, smoke = read_json(snapshot_path), read_json(smoke_path)
        if not re.fullmatch("[0-9a-f]{40,64}", snapshot["revision"]):
            raise ValueError("An immutable model revision is required")
        if snapshot["model_id"] != "CohereLabs/aya-23-8B":
            raise ValueError("Aya-23-8B is the preregistered core model")
        if smoke.get("status") != "mechanical_checks_passed" or not all(c["passed"] for c in smoke["checks"]):
            raise ValueError("Aya feasibility checks have not passed")
        if smoke["model_source"]["revision"] != snapshot["revision"] or smoke["precision"] != config["precision"]:
            raise ValueError("Feasibility report used a different model revision/precision")
        expected_compute = config.get("nf4_compute_dtype", "fp16") if config["precision"] == "nf4" else config["precision"]
        if smoke.get("compute_dtype") != expected_compute:
            raise ValueError("Feasibility compute dtype differs from the frozen setting; rerun smoke with the explicit dtype")
        if not config.get("native_smoke_review", {}).get("reviewer") or not config.get("workload_pilot_report"):
            raise ValueError("Record native smoke review and a real workload pilot report before freezing")
        pilot = read_json(config["workload_pilot_report"])
        if pilot.get("status") != "workload_passed" or pilot.get("subject_count", 0) < 50 or pilot.get("revision") != snapshot["revision"] or pilot.get("precision") != config["precision"]:
            raise ValueError("Workload pilot does not match this model/precision or has not passed")
        identity = {k: snapshot[k] for k in ("model_id", "revision")}
        identity["precision"] = config["precision"]
        identity["compute_dtype"] = expected_compute
        evidence = {"smoke_sha256": file_hash(smoke_path), "workload_sha256": file_hash(config["workload_pilot_report"])}
    files = {"config": {"path": str(Path(config_path)), "sha256": file_hash(config_path)},
             "facts": {"path": str(Path(facts_path)), "sha256": file_hash(facts_path)},
             "templates": {"path": str(Path(templates_path)), "sha256": file_hash(templates_path)}}
    if stage == "mechanism":
        windows = config.get("windows", [])
        if not windows or config.get("primary_window") not in [w["name"] for w in windows]:
            raise ValueError("Freeze windows and a single primary window before E5 test")
        if len({w["name"] for w in windows}) != len(windows) or any(
            not w["layers"] or len(set(w["layers"])) != len(w["layers"]) or min(w["layers"]) < 0 for w in windows):
            raise ValueError("Window names and layers must be unique, valid indices")
        if not development_evidence or not config.get("window_selection_reason"):
            raise ValueError("Mechanism freeze requires development evidence and a selection rationale")
        selection = read_json(development_evidence)
        if selection.get("split") == "dev" and selection.get("experiment") == "E4":
            evidence["selection_type"] = "english_development_E4"
            evidence["development_sha256"] = file_hash(development_evidence)
            evidence["development_path"] = str(development_evidence)
        elif selection.get("split") == "pilot" and selection.get("experiment") == "E5":
            # Prospective replication of a window selected from *pilot* E5.
            # This is never represented as independent English E4 localization.
            if (config.get("window_selection_source") != "pilot_E5_exploratory"
                    or not selection.get("selected_before_test_E5")
                    or selection.get("primary_window") != config["primary_window"]
                    or selection.get("model_revision") != identity["revision"]):
                raise ValueError("Invalid pilot-selected replication declaration")
            pilot_path = selection.get("pilot_analysis_path")
            if not pilot_path or file_hash(pilot_path) != selection.get("pilot_analysis_sha256"):
                raise ValueError("Pilot analysis evidence is absent or changed")
            pilot = read_json(pilot_path)
            if (pilot.get("data_kind") != "research"
                    or pilot.get("model", {}).get("revision") != identity["revision"]
                    or not any(row.get("split") == "pilot"
                               and row.get("window") == config["primary_window"]
                               for row in pilot.get("patch_specificity", []))):
                raise ValueError("Pilot selection does not resolve to an actual E5 result")
            evidence["selection_type"] = "pilot_selected_E5_replication"
            evidence["selection_sha256"] = file_hash(development_evidence)
            evidence["selection_path"] = str(development_evidence)
            evidence["pilot_analysis_sha256"] = selection["pilot_analysis_sha256"]
        else:
            raise ValueError("Layer selection requires English E4 development data or declared pilot E5 evidence")
    content = {"schema_version": 1, "stage": stage, "fixture": fixture, "files": files,
               "model": identity, "evidence": evidence, "code_sha256": code_hash(), "created_at": now()}
    content["protocol_hash"] = digest(content)
    if Path(output).exists():
        raise FileExistsError("Frozen protocol already exists; create a new version explicitly")
    write_json(output, content)
    return content


def verify(protocol_path, config_path=None, facts_path=None, templates_path=None):
    record = read_json(protocol_path)
    h = record.pop("protocol_hash")
    if digest(record) != h:
        raise ValueError("Frozen protocol manifest was edited")
    if record["code_sha256"] != code_hash():
        raise ValueError("Code changed after freeze. Document the change and freeze a new version before continuing")
    paths = {"config": config_path, "facts": facts_path, "templates": templates_path}
    for key, item in record["files"].items():
        path = paths[key] or item["path"]
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Frozen {key} content changed")
        item["resolved_path"] = str(path)
    record["protocol_hash"] = h
    return record

import json
import zipfile
import pytest
from fragmented_facts.analysis import load_runs
from fragmented_facts.io import ResultStore, digest, read_jsonl, write_json
from fragmented_facts.sources import import_mlama


def test_source_join_uses_qids_not_line_numbers(tmp_path):
    archive = tmp_path / "mlama.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i, lang in enumerate(("en", "he", "ar")):
            rows = [{"sub_uri": "Q1", "obj_uri": "Q2", "sub_label": f"subject-{lang}", "obj_label": f"object-{lang}", "lineid": i}]
            z.writestr(f"mlama1.1/{lang}/P19.jsonl", json.dumps(rows[0]) + "\n")
    output = tmp_path / "facts.jsonl"
    report = import_mlama(archive, output, relations=["P19"])
    assert report["selected"] == 1
    assert set(read_jsonl(output)[0]["subject_labels"]) == {"en", "he", "ar"}


def test_unaligned_rows_never_receive_invented_labels(tmp_path):
    archive = tmp_path / "mlama.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("mlama1.1/en/P19.jsonl", json.dumps({"sub_uri": "Q1", "obj_uri": "Q2", "sub_label": "S", "obj_label": "O", "lineid": 0}) + "\n")
    output = tmp_path / "facts.jsonl"
    assert import_mlama(archive, output, relations=["P19"])["selected"] == 0
    assert import_mlama(archive, output, relations=["P19"], require_aligned=False)["selected"] == 1
    assert "ar" not in read_jsonl(output)[0]["subject_labels"]


def test_analysis_rejects_missing_items_and_synthetic_results(tmp_path):
    manifest = {"data_kind": "synthetic_fixture", "model": {}, "protocol_hash": "a",
                "data_hash": "a", "templates_hash": "b", "code_hash": "c", "config": {}}
    store = ResultStore(tmp_path, manifest)
    key = {"fact_id": "Q1"}
    store.put(key, {})
    write_json(store.path / "completion.json", {"status": "complete", "expected_keys": [digest(key), "missing"]})
    with pytest.raises(ValueError, match="Synthetic"):
        load_runs([store.path])
    with pytest.raises(ValueError, match="Missing"):
        load_runs([store.path], allow_fixture=True)

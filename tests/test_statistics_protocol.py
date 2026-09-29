import copy
import pytest
from fragmented_facts.analysis import cluster_effect, holm, relation_macro, sign_flip_p
from fragmented_facts.io import read_json, write_json, write_jsonl
from fragmented_facts.protocol import freeze, verify


def test_repeated_templates_do_not_inflate_subject_count():
    one = cluster_effect([1., 0., -1.], ["a", "b", "c"], ["r"] * 3, repeats=300)
    repeated = cluster_effect([1.] * 3 + [0.] * 3 + [-1.] * 3, ["a"] * 3 + ["b"] * 3 + ["c"] * 3, ["r"] * 9, repeats=300)
    assert one["n_subjects"] == repeated["n_subjects"] == 3
    assert one["estimate"] == repeated["estimate"]
    assert one["ci95"] == repeated["ci95"]


def test_relation_macro_not_dominated_by_large_relation():
    assert relation_macro([1.] * 9 + [0.], list(range(10)), ["a"] * 9 + ["b"]) == 0.5


def test_bootstrap_keeps_repeated_subject_draws_weighted():
    effect = cluster_effect([0., 1.], ["a", "b"], ["r", "r"], repeats=1000)
    assert effect["estimate"] == 0.5
    assert effect["ci95"] == [0., 1.]


def test_zero_effect_permutation_and_holm():
    assert sign_flip_p([0., 0.], ["a", "b"], ["r", "r"], repeats=100) == 1
    assert holm([0.01, 0.04, None]) == [0.02, 0.04, None]


def test_one_subject_has_no_confidence_interval():
    assert cluster_effect([1., 0.], ["a", "a"], ["r", "r"])["ci95"] is None


def test_nonfinite_data_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        cluster_effect([float("nan")], ["a"], ["r"])


def test_freeze_detects_changed_data_and_manifest(tmp_path, facts, templates):
    write_json(tmp_path / "config.json", {"model_kind": "tiny_random_cohere"})
    write_jsonl(tmp_path / "facts.jsonl", facts)
    write_json(tmp_path / "templates.json", templates)
    path = tmp_path / "protocol.json"
    freeze(tmp_path / "config.json", tmp_path / "facts.jsonl", tmp_path / "templates.json", path, fixture=True)
    assert verify(path)["fixture"]
    changed = copy.deepcopy(facts)
    changed[0]["subject_labels"]["en"] = "Changed"
    write_jsonl(tmp_path / "facts.jsonl", changed)
    with pytest.raises(ValueError, match="facts content changed"):
        verify(path)
    data = read_json(path)
    data["fixture"] = False
    write_json(path, data)
    with pytest.raises(ValueError, match="manifest was edited"):
        verify(path)


def test_fixture_flag_cannot_bypass_research_review(tmp_path, facts, templates):
    facts = copy.deepcopy(facts)
    facts[0]["data_kind"] = "research"
    write_json(tmp_path / "config.json", {"model_kind": "tiny_random_cohere"})
    write_jsonl(tmp_path / "facts.jsonl", facts)
    write_json(tmp_path / "templates.json", templates)
    with pytest.raises(ValueError, match="synthetic data"):
        freeze(tmp_path / "config.json", tmp_path / "facts.jsonl", tmp_path / "templates.json", tmp_path / "protocol.json", fixture=True)

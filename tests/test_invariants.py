import copy
import pytest
from fragmented_facts.data import assign_splits, validate_facts
from fragmented_facts.io import ResultStore, write_json
from fragmented_facts.scoring import evaluate_answer
from fragmented_facts.unicode import normalize_answer, strip_marks, validate_pair


@pytest.mark.parametrize("language,u,d", [("he", "שלום", "שָׁלוֹם"), ("ar", "فرنسا", "فَرَنْسَا")])
def test_optional_diacritic_pairs(language, u, d):
    assert validate_pair(u, d, language) == (u, d)
    assert strip_marks(d, language) == u


def test_arabic_hamza_is_not_an_optional_vowel():
    with pytest.raises(ValueError, match="base characters"):
        validate_pair("احمد", "أَحْمَد", "ar")


def test_hebrew_full_defective_spelling_rejected():
    with pytest.raises(ValueError, match="base characters"):
        validate_pair("דויד", "דָּוִד", "he")


def test_hidden_characters_rejected():
    with pytest.raises(ValueError):
        validate_pair("שלום", "שָׁלוֹם\u200b", "he")


def test_normalization_preserves_internal_words():
    assert normalize_answer(' "New York City." ') == "new york city"
    assert normalize_answer("New York but not New York City") != "new york city"


def test_full_answer_and_language_separated():
    registry = {"new york city": {"Q60": ["en"]}, "ניו יורק": {"Q60": ["he"]}}
    assert not evaluate_answer("New", "Q60", "en", registry)["entity_correct"]
    assert not evaluate_answer("New York City and London", "Q60", "en", registry)["entity_correct"]
    correct = evaluate_answer("New York City", "Q60", "he", registry)
    assert correct["entity_correct"] and not correct["requested_language_correct"]
    assert evaluate_answer("ניו יורק", "Q60", "he", registry)["requested_language_correct"]


def test_alias_collision_is_not_guessed():
    registry = {"springfield": {"Q1": ["en"], "Q2": ["en"]}}
    result = evaluate_answer("Springfield", "Q1", "en", registry)
    assert result["category"] == "ambiguous_alias"
    assert not result["entity_correct"]


def test_abstention_and_truncation_are_distinct():
    assert evaluate_answer("I don't know", "Q1", "en", {})["category"] == "abstention"
    assert evaluate_answer("Place", "Q1", "en", {"place": {"Q1": ["en"]}}, True)["category"] == "malformed_truncated"


def test_split_determinism_and_group_integrity(facts):
    extra = {**facts[0], "fact_id": "other-relation", "relation": "P20"}
    a = assign_splits(facts + [extra])
    b = assign_splits(list(reversed(facts + [extra])))
    assert {f["fact_id"]: f["split"] for f in a} == {f["fact_id"]: f["split"] for f in b}
    group = [f["split"] for f in a if f["subject_qid"] == facts[0]["subject_qid"]]
    assert len(set(group)) == 1


def test_split_leakage_and_duplicate_answers_refused(facts):
    rows = copy.deepcopy(facts)
    rows.append({**rows[0], "fact_id": "leaked", "split": "test"})
    with pytest.raises(ValueError, match="leakage"):
        validate_facts(rows)
    rows[-1]["split"] = "pilot"
    rows[-1]["object_qid"] = "Q888888"
    with pytest.raises(ValueError, match="Multiple gold"):
        validate_facts(rows)


def test_fixture_cannot_claim_human_review(facts):
    with pytest.raises(ValueError, match="reviewed"):
        validate_facts(facts, require_review=True)


def test_atomic_store_and_resume(tmp_path):
    key = {"fact_id": "Q42", "variant": "U"}
    with ResultStore(tmp_path, {"revision": "abc"}) as store:
        assert store.get(key) is None
        store.put(key, {"value": 1})
        with pytest.raises(ValueError, match="Duplicate"):
            store.put(key, {"value": 2})
        with pytest.raises(RuntimeError, match="locked"):
            with ResultStore(tmp_path, {"revision": "abc"}):
                pass
    with ResultStore(tmp_path, {"revision": "abc"}) as resumed:
        assert resumed.get(key)["value"] == 1
        assert len(resumed.rows()) == 1
        (resumed.items / ".write-interrupted").write_text("incomplete")
        assert len(resumed.rows()) == 1


def test_invalid_existing_result_never_silently_skipped(tmp_path):
    from fragmented_facts.io import digest
    key = {"fact": "Q1"}
    store = ResultStore(tmp_path, {"revision": "a"})
    write_json(store.items / (digest(key) + ".json"), {"key": key, "status": "failed"})
    with pytest.raises(ValueError, match="Invalid existing"):
        store.get(key)

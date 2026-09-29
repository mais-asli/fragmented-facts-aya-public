from fragmented_facts.data import alias_registry
from fragmented_facts.scoring import evaluate_answer


def test_historical_alias_is_sensitivity_only():
    fact = {
        "object_qid": "Q1",
        "object_labels": {"en": "Rome", "he": "רומא", "ar": "روما"},
        "object_aliases": {
            "en": ["Rome", "Eternal City"], "he": ["רומא"], "ar": ["روما"]},
    }
    canonical = alias_registry([fact], mode="canonical")
    broad = alias_registry([fact], mode="all")
    assert evaluate_answer("Eternal City", "Q1", "en", canonical)["category"] == "wrong_assertion"
    assert evaluate_answer("Eternal City", "Q1", "en", broad)["entity_correct"]
    assert evaluate_answer("רומא", "Q1", "he", canonical)["requested_language_correct"]
    assert evaluate_answer("Rome", "Q1", "he", canonical)["entity_correct"]
    assert not evaluate_answer("Rome", "Q1", "he", canonical)["requested_language_correct"]

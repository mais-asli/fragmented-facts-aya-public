import pytest
import torch
from fragmented_facts.hooks import Patch, capture_context
from fragmented_facts.prompts import encode_prompt, paired_prompts, template_for
from fragmented_facts.scoring import continuation_ids, sequence_logprobs
from fragmented_facts.tokenization import split_token_alternative
from fragmented_facts.verification import mechanical_checks


@pytest.mark.parametrize("language", ["en", "he", "ar"])
def test_full_chat_and_subject_offsets(runner, facts, templates, language):
    f = facts[0]
    prompt = encode_prompt(runner.tokenizer, template_for(templates, "P19", language, "t1"), f["subject_labels"][language], language)
    assert prompt.subject_indices
    assert prompt.position("prediction") > prompt.position("delimiter") > prompt.position("last_subject")
    assert prompt.text[prompt.span[0]:prompt.span[1]] == f["subject_labels"][language]
    for i in prompt.subject_indices:
        a, b = prompt.offsets[i]
        assert b > prompt.span[0] and a < prompt.span[1]


@pytest.mark.parametrize("language", ["he", "ar"])
def test_paired_context_and_constant_denominator(runner, facts, templates, language):
    u, d = paired_prompts(runner.tokenizer, template_for(templates, "P19", language, "t1"), facts[0]["pairs"][language], language)
    assert u.record()["base_characters"] == d.record()["base_characters"]
    assert u.text[:u.span[0]] == d.text[:d.span[0]]
    assert u.text[u.span[1]:] == d.text[d.span[1]:]


def test_repeated_subject_is_rejected(runner, templates):
    with pytest.raises(ValueError, match="once"):
        encode_prompt(runner.tokenizer, "Entity {subject}", "Entity", "en")


def test_all_answer_tokens_are_scored():
    logits = torch.zeros(1, 5, 3)
    scored = sequence_logprobs(logits, 3, [1, 2])
    assert scored["sum_logprob"] == pytest.approx(-2 * torch.log(torch.tensor(3.)).item())
    logits[0, 3, 2] = -20
    changed = sequence_logprobs(logits, 3, [1, 2])
    assert changed["sum_logprob"] < scored["sum_logprob"] - 15


def test_shifted_teacher_forcing_indices():
    logits = torch.full((1, 6, 4), -100.)
    logits[0, 2, 1] = 0
    logits[0, 3, 2] = 0
    logits[0, 4, 3] = 0
    assert sequence_logprobs(logits, 3, [1, 2, 3])["sum_logprob"] == pytest.approx(0)


def test_cohere_identity_and_positive_controls(runner, facts, templates):
    prompts = [encode_prompt(runner.tokenizer, template_for(templates, "P19", "en", "t1"), f["subject_labels"]["en"], "en") for f in facts[:2]]
    checks = mechanical_checks(runner, *prompts)
    assert len(checks) == 6 and all(c["passed"] for c in checks)


def test_hooks_removed_even_on_exception(runner):
    patch = Patch(1, "mlp", 99999, torch.zeros(1, 32))
    with pytest.raises(ValueError):
        runner.logits([1, 2, 3], [patch])
    assert all(not m._forward_hooks for m in runner.model.modules())
    with pytest.raises(RuntimeError):
        with capture_context(runner.model, [(0, "residual", 0)]):
            raise RuntimeError("simulated failure")
    assert all(not m._forward_hooks for m in runner.model.modules())


def test_scoring_matches_direct_multitoken_likelihood(runner, facts, templates):
    f = facts[0]
    prompt = encode_prompt(runner.tokenizer, template_for(templates, "P19", "en", "t1"), f["subject_labels"]["en"], "en")
    answer = f["object_labels"]["en"]
    ids = continuation_ids(runner.tokenizer, prompt, answer)
    actual = runner.score(prompt, answer)
    with torch.inference_mode():
        result = runner.model(**runner.inputs(prompt.input_ids + ids))
        expected = sum(result.logits[0, len(prompt.input_ids) - 1 + j].float().log_softmax(0)[v].item() for j, v in enumerate(ids))
    assert len(ids) > 1
    assert actual["sum_logprob"] == pytest.approx(expected)


def test_same_string_alternatives_keep_outside_ids(runner, facts, templates):
    prompt = encode_prompt(runner.tokenizer, template_for(templates, "P19", "en", "t1"), facts[0]["subject_labels"]["en"], "en")
    alt = split_token_alternative(runner.tokenizer, prompt)
    assert alt is not None
    assert len(alt.input_ids) == len(prompt.input_ids) + 1
    assert runner.tokenizer.decode(alt.input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == prompt.text
    assert alt.input_ids[:alt.subject_indices[0]] == prompt.input_ids[:prompt.subject_indices[0]]
    assert alt.input_ids[alt.subject_indices[-1] + 1:] == prompt.input_ids[prompt.subject_indices[-1] + 1:]


def test_fixed_prediction_patch_stays_at_prompt_position(runner, facts, templates):
    a, b = [encode_prompt(runner.tokenizer, template_for(templates, "P19", "en", "t1"), f["subject_labels"]["en"], "en") for f in facts[:2]]
    key = (3, "residual", a.position("prediction"))
    vector = runner.capture(a, [key])[key]
    patch = Patch(3, "residual", b.position("prediction"), vector)
    first = runner.logits(b.input_ids, [patch])[:, -1]
    extended = runner.logits(b.input_ids + [10], [patch])
    assert torch.allclose(extended[:, -2], first, atol=1e-5)
    assert not torch.allclose(extended[:, -1], first, atol=1e-5)

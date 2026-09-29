"""Complete-answer evaluation and teacher-forced continuation boundaries."""
from .unicode import normalize_answer

ABSTENTIONS = {normalize_answer(x) for x in (
    "I don't know", "I do not know", "Unknown", "I am not sure", "לא יודע", "לא ידוע", "איני יודע",
    "אני לא יודע", "אני לא יודעת", "لا أعرف", "لا أعلم", "غير معروف")}


def evaluate_answer(text, object_qid, language, registry, truncated=False):
    normalized = normalize_answer(text)
    common = {"normalized": normalized, "entity_correct": False, "requested_language_correct": False}
    if truncated:
        return {**common, "category": "malformed_truncated"}
    if not normalized:
        return {**common, "category": "malformed_empty"}
    matches = registry.get(normalized, {})
    if len(matches) > 1:
        return {**common, "category": "ambiguous_alias", "candidate_qids": sorted(matches)}
    if object_qid in matches:
        return {**common, "category": "correct_entity", "entity_correct": True,
                "requested_language_correct": language in matches[object_qid],
                "alias_languages": matches[object_qid]}
    if normalized in ABSTENTIONS:
        return {**common, "category": "abstention"}
    return {**common, "category": "wrong_assertion", "manual_review_needed": not bool(matches)}


def continuation_ids(tokenizer, prompt, answer):
    if not answer or answer != answer.strip():
        raise ValueError("Canonical answer must be nonempty and trimmed")
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids or any(i in tokenizer.all_special_ids for i in ids):
        raise ValueError("Answer contains unknown or special token IDs")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != answer:
        raise ValueError("Answer token sequence does not decode exactly")
    canonical_prompt = tokenizer(prompt.text, add_special_tokens=False)["input_ids"]
    joint = tokenizer(prompt.text + answer, add_special_tokens=False)["input_ids"]
    if joint != canonical_prompt + ids:
        raise ValueError("Prompt/answer boundary changes tokenization; resolve before inference")
    # E3 uses deliberately noncanonical prompt IDs; append the same verified continuation IDs.
    return ids


def sequence_logprobs(logits, prompt_length, answer_ids):
    import torch
    if prompt_length < 1 or not answer_ids:
        raise ValueError("Need a nonempty prompt and answer")
    if logits.shape[1] < prompt_length + len(answer_ids) - 1:
        raise ValueError("Not enough logits for the full continuation")
    prediction = logits[:, prompt_length - 1:prompt_length + len(answer_ids) - 1, :].float()
    targets = torch.tensor(answer_ids, device=logits.device).view(1, -1, 1)
    token_lp = prediction.log_softmax(-1).gather(-1, targets).squeeze(-1)
    if not torch.isfinite(token_lp).all():
        raise ValueError("Nonfinite answer likelihood")
    values = token_lp[0].detach().cpu().tolist()
    return {"token_logprobs": values, "sum_logprob": sum(values), "mean_logprob": sum(values) / len(values),
            "answer_ids": list(answer_ids), "answer_tokens": len(values)}

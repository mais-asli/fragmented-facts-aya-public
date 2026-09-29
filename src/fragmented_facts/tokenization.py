"""Conservative same-string alternative segmentations; reject inexact round trips."""
from dataclasses import replace


def split_token_alternative(tokenizer, prompt, outside=False, max_extra=1):
    """Try character-boundary encodings of one token's decoded text.

    This does not enumerate the full tokenizer merge lattice. Eligibility is a
    measured coverage result, and byte-fragment tokens that do not independently
    decode are deliberately excluded. No pretrained embedding is added.
    """
    if tokenizer.decode(prompt.input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != prompt.text:
        return None
    candidates = range(len(prompt.input_ids)) if outside else prompt.subject_indices
    special = set(tokenizer.all_special_ids)
    for index in candidates:
        if outside and (index in prompt.subject_indices or index >= prompt.subject_indices[0]):
            continue  # Matched extra positions BEFORE the subject are a positional control.
        if not outside:
            a, b = prompt.offsets[index]
            if a < prompt.span[0] or b > prompt.span[1]:
                continue
        token = prompt.input_ids[index]
        if token in special:
            continue
        piece = tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if "\ufffd" in piece or len(piece) < 2:
            continue
        for cut in range(1, len(piece)):
            alt = tokenizer.encode(piece[:cut], add_special_tokens=False) + tokenizer.encode(piece[cut:], add_special_tokens=False)
            extra = len(alt) - 1
            if extra != max_extra or any(i in special for i in alt):
                continue
            ids = prompt.input_ids[:index] + alt + prompt.input_ids[index + 1:]
            if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != prompt.text:
                continue
            # Exact non-subject IDs and fixed full-string decoding are explicit invariants.
            a, b = prompt.offsets[index]
            offsets = prompt.offsets[:index] + [(a, b)] * len(alt) + prompt.offsets[index + 1:]
            subject = []
            for old in prompt.subject_indices:
                subject.extend(range(old, old + len(alt)) if old == index else [old + extra if old > index else old])
            result = replace(prompt, input_ids=ids, offsets=offsets, subject_indices=subject,
                             delimiter_index=prompt.delimiter_index + (extra if prompt.delimiter_index > index else 0))
            return result
    return None

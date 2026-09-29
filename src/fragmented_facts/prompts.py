"""Use official chat serialization and identify the subject in the full token sequence."""
from dataclasses import asdict, dataclass

from .unicode import base_length, nfc


@dataclass
class EncodedPrompt:
    text: str
    input_ids: list
    subject: str
    language: str
    span: tuple
    offsets: list
    subject_indices: list
    delimiter_index: int
    boundary_crossing: bool

    def position(self, site):
        return {"last_subject": self.subject_indices[-1], "first_subject": self.subject_indices[0],
                "delimiter": self.delimiter_index, "prediction": len(self.input_ids) - 1}[site]

    def record(self):
        row = asdict(self)
        base = base_length(self.subject, self.language)
        row.update(subject_tokens=len(self.subject_indices), base_characters=base,
            subject_words=len(self.subject.split()), tokens_per_base_character=len(self.subject_indices) / max(base, 1))
        return row


def encode_prompt(tokenizer, template, subject, language):
    if template.count("{subject}") != 1:
        raise ValueError("Template must have exactly one subject placeholder")
    if subject != nfc(subject):
        raise ValueError("Subject must be NFC normalized")
    content = template.replace("{subject}", subject)
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if text.count(subject) != 1:
        raise ValueError("Subject must occur once in the full chat prompt")
    start, end = text.index(subject), text.index(subject) + len(subject)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if ids != expected:
        raise ValueError("Chat-template token IDs disagree with rendered prompt token IDs")
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    subject_indices = [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]
    if not subject_indices:
        raise ValueError("Subject maps to no tokens")
    if subject_indices != list(range(subject_indices[0], subject_indices[-1] + 1)):
        raise ValueError("Subject maps to a noncontiguous token span")
    delimiter = next((i for i, (a, b) in enumerate(offsets) if i > subject_indices[-1] and b > a and a >= end), None)
    if delimiter is None:
        raise ValueError("No non-subject delimiter token after subject")
    crossing = any(offsets[i][0] < start or offsets[i][1] > end for i in subject_indices)
    return EncodedPrompt(text, ids, subject, language, (start, end), offsets, subject_indices, delimiter, crossing)


def paired_prompts(tokenizer, template, pair, language):
    u = encode_prompt(tokenizer, template, pair["U"], language)
    d = encode_prompt(tokenizer, template, pair["D"], language)
    if u.text[:u.span[0]] != d.text[:d.span[0]] or u.text[u.span[1]:] != d.text[d.span[1]:]:
        raise ValueError("Pair changes text outside the subject span")
    return u, d


def template_for(config, relation, language, template_id, stage="evaluation"):
    return config[stage][relation][language][template_id]["text"]


def validate_templates(config, require_review=False):
    for stage in ("screen", "evaluation"):
        for relation, langs in config[stage].items():
            for lang, templates in langs.items():
                for tid, item in templates.items():
                    text = item["text"]
                    if text.count("{subject}") != 1 or "{answer}" in text:
                        raise ValueError(f"Invalid template {stage}/{relation}/{lang}/{tid}")
                    if require_review and (item.get("status") != "approved" or not item.get("reviewer")):
                        raise ValueError(f"Template needs native review: {stage}/{relation}/{lang}/{tid}")
    for relation, langs in config["screen"].items():
        if len(langs["en"]) != 2:
            raise ValueError("Exactly two dedicated English screening templates are required")
        if {x["text"] for x in langs["en"].values()} & {x["text"] for x in config["evaluation"][relation]["en"].values()}:
            raise ValueError("Screen and reported English templates overlap")

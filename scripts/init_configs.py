"""Create draft templates once. Never overwrite subsequent native review."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
questions = {
    "P19": {
        "en": ["Where was this person born?", "What is this person's birthplace?", "Name the place where this person was born."],
        "he": ["היכן נולד האדם הזה?", "מהו מקום הלידה של האדם הזה?", "מה שם המקום שבו נולד האדם הזה?"],
        "ar": ["أين وُلد هذا الشخص؟", "ما مكان ولادة هذا الشخص؟", "ما اسم المكان الذي وُلد فيه هذا الشخص؟"]},
    "P20": {
        "en": ["Where did this person die?", "What is this person's place of death?", "Name the place where this person died."],
        "he": ["היכן נפטר האדם הזה?", "מהו מקום הפטירה של האדם הזה?", "מה שם המקום שבו נפטר האדם הזה?"],
        "ar": ["أين توفي هذا الشخص؟", "ما مكان وفاة هذا الشخص؟", "ما اسم المكان الذي توفي فيه هذا الشخص؟"]},
    "P159": {
        "en": ["Where is this organization's headquarters?", "What is the location of this organization's headquarters?", "Name the place where this organization is headquartered."],
        "he": ["היכן נמצא המטה של הארגון הזה?", "מהו מיקום המטה של הארגון הזה?", "מה שם המקום שבו נמצא מטה הארגון הזה?"],
        "ar": ["أين يقع المقر الرئيسي لهذه المنظمة؟", "ما موقع المقر الرئيسي لهذه المنظمة؟", "ما اسم المكان الذي يقع فيه المقر الرئيسي لهذه المنظمة؟"]},
    "P740": {
        "en": ["Where was this organization founded?", "What is this organization's place of formation?", "Name the place where this organization was formed."],
        "he": ["היכן נוסד הארגון הזה?", "מהו מקום ההקמה של הארגון הזה?", "מה שם המקום שבו הוקם הארגון הזה?"],
        "ar": ["أين تأسست هذه المنظمة؟", "ما مكان تأسيس هذه المنظمة؟", "ما اسم المكان الذي تأسست فيه هذه المنظمة؟"]},
}
labels = {"en": ("Subject", "Question", "Give only the place name."),
          "he": ("נושא", "שאלה", "יש להשיב רק בשם המקום."),
          "ar": ("الموضوع", "السؤال", "أجب باسم المكان فقط.")}
screen = {"P19": ["State the birthplace of {subject}. Answer with only the place name.", "Where was {subject} born? Return only the location."],
          "P20": ["State the place of death of {subject}. Answer with only the place name.", "Where did {subject} die? Return only the location."],
          "P159": ["State the headquarters location of {subject}. Answer with only the place name.", "Where are the headquarters of {subject}? Return only the location."],
          "P740": ["State the founding location of {subject}. Answer with only the place name.", "Where was {subject} founded? Return only the location."]}


def item(text):
    return {"text": text, "status": "pending_native_review", "reviewer": "", "notes": "Draft generated with AI assistance; verify relation semantics and naturalness."}


def save(name, value):
    path = ROOT / "configs" / name
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Will not overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    evaluation = {}
    for relation, languages in questions.items():
        evaluation[relation] = {}
        for language, variants in languages.items():
            subject, question, answer = labels[language]
            evaluation[relation][language] = {f"t{i + 1}": item(f"{subject}: {{subject}}\n{question}: {q}\n{answer}") for i, q in enumerate(variants)}
    save("templates.json", {"schema_version": 1, "evaluation": evaluation,
        "screen": {r: {"en": {f"s{i + 1}": item(t) for i, t in enumerate(v)}} for r, v in screen.items()}})
    save("study.json", {"schema_version": 1, "model_id": "CohereLabs/aya-23-8B", "model_kind": "aya23",
        "precision": "nf4", "nf4_compute_dtype": "fp16", "attention": "eager", "max_new_tokens": 24, "seed": 17,
        "languages": ["en", "he", "ar"], "relations": list(questions), "mechanism_template": "t1",
        "partial_diacritics": False, "segmentation_extra_tokens": [1, 2], "patch_generate": True,
        "patch_components": ["residual"], "native_smoke_review": {"reviewer": "", "notes": ""},
        "workload_pilot_report": "", "readout_layers": None, "readout_layers_frozen": False,
        "windows": [], "primary_window": None, "window_selection_reason": "",
        "population": "core", "sample_limits": {"E2": {"dev": 60, "test": 180}, "E5": {"dev": 60, "test": 120}},
        "relation_change": "P50 is absent from the original mLAMA archive; P740 (location of formation) replaces it before screening. Native/source validation is required."})

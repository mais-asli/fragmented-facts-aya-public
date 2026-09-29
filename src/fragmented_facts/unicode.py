"""Conservative, explicitly enumerated normalization policy."""
import unicodedata

# Arabic short vowels/tanwin, shadda, sukun, superscript alef. Hamza is NOT removed.
AR_MARKS = frozenset(chr(i) for i in range(0x064B, 0x0653)) | {"\u0670"}
# Hebrew niqqud, dagesh, shin/sin dots, meteg, rafe. Cantillation is not silently removed.
HE_MARKS = frozenset(chr(i) for i in range(0x05B0, 0x05BE)) | {"\u05BF", "\u05C1", "\u05C2", "\u05C7"}
MARKS = {"en": frozenset(), "ar": AR_MARKS, "he": HE_MARKS}


def nfc(text):
    return unicodedata.normalize("NFC", text)


def strip_marks(text, language):
    return "".join(c for c in nfc(text) if c not in MARKS[language])


def validate_pair(unmarked, marked, language):
    u, d = nfc(unmarked), nfc(marked)
    if not u or not d or u == d:
        raise ValueError("A pair needs two nonempty, different strings")
    if strip_marks(u, language) != u:
        raise ValueError("U must be unmarked under the declared allowlist")
    if strip_marks(d, language) != u:
        raise ValueError("Marked and unmarked spellings differ in base characters")
    if any(unicodedata.category(c).startswith("C") for c in u + d):
        raise ValueError("Hidden/control characters are not allowed in a pair")
    return u, d


def normalize_answer(text):
    text = nfc(text).strip()
    # Strip only outer presentation punctuation; preserve internal words/punctuation.
    text = text.strip(' \t\n\r"“”«».,!؟?؛;:')
    text = " ".join(text.split()).casefold()
    return "".join(c for c in text if c not in AR_MARKS and c not in HE_MARKS)


def base_length(text, language):
    return sum(not c.isspace() for c in strip_marks(text, language))

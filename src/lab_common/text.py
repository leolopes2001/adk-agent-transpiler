import re
import unicodedata
from dataclasses import dataclass

# Hangul fillers render as blank but are letters (Lo), so the category check misses them.
_INVISIBLE_LETTERS = frozenset("\u115f\u1160\u3164\uffa0")
_KEPT_CONTROLS = frozenset("\n\t")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_NON_WORD = re.compile(r"[\W_]+")

STOPWORDS = frozenset(
    {"de", "da", "do", "das", "dos", "e", "em", "para", "com", "o", "a", "exame", "dosagem"}
)


@dataclass(frozen=True, slots=True)
class SanitizedText:
    text: str
    truncated: bool
    removed_chars: int
    """Non-zero is an injection signal: legitimate OCR output rarely has these characters."""


def sanitize(text: str, max_chars: int) -> SanitizedText:
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    text = _unify_line_breaks(text)
    text, removed = _strip_removable(text)
    # Stripping first lets NFKC compose across removed characters; NFKC can expose new ones.
    text = unicodedata.normalize("NFKC", text)
    text, removed_after_nfkc = _strip_removable(text)
    text = _collapse_whitespace(text)

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip()

    return SanitizedText(
        text=text,
        truncated=truncated,
        removed_chars=removed + removed_after_nfkc,
    )


def tokens(text: str) -> list[str]:
    return [t for t in _search_words(text) if t not in STOPWORDS]


def normalize_for_search(text: str) -> str:
    words = _search_words(text)
    kept = [t for t in words if t not in STOPWORDS]
    # A final one-letter stopword names the exam ("Vitamina E"); elsewhere it is a conjunction.
    last = words[-1] if words else ""
    if last in STOPWORDS and len(last) == 1 and kept:
        kept.append(last)

    return " ".join(kept)


def _unify_line_breaks(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\u2028", "\n").replace("\u2029", "\n")


def _collapse_whitespace(text: str) -> str:
    lines = (_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _is_removable(char: str) -> bool:
    if char in _KEPT_CONTROLS:
        return False

    return unicodedata.category(char).startswith("C") or char in _INVISIBLE_LETTERS


def _strip_removable(text: str) -> tuple[str, int]:
    kept = [c for c in text if not _is_removable(c)]
    return "".join(kept), len(text) - len(kept)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.category(c).startswith("M"))


def _search_words(text: str) -> list[str]:
    # Case folding can add combining marks (U+0130 -> "i" + U+0307), so accents go twice.
    folded = _strip_accents(_strip_accents(text).casefold())
    return _NON_WORD.sub(" ", folded).split()

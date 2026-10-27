"""Text sanitization for untrusted input and normalization for search.

`sanitize` makes OCR text safe to inspect and hand to a model: it strips
characters that are invisible or change rendering order (a common way to
smuggle instructions past a human reviewer) while keeping the visible content,
accents and line structure intact.

`normalize_for_search` produces the canonical form used for matching exam
names; the RAG index and the runtime grounding check must agree on it, so both
call this function.
"""

import re
import unicodedata
from dataclasses import dataclass

# Fillers that render as blank but are letters (category Lo), so the
# category-based filter below does not catch them.
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
    """Invisible or control characters dropped; a non-zero value is a signal for the
    injection scanner, since legitimate OCR output rarely contains them."""


def _is_removable(char: str) -> bool:
    if char in _KEPT_CONTROLS:
        return False
    # Category C covers controls (Cc), format characters such as zero-width and
    # bidi overrides (Cf), surrogates, private use and unassigned code points.
    return unicodedata.category(char).startswith("C") or char in _INVISIBLE_LETTERS


def _strip_removable(text: str) -> tuple[str, int]:
    kept = [c for c in text if not _is_removable(c)]
    return "".join(kept), len(text) - len(kept)


def sanitize(text: str, max_chars: int) -> SanitizedText:
    """Normalize untrusted text and cap its length.

    Applies NFKC, drops control/format/invisible characters (keeping newlines;
    tabs are kept as whitespace), collapses horizontal whitespace, trims lines,
    squeezes runs of blank lines and truncates to `max_chars`.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text, removed = _strip_removable(text)
    # Removing a format character can leave a base letter next to a combining
    # mark, so composition runs after the first strip; NFKC itself can expose
    # new removable characters, hence the second strip.
    text = unicodedata.normalize("NFKC", text)
    text, removed_after = _strip_removable(text)

    lines = (_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip()
    return SanitizedText(text=text, truncated=truncated, removed_chars=removed + removed_after)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.category(c).startswith("M"))


def _words(text: str) -> list[str]:
    # Case folding can itself produce combining marks (U+0130 -> "i" + U+0307),
    # so accents are stripped both before and after it.
    folded = _strip_accents(_strip_accents(text).casefold())
    return _NON_WORD.sub(" ", folded).split()


def tokens(text: str) -> list[str]:
    """Search tokens of `text`, in order and with repetitions, stopwords removed."""
    return [t for t in _words(text) if t not in STOPWORDS]


def normalize_for_search(text: str) -> str:
    """Accent-free, lowercase, punctuation-free form of `text` without stopwords.

    A one-letter stopword that ends the text after other words is kept: there
    it names the exam ("Vitamina E", "Vitamina A"), while in the middle it is
    a conjunction ("Colesterol total e frações").
    """
    words = _words(text)
    kept = [t for t in words if t not in STOPWORDS]
    last = words[-1] if words else ""
    if last in STOPWORDS and len(last) == 1 and kept:
        kept.append(last)
    return " ".join(kept)

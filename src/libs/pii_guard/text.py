import re
import unicodedata

_NON_WORD = re.compile(r"[\W_]+")

STOPWORDS = frozenset(
    {"de", "da", "do", "das", "dos", "e", "em", "para", "com", "o", "a", "exame", "dosagem"}
)


def tokens(text: str) -> list[str]:
    return [t for t in _search_words(text) if t not in STOPWORDS]


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.category(c).startswith("M"))


def _search_words(text: str) -> list[str]:
    # Case folding can add combining marks (U+0130 -> "i" + U+0307), so accents go twice.
    folded = _strip_accents(_strip_accents(text).casefold())
    return _NON_WORD.sub(" ", folded).split()

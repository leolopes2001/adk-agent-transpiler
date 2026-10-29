import re
import unicodedata

_NON_WORD = re.compile(r"[\W_]+")

# One edit per six letters: "JELEFONE" and "PATA DE NAScIMENTO" match TELEFONE and
# DATA DE NASCIMENTO, "FERRLTINA" matches ferritina; "CRESCIMENTO" is not NASCIMENTO.
LETTERS_PER_EDIT = 6

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


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current

    return previous[-1]

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from lab_common.text import normalize_for_search
from rag_mcp.catalog import Catalog

CANDIDATES_PER_RETRIEVER = 20

FieldKind = Literal["name", "alias"]


@dataclass(frozen=True, slots=True)
class FieldDoc:
    code: str
    kind: FieldKind
    text: str
    tokens: tuple[str, ...]


class HybridIndex:
    """Retrievers only pick candidates: raw BM25 values are not comparable across queries."""

    def __init__(self, catalog: Catalog) -> None:
        docs = _field_docs(catalog)
        if not docs:
            raise ValueError("cannot index an empty catalog")

        self._docs = tuple(docs)
        self._exact = {doc.text: doc for doc in docs}
        self._by_code = _docs_by_code(docs)
        self._bm25 = BM25Okapi([list(doc.tokens) for doc in docs])
        self._trigrams = tuple(trigrams(doc.text) for doc in docs)
        self._postings = _trigram_postings(self._trigrams)

    @property
    def docs(self) -> Sequence[FieldDoc]:
        return self._docs

    def exact(self, text: str) -> FieldDoc | None:
        return self._exact.get(text)

    def docs_of(self, code: str) -> Sequence[FieldDoc]:
        return self._by_code.get(code, ())

    def bm25_top(self, tokens: Sequence[str], limit: int = CANDIDATES_PER_RETRIEVER) -> list[int]:
        if not tokens:
            return []

        scores = self._bm25.get_scores(list(tokens))
        return _best_doc_ids(((doc_id, float(score)) for doc_id, score in enumerate(scores)), limit)

    def trigram_top(self, text: str, limit: int = CANDIDATES_PER_RETRIEVER) -> list[int]:
        """Catches OCR typos and truncated words ("Hemogrma"), which share no token."""
        query = trigrams(text)
        if not query:
            return []

        shared: dict[int, int] = defaultdict(int)
        for gram in query:
            for doc_id in self._postings.get(gram, ()):
                shared[doc_id] += 1

        jaccard = (
            (doc_id, n / (len(query) + len(self._trigrams[doc_id]) - n))
            for doc_id, n in shared.items()
        )
        return _best_doc_ids(jaccard, limit)

    def candidate_codes(self, text: str) -> list[str]:
        codes: dict[str, None] = {}
        if (exact := self.exact(text)) is not None:
            codes[exact.code] = None

        for doc_id in [*self.bm25_top(text.split()), *self.trigram_top(text)]:
            codes.setdefault(self._docs[doc_id].code, None)

        return list(codes)


def trigrams(text: str) -> frozenset[str]:
    if not text:
        return frozenset()

    # Padding makes word starts and ends count as trigrams too.
    padded = f"  {text} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def _field_docs(catalog: Catalog) -> list[FieldDoc]:
    docs: list[FieldDoc] = []
    for exam in catalog:
        terms: list[tuple[FieldKind, str]] = [("name", exam.name)]
        terms += [("alias", alias) for alias in exam.aliases]
        for kind, term in terms:
            text = normalize_for_search(term)
            docs.append(FieldDoc(exam.code, kind, text, tuple(text.split())))

    return docs


def _docs_by_code(docs: Iterable[FieldDoc]) -> dict[str, list[FieldDoc]]:
    by_code: dict[str, list[FieldDoc]] = defaultdict(list)
    for doc in docs:
        by_code[doc.code].append(doc)

    return dict(by_code)


def _trigram_postings(trigram_sets: Iterable[frozenset[str]]) -> dict[str, list[int]]:
    postings: dict[str, list[int]] = defaultdict(list)
    for doc_id, grams in enumerate(trigram_sets):
        for gram in grams:
            postings[gram].append(doc_id)

    return dict(postings)


def _best_doc_ids(scores: Iterable[tuple[int, float]], limit: int) -> list[int]:
    ranked = sorted((-score, doc_id) for doc_id, score in scores if score > 0)
    return [doc_id for _, doc_id in ranked[:limit]]

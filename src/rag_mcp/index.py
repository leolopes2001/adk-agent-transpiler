"""Search index over the catalog, built once at startup and never changed.

Each exam contributes one field doc for its name and one per alias, all in the
`normalize_for_search` form. Two retrievers propose candidates:

- BM25 (`rank_bm25.BM25Okapi`) over field doc tokens, for queries that share
  whole words with an exam;
- character trigrams (padded string, Jaccard similarity), for queries with
  OCR errors or truncated words ("Hemogrma", "Glicemla"), which share no token.

Retrievers only choose which exams get scored; the score itself comes from
`scoring`, so raw BM25 values, which are not comparable across queries, never
reach the client.
"""

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
    """Normalized name or alias."""
    tokens: tuple[str, ...]


def trigrams(text: str) -> frozenset[str]:
    """Character trigrams of `text`, padded so word starts and ends count too."""
    if not text:
        return frozenset()
    padded = f"  {text} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def _top(scores: Iterable[tuple[int, float]], limit: int) -> list[int]:
    """Doc ids with a positive score, best first; ties by id, so it is deterministic."""
    ranked = sorted(((-score, doc_id) for doc_id, score in scores if score > 0))
    return [doc_id for _, doc_id in ranked[:limit]]


class HybridIndex:
    def __init__(self, catalog: Catalog) -> None:
        docs: list[FieldDoc] = []
        for exam in catalog:
            kinds: list[tuple[FieldKind, str]] = [("name", exam.name)]
            kinds += [("alias", alias) for alias in exam.aliases]
            for kind, term in kinds:
                text = normalize_for_search(term)
                docs.append(FieldDoc(exam.code, kind, text, tuple(text.split())))
        if not docs:
            raise ValueError("cannot index an empty catalog")

        self._docs = tuple(docs)
        # The catalog guarantees normalized terms are unique, so this is a 1:1 map.
        self._exact = {doc.text: doc for doc in docs}
        self._by_code: dict[str, list[FieldDoc]] = defaultdict(list)
        for doc in docs:
            self._by_code[doc.code].append(doc)
        self._bm25 = BM25Okapi([list(doc.tokens) for doc in docs])
        self._trigrams = tuple(trigrams(doc.text) for doc in docs)
        postings: dict[str, list[int]] = defaultdict(list)
        for doc_id, grams in enumerate(self._trigrams):
            for gram in grams:
                postings[gram].append(doc_id)
        self._postings = dict(postings)

    @property
    def docs(self) -> Sequence[FieldDoc]:
        return self._docs

    def exact(self, text: str) -> FieldDoc | None:
        """Field doc whose normalized text equals `text`, if any."""
        return self._exact.get(text)

    def docs_of(self, code: str) -> Sequence[FieldDoc]:
        return self._by_code.get(code, ())

    def bm25_top(self, tokens: Sequence[str], limit: int = CANDIDATES_PER_RETRIEVER) -> list[int]:
        if not tokens:
            return []
        scores = self._bm25.get_scores(list(tokens))
        return _top(((i, float(s)) for i, s in enumerate(scores)), limit)

    def trigram_top(self, text: str, limit: int = CANDIDATES_PER_RETRIEVER) -> list[int]:
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
        return _top(jaccard, limit)

    def candidate_codes(self, text: str) -> list[str]:
        """Exams proposed by either retriever for the normalized `text`.

        The exam of an exact match is always included. Order is the order of
        first appearance (exact, BM25, trigrams); callers rank by score.
        """
        codes: dict[str, None] = {}
        if (exact := self.exact(text)) is not None:
            codes[exact.code] = None
        for doc_id in [*self.bm25_top(text.split()), *self.trigram_top(text)]:
            codes.setdefault(self._docs[doc_id].code, None)
        return list(codes)

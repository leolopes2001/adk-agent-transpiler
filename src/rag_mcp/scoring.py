"""Scores in [0, 1] that mean the same thing for every query.

For each candidate exam (proposed by the index), over its name and aliases:

- exact: the normalized query equals a name or alias -> 1.0;
- fuzzy: best `rapidfuzz.fuzz.WRatio` / 100, which tolerates typos, extra
  words and reordering;
- coverage: best share of the query's tokens that appear in the field, which
  keeps a long field from winning on one shared fragment;
- score = round(0.75 * fuzzy + 0.25 * coverage, 4).

Both parts depend only on the query and the field text (not on corpus
statistics, as BM25 does), so a threshold means the same for any query.
Candidates are ordered by (-score, code): equal scores always come out in the
same order.

Before normalizing, one leading list marker is removed ("1. ", "2) ", "- ",
"a) "), since OCR returns order items with their numbering; a number that is
part of the exam name ("25-hidroxivitamina D") is kept.

Scoring never decides acceptance: that is the runtime's confidence gate.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from lab_common.contracts.rag import Candidate, MatchedOn, SearchResult
from lab_common.text import normalize_for_search
from rag_mcp.catalog import Catalog
from rag_mcp.index import FieldDoc, HybridIndex

FUZZY_WEIGHT = 0.75
COVERAGE_WEIGHT = 0.25
EXACT_SCORE = 1.0

_LIST_MARKER = re.compile(
    r"""^\s*(?:
        \(?[0-9]{1,3}\s*[.)º°]               # "1." "1)" "(1)" "1º"
      | [0-9]{1,3}\s*[-\u2013\u2014:](?=\s) # "1 - " "1: " (not "25-OH")
      | \(?[a-zA-Z]\)                        # "a)" "(b)"
      | [•·▪*\-\u2013\u2014](?=\s|[^\W\d_]) # bullets and dashes
    )\s*""",
    re.VERBOSE,
)


def strip_list_marker(query: str) -> str:
    """`query` without one leading enumeration or bullet marker."""
    return _LIST_MARKER.sub("", query, count=1)


def prepare_query(query: str) -> str:
    """Normalized form of a raw mention, as it is matched against the index."""
    return normalize_for_search(strip_list_marker(query))


def fuzzy(query: str, field: str) -> float:
    """WRatio of two normalized strings, in [0, 1]."""
    return fuzz.WRatio(query, field, processor=None) / 100


def coverage(query_tokens: Sequence[str], field_tokens: Sequence[str]) -> float:
    """Share of the distinct query tokens found in the field, in [0, 1]."""
    wanted = set(query_tokens)
    if not wanted:
        return 0.0
    return len(wanted & set(field_tokens)) / len(wanted)


def combine(fuzzy_score: float, coverage_score: float) -> float:
    return round(FUZZY_WEIGHT * fuzzy_score + COVERAGE_WEIGHT * coverage_score, 4)


@dataclass(frozen=True, slots=True)
class ScoredExam:
    code: str
    score: float
    matched_on: MatchedOn


def score_exam(query: str, docs: Sequence[FieldDoc]) -> ScoredExam:
    """Score of one exam (given by its field docs) for the normalized `query`."""
    if not docs:
        raise ValueError("an exam needs at least one field doc")
    code = docs[0].code
    for doc in docs:
        if doc.text == query:
            return ScoredExam(
                code, EXACT_SCORE, "exact_name" if doc.kind == "name" else "exact_alias"
            )

    query_tokens = query.split()
    best_fuzzy, best_doc = max(
        # Ties keep the first doc, so the name wins over an alias.
        ((fuzzy(query, doc.text), doc) for doc in docs),
        key=lambda pair: pair[0],
    )
    best_coverage = max(coverage(query_tokens, doc.tokens) for doc in docs)
    return ScoredExam(code, combine(best_fuzzy, best_coverage), best_doc.kind)


def rank(scored: Sequence[ScoredExam], top_k: int) -> list[ScoredExam]:
    return sorted(scored, key=lambda s: (-s.score, s.code))[:top_k]


class ExamSearcher:
    """Retrieval and scoring over one loaded catalog."""

    def __init__(self, catalog: Catalog, index: HybridIndex | None = None) -> None:
        self._catalog = catalog
        self._index = index or HybridIndex(catalog)

    @property
    def catalog(self) -> Catalog:
        return self._catalog

    def search(self, query: str, top_k: int) -> SearchResult:
        normalized = prepare_query(query)
        scored = [
            score_exam(normalized, self._index.docs_of(code))
            for code in self._index.candidate_codes(normalized)
        ]
        candidates = []
        for item in rank(scored, top_k):
            exam = self._catalog.get(item.code)
            # The index is built from this catalog; a miss would be a bug, and
            # a code outside the catalog must never be returned.
            if exam is None:
                raise RuntimeError("index returned a code that is not in the catalog")
            candidates.append(
                Candidate(
                    code=exam.code,
                    name=exam.name,
                    score=item.score,
                    matched_on=item.matched_on,
                    category=exam.category,
                )
            )
        return SearchResult(
            catalog_version=self._catalog.version,
            catalog_sha256=self._catalog.sha256,
            query=query,
            normalized_query=normalized,
            candidates=candidates,
        )

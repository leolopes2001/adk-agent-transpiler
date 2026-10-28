import re
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from lab_common.contracts.rag import Candidate, MatchedOn, SearchResult
from lab_common.text import normalize_for_search
from rag_mcp.catalog import Catalog
from rag_mcp.index import FieldDoc, HybridIndex

# No corpus statistics in the score, so a threshold means the same for every query.
FUZZY_WEIGHT = 0.75
COVERAGE_WEIGHT = 0.25
EXACT_SCORE = 1.0
# Scales of rapidfuzz's WRatio, mirrored by _partial_without_shared_tokens.
WRATIO_PARTIAL_FROM = 1.5
WRATIO_LONG_FROM = 8
WRATIO_PARTIAL_SCALE = 0.9
WRATIO_LONG_SCALE = 0.6
WRATIO_UNBASE = 0.95
WRATIO_MAX = 100.0

# OCR keeps the order's numbering; only one marker goes, so "25-hidroxivitamina D" keeps its 25.
_LIST_MARKER = re.compile(
    r"""^\s*(?:
        \(?[0-9]{1,3}\s*[.)º°]               # "1." "1)" "(1)" "1º"
      | [0-9]{1,3}\s*[-\u2013\u2014:](?=\s) # "1 - " "1: " (not "25-OH")
      | \(?[a-zA-Z]\)                        # "a)" "(b)"
      | [•·▪*\-\u2013\u2014](?=\s|[^\W\d_]) # bullets and dashes
    )\s*""",
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class ScoredExam:
    code: str
    score: float
    matched_on: MatchedOn


class ExamSearcher:
    """Scores only rank candidates; accepting one is the runtime's confidence gate."""

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
        return SearchResult(
            catalog_version=self._catalog.version,
            catalog_sha256=self._catalog.sha256,
            query=query,
            normalized_query=normalized,
            candidates=[self._candidate(item) for item in rank(scored, top_k)],
        )

    def _candidate(self, scored: ScoredExam) -> Candidate:
        exam = self._catalog.get(scored.code)
        if exam is None:
            raise RuntimeError("index returned a code that is not in the catalog")

        return Candidate(
            code=exam.code,
            name=exam.name,
            score=scored.score,
            matched_on=scored.matched_on,
            category=exam.category,
        )


def strip_list_marker(query: str) -> str:
    return _LIST_MARKER.sub("", query, count=1)


def prepare_query(query: str) -> str:
    return normalize_for_search(strip_list_marker(query))


def fuzzy(query: str, field: str) -> float:
    score = fuzz.WRatio(query, field, processor=None)
    shared = set(query.split()) & set(field.split())
    # One shared letter ("b" in "hepatite b") would score 100 through partial_token_set_ratio.
    if shared and max(map(len, shared)) == 1:
        score = min(score, _partial_without_shared_tokens(query, field))

    return score / WRATIO_MAX


def coverage(query_tokens: Sequence[str], field_tokens: Sequence[str]) -> float:
    """Keeps a long field from winning on one shared fragment."""
    wanted = set(query_tokens)
    if not wanted:
        return 0.0

    return len(wanted & set(field_tokens)) / len(wanted)


def combine(fuzzy_score: float, coverage_score: float) -> float:
    return round(FUZZY_WEIGHT * fuzzy_score + COVERAGE_WEIGHT * coverage_score, 4)


def score_exam(query: str, docs: Sequence[FieldDoc]) -> ScoredExam:
    if not docs:
        raise ValueError("an exam needs at least one field doc")

    code = docs[0].code
    exact = _exact_match(code, query, docs)
    if exact is not None:
        return exact

    fuzzy_scores = [fuzzy(query, doc.text) for doc in docs]
    best_fuzzy = max(fuzzy_scores)
    # index() takes the first tie, so the name wins over an alias.
    best_doc = docs[fuzzy_scores.index(best_fuzzy)]
    query_tokens = query.split()
    best_coverage = max(coverage(query_tokens, doc.tokens) for doc in docs)
    return ScoredExam(code, combine(best_fuzzy, best_coverage), best_doc.kind)


def rank(scored: Sequence[ScoredExam], top_k: int) -> list[ScoredExam]:
    """Equal scores come out by code, so the order never varies."""
    return sorted(scored, key=lambda item: (-item.score, item.code))[:top_k]


def _exact_match(code: str, query: str, docs: Sequence[FieldDoc]) -> ScoredExam | None:
    for doc in docs:
        if doc.text == query:
            matched_on: MatchedOn = "exact_name" if doc.kind == "name" else "exact_alias"
            return ScoredExam(code, EXACT_SCORE, matched_on)

    return None


def _partial_without_shared_tokens(query: str, field: str) -> float:
    shorter, longer = sorted((len(query), len(field)))
    length_ratio = longer / shorter
    # Below this ratio WRatio never uses the partial parts, so its score stays as is.
    if length_ratio < WRATIO_PARTIAL_FROM:
        return WRATIO_MAX

    scale = WRATIO_PARTIAL_SCALE if length_ratio < WRATIO_LONG_FROM else WRATIO_LONG_SCALE
    return max(
        fuzz.ratio(query, field, processor=None),
        fuzz.partial_ratio(query, field, processor=None) * scale,
        fuzz.partial_token_sort_ratio(query, field, processor=None) * WRATIO_UNBASE * scale,
    )

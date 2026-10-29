"""Retrieval quality on the versioned evaluation set, with the spec's gate thresholds.

`tests/data/rag_eval.jsonl` has one mention per line: `expected` is the right
catalog code (null for exams the catalog does not have and OCR debris), `kind`
says what the mention exercises, and ambiguous mentions list every reading in
`acceptable`. Decisions use the runtime's gate rule (`lab_common.gate`).
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lab_common.contracts.rag import SearchResult
from lab_common.gate import gate_decision
from lab_common.text import normalize_for_search
from rag_mcp.catalog import Catalog, load_catalog
from rag_mcp.scoring import ExamSearcher

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_PATH = REPO_ROOT / "tests" / "data" / "rag_eval.jsonl"
CATALOG_PATH = REPO_ROOT / "data" / "exams_catalog.json"
SPEC_PATH = REPO_ROOT / "agent_specs" / "exam_scheduling_agent.json"
KINDS = {"exact", "alias", "typo", "ocr_noise", "abbrev", "ambiguous", "negative"}
MIN_PRECISION_AT_1 = 0.95
# The sample order as the OCR server returns it (see samples/expected).
OCR_LINES = [
    "1. Hemograma completo",
    "2. Glicemia de jejum",
    "3. TSH",
    "4. T4 livre",
    "5. Colesterol total e frações",
    "6. Urina tipo 1 (EAS)",
    "7. PCR",
]


@dataclass(frozen=True)
class Thresholds:
    accept: float
    review: float
    margin: float


def gate(result: SearchResult, thresholds: Thresholds) -> str:
    """Decision of the runtime's confidence gate for one mention."""
    status, _ = gate_decision(
        (c.score for c in result.candidates),
        accept=thresholds.accept,
        review=thresholds.review,
        margin=thresholds.margin,
    )
    return status


def spec_thresholds() -> Thresholds:
    spec = json.loads(SPEC_PATH.read_text("utf-8"))
    [params] = [n["params"] for n in spec["workflow"]["nodes"] if n["kind"] == "confidence_gate"]
    return Thresholds(params["accept_threshold"], params["review_threshold"], params["min_margin"])


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    with EVAL_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog(CATALOG_PATH)


@pytest.fixture(scope="module")
def results(rows: list[dict[str, Any]], catalog: Catalog) -> list[SearchResult]:
    searcher = ExamSearcher(catalog)
    return [searcher.search(row["query"], 5) for row in rows]


def correct(row: dict[str, Any], result: SearchResult) -> bool:
    if not result.candidates:
        return False
    top = result.candidates[0].code
    return top == row["expected"] or top in row.get("acceptable", [])


class TestEvalSet:
    def test_rows_are_well_formed(self, rows: list[dict[str, Any]], catalog: Catalog) -> None:
        for row in rows:
            assert set(row) <= {"query", "expected", "kind", "acceptable"}, row
            assert row["kind"] in KINDS, row
            assert 1 <= len(row["query"]) <= 200, row
            if row["kind"] == "negative":
                assert row["expected"] is None, row
                assert "acceptable" not in row, row
            elif row["kind"] == "ambiguous":
                assert row["expected"] is None, row
                assert len(row["acceptable"]) >= 2, row
                assert all(code in catalog for code in row["acceptable"]), row
            else:
                assert row["expected"] in catalog, row
                assert "acceptable" not in row, row

    def test_has_every_kind_with_enough_negatives(self, rows: list[dict[str, Any]]) -> None:
        kinds = Counter(row["kind"] for row in rows)
        assert set(kinds) == KINDS
        assert len(rows) >= 80
        assert kinds["negative"] >= 15
        assert kinds["ambiguous"] >= 5

    def test_queries_are_unique(self, rows: list[dict[str, Any]]) -> None:
        queries = [row["query"] for row in rows]
        assert len(queries) == len(set(queries))

    def test_includes_the_sample_order_as_ocr_returns_it(self, rows: list[dict[str, Any]]) -> None:
        queries = {row["query"] for row in rows}
        assert set(OCR_LINES) <= queries

    def test_includes_the_documented_ambiguities(self, rows: list[dict[str, Any]]) -> None:
        ambiguous = {row["query"] for row in rows if row["kind"] == "ambiguous"}
        assert {"T4", "colesterol"} <= ambiguous

    def test_negatives_are_not_catalog_terms(
        self, rows: list[dict[str, Any]], catalog: Catalog
    ) -> None:
        terms = {
            normalize_for_search(term) for exam in catalog for term in [exam.name, *exam.aliases]
        }
        negatives = [row["query"] for row in rows if row["kind"] == "negative"]
        assert [q for q in negatives if normalize_for_search(q) in terms] == []


class TestRegression:
    def test_precision_at_1_on_non_negatives(
        self, rows: list[dict[str, Any]], results: list[SearchResult]
    ) -> None:
        scored = [
            correct(row, result)
            for row, result in zip(rows, results, strict=True)
            if row["kind"] != "negative"
        ]
        precision = sum(scored) / len(scored)
        assert precision >= MIN_PRECISION_AT_1

    def test_no_negative_is_accepted(
        self, rows: list[dict[str, Any]], results: list[SearchResult]
    ) -> None:
        thresholds = spec_thresholds()
        accepted = [
            row["query"]
            for row, result in zip(rows, results, strict=True)
            if row["kind"] == "negative" and gate(result, thresholds) == "accepted"
        ]
        assert accepted == []

    def test_no_ambiguous_mention_is_accepted(
        self, rows: list[dict[str, Any]], results: list[SearchResult]
    ) -> None:
        thresholds = spec_thresholds()
        accepted = [
            row["query"]
            for row, result in zip(rows, results, strict=True)
            if row["kind"] == "ambiguous" and gate(result, thresholds) == "accepted"
        ]
        assert accepted == []

    def test_every_accepted_mention_is_right(
        self, rows: list[dict[str, Any]], results: list[SearchResult]
    ) -> None:
        thresholds = spec_thresholds()
        wrong = [
            row["query"]
            for row, result in zip(rows, results, strict=True)
            if gate(result, thresholds) == "accepted"
            and result.candidates[0].code != row["expected"]
        ]
        assert wrong == []

    def test_sample_order_is_accepted_as_expected(
        self, rows: list[dict[str, Any]], results: list[SearchResult]
    ) -> None:
        thresholds = spec_thresholds()
        by_query = dict(zip((row["query"] for row in rows), results, strict=True))
        expected = json.loads(
            (REPO_ROOT / "samples" / "expected" / "pedido_medico_ficticio.json").read_text("utf-8")
        )
        for line, exam in zip(OCR_LINES, expected["exams"], strict=True):
            result = by_query[line]
            assert gate(result, thresholds) == "accepted", line
            assert result.candidates[0].code == exam["code"], line


class TestGate:
    def result(self, *scores: float) -> SearchResult:
        candidates = [
            {
                "code": f"LAB-1000{i}",
                "name": "x",
                "score": s,
                "matched_on": "name",
                "category": "c",
            }
            for i, s in enumerate(scores)
        ]
        return SearchResult.model_validate(
            {
                "catalog_version": "2026.10.0",
                "catalog_sha256": "0" * 64,
                "query": "q",
                "normalized_query": "q",
                "candidates": candidates,
            }
        )

    @pytest.mark.parametrize(
        ("scores", "decision"),
        [
            ((), "rejected"),
            ((0.9,), "accepted"),
            ((0.9, 0.8), "accepted"),
            ((0.9, 0.86), "review"),
            ((0.8, 0.1), "review"),
            ((0.5,), "rejected"),
            ((0.85, 0.80), "accepted"),
            ((0.6,), "review"),
        ],
    )
    def test_decisions(self, scores: tuple[float, ...], decision: str) -> None:
        assert gate(self.result(*scores), Thresholds(0.85, 0.6, 0.05)) == decision

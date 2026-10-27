from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lab_common.text import normalize_for_search
from rag_mcp.catalog import Catalog, load_catalog
from rag_mcp.index import FieldDoc
from rag_mcp.scoring import (
    ExamSearcher,
    ScoredExam,
    combine,
    coverage,
    fuzzy,
    prepare_query,
    rank,
    score_exam,
    strip_list_marker,
)

pytestmark = pytest.mark.unit

CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "exams_catalog.json"


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog(CATALOG_PATH)


@pytest.fixture(scope="module")
def searcher(catalog: Catalog) -> ExamSearcher:
    return ExamSearcher(catalog)


def doc(text: str, kind: str = "name", code: str = "LAB-00001") -> FieldDoc:
    return FieldDoc(code, kind, text, tuple(text.split()))  # type: ignore[arg-type]


class TestListMarker:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("1. Hemograma completo", "Hemograma completo"),
            ("7. PCR", "PCR"),
            ("12) TSH", "TSH"),
            ("(3) T4 livre", "T4 livre"),
            ("4 - Ureia", "Ureia"),
            ("4 \u2013 Ureia", "Ureia"),
            ("\u2014 Ureia", "Ureia"),
            ("5: Creatinina", "Creatinina"),
            ("1.Hemograma", "Hemograma"),
            ("2º Glicemia", "Glicemia"),
            ("a) Urocultura", "Urocultura"),
            ("- Ferritina", "Ferritina"),
            ("• Ácido úrico", "Ácido úrico"),
            ("  3.  TSH", "TSH"),
        ],
    )
    def test_removes_one_leading_marker(self, query: str, expected: str) -> None:
        assert strip_list_marker(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "25-hidroxivitamina D",
            "Vitamina D (25-OH)",
            "Urina tipo 1",
            "CA 19-9",
            "T4 livre",
            "HbA1c",
            "1 Hemograma completo",
            "Proteinúria de 24 horas",
        ],
    )
    def test_keeps_numbers_that_belong_to_the_name(self, query: str) -> None:
        assert strip_list_marker(query) == query

    def test_removes_only_one_marker(self) -> None:
        assert strip_list_marker("1. 2. TSH") == "2. TSH"

    def test_prepare_query_strips_and_normalizes(self) -> None:
        assert prepare_query("5. Colesterol total e frações") == "colesterol total fracoes"
        assert prepare_query("7. ") == ""


class TestParts:
    def test_fuzzy_is_one_for_equal_text(self) -> None:
        assert fuzzy("tsh", "tsh") == 1.0

    def test_fuzzy_tolerates_typos(self) -> None:
        assert fuzzy("hemogrma completo", "hemograma completo") > 0.9

    def test_fuzzy_of_unrelated_text_is_low(self) -> None:
        assert fuzzy("tsh", "ferritina") < 0.4

    def test_a_shared_single_letter_is_not_a_match(self) -> None:
        assert fuzzy("apolipoproteina b", "antigeno superficie hepatite b") < 0.6

    def test_a_shared_word_still_matches_inside_a_longer_field(self) -> None:
        assert fuzzy("hepatite b", "antigeno superficie hepatite b") > 0.85
        assert fuzzy("vitamina d", "25 hidroxivitamina d") > 0.85

    def test_coverage_counts_distinct_query_tokens(self) -> None:
        assert coverage(["urina", "tipo", "1", "eas"], ["urina", "tipo", "1"]) == 0.75
        assert coverage(["t4", "t4"], ["t4", "livre"]) == 1.0
        assert coverage([], ["t4"]) == 0.0

    def test_combine_weights_and_rounds(self) -> None:
        assert combine(1.0, 0.0) == 0.75
        assert combine(0.0, 1.0) == 0.25
        assert combine(0.9, 0.5) == 0.8
        assert combine(1 / 3, 1 / 3) == 0.3333


class TestScoreExam:
    def test_exact_name(self) -> None:
        scored = score_exam("tsh", [doc("tsh"), doc("tireotrofina", "alias")])
        assert scored == ScoredExam("LAB-00001", 1.0, "exact_name")

    def test_exact_alias(self) -> None:
        scored = score_exam("pcr", [doc("proteina c reativa"), doc("pcr", "alias")])
        assert scored == ScoredExam("LAB-00001", 1.0, "exact_alias")

    def test_fuzzy_reports_the_field_that_matched(self) -> None:
        scored = score_exam("lipidogrma", [doc("colesterol total"), doc("lipidograma", "alias")])
        assert scored.matched_on == "alias"
        assert 0 < scored.score < 1

    def test_tie_prefers_the_name(self) -> None:
        scored = score_exam("xyz", [doc("abc"), doc("def", "alias")])
        assert scored.matched_on == "name"

    def test_needs_a_doc(self) -> None:
        with pytest.raises(ValueError, match="at least one field doc"):
            score_exam("tsh", [])


class TestRank:
    def test_orders_by_score_then_code(self) -> None:
        scored = [
            ScoredExam("LAB-00003", 0.9, "name"),
            ScoredExam("LAB-00002", 0.9, "name"),
            ScoredExam("LAB-00001", 0.5, "name"),
            ScoredExam("LAB-00004", 1.0, "exact_name"),
        ]
        assert [s.code for s in rank(scored, 10)] == [
            "LAB-00004",
            "LAB-00002",
            "LAB-00003",
            "LAB-00001",
        ]

    def test_respects_top_k(self) -> None:
        scored = [ScoredExam(f"LAB-0000{i}", 0.5, "name") for i in range(5)]
        assert len(rank(scored, 2)) == 2

    def test_does_not_depend_on_input_order(self) -> None:
        scored = [ScoredExam(f"LAB-0000{i}", round(i / 10, 1) % 0.3, "name") for i in range(9)]
        assert rank(scored, 9) == rank(list(reversed(scored)), 9)


class TestSearch:
    @pytest.mark.parametrize(
        ("query", "code", "matched_on"),
        [
            ("1. Hemograma completo", "LAB-10001", "exact_name"),
            ("2. Glicemia de jejum", "LAB-10012", "exact_name"),
            ("3. TSH", "LAB-10043", "exact_name"),
            ("4. T4 livre", "LAB-10044", "exact_name"),
            ("5. Colesterol total e frações", "LAB-10016", "exact_name"),
            ("7. PCR", "LAB-10037", "exact_alias"),
            ("HMG", "LAB-10001", "exact_alias"),
            ("GLICEMIA DE JEJUM", "LAB-10012", "exact_name"),
        ],
    )
    def test_exact_mentions_score_one(
        self, searcher: ExamSearcher, query: str, code: str, matched_on: str
    ) -> None:
        top = searcher.search(query, 5).candidates[0]
        assert (top.code, top.score, top.matched_on) == (code, 1.0, matched_on)

    def test_every_catalog_term_is_its_own_exact_match(
        self, catalog: Catalog, searcher: ExamSearcher
    ) -> None:
        for exam in catalog:
            for term in [exam.name, *exam.aliases]:
                top = searcher.search(term, 1).candidates[0]
                assert (top.code, top.score) == (exam.code, 1.0), term

    def test_pcr_alone_beats_rt_pcr_by_a_margin(self, searcher: ExamSearcher) -> None:
        first, second = searcher.search("PCR", 2).candidates
        assert (first.code, second.code) == ("LAB-10037", "LAB-10104")
        assert first.score - second.score >= 0.05

    @pytest.mark.parametrize(
        ("query", "codes"),
        [("T4", ["LAB-10044", "LAB-10045"]), ("T3", ["LAB-10046", "LAB-10047"])],
    )
    def test_ambiguous_mentions_tie_and_break_by_code(
        self, searcher: ExamSearcher, query: str, codes: list[str]
    ) -> None:
        first, second = searcher.search(query, 2).candidates
        assert [first.code, second.code] == codes
        assert first.score == second.score < 1.0

    def test_colesterol_alone_is_not_exact(self, searcher: ExamSearcher) -> None:
        candidates = searcher.search("colesterol", 4).candidates
        assert {c.code for c in candidates} == {"LAB-10016", "LAB-10017", "LAB-10018", "LAB-10019"}
        assert candidates[0].score - candidates[1].score < 0.05
        assert all(c.score < 1.0 for c in candidates)

    @pytest.mark.parametrize(
        ("query", "code"),
        [
            ("Hemogrma completo", "LAB-10001"),
            ("Glicemla de jejum", "LAB-10012"),
            ("6. Urina tipo 1 (EAS)", "LAB-10079"),
        ],
    )
    def test_ocr_errors_and_extra_words_still_rank_first(
        self, searcher: ExamSearcher, query: str, code: str
    ) -> None:
        top = searcher.search(query, 1).candidates[0]
        assert top.code == code
        assert 0.8 < top.score < 1.0

    @pytest.mark.parametrize("query", ["Ressonância magnética de crânio", "Apolipoproteina B"])
    def test_unknown_exam_scores_low(self, searcher: ExamSearcher, query: str) -> None:
        result = searcher.search(query, 5)
        assert all(c.score < 0.6 for c in result.candidates)

    @pytest.mark.parametrize("query", ["|||| ~~ ;;", "de do da", "7. ", "  "])
    def test_nothing_searchable_gives_no_candidates(
        self, searcher: ExamSearcher, query: str
    ) -> None:
        result = searcher.search(query, 5)
        assert (result.normalized_query, result.candidates) == ("", [])

    def test_result_carries_query_and_catalog_identity(
        self, catalog: Catalog, searcher: ExamSearcher
    ) -> None:
        result = searcher.search("1. Hemograma completo", 3)
        assert result.query == "1. Hemograma completo"
        assert result.normalized_query == "hemograma completo"
        assert (result.catalog_version, result.catalog_sha256) == (catalog.version, catalog.sha256)
        assert result.contract_version == "1"
        top = result.candidates[0]
        assert (top.name, top.category) == ("Hemograma completo", "hematologia")


queries = st.one_of(
    st.text(max_size=80),
    st.sampled_from(["T4", "colesterol", "PCR", "hemogrma", "1. TSH", "vitamina"]),
    st.lists(
        st.sampled_from(["total", "livre", "urina", "de", "t4", "pcr", "1", "glicose", "x"]),
        max_size=5,
    ).map(" ".join),
)


class TestProperties:
    @settings(max_examples=300, deadline=None)
    @given(query=queries, top_k=st.integers(min_value=1, max_value=10))
    def test_result_invariants(
        self, catalog: Catalog, searcher: ExamSearcher, query: str, top_k: int
    ) -> None:
        result = searcher.search(query, top_k)
        candidates = result.candidates

        assert len(candidates) <= top_k
        assert all(0.0 <= c.score <= 1.0 for c in candidates)
        assert all(c.code in catalog for c in candidates)
        assert len({c.code for c in candidates}) == len(candidates)
        keys = [(-c.score, c.code) for c in candidates]
        assert keys == sorted(keys)
        assert all(
            (c.score == 1.0) for c in candidates if c.matched_on in ("exact_name", "exact_alias")
        )
        assert result.normalized_query == normalize_for_search(result.normalized_query)

    @settings(max_examples=100, deadline=None)
    @given(query=queries)
    def test_is_deterministic_and_top_k_is_a_prefix(
        self, searcher: ExamSearcher, query: str
    ) -> None:
        full = searcher.search(query, 10)
        assert searcher.search(query, 10) == full
        assert searcher.search(query, 3).candidates == full.candidates[:3]

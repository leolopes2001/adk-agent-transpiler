from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_mcp.catalog import Catalog, load_catalog
from rag_mcp.index import CANDIDATES_PER_RETRIEVER, HybridIndex, trigrams

pytestmark = pytest.mark.unit

CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "exams_catalog.json"


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog(CATALOG_PATH)


@pytest.fixture(scope="module")
def index(catalog: Catalog) -> HybridIndex:
    return HybridIndex(catalog)


class TestFieldDocs:
    def test_one_doc_per_name_and_alias(self, catalog: Catalog, index: HybridIndex) -> None:
        expected = sum(1 + len(exam.aliases) for exam in catalog)
        assert len(index.docs) == expected

    def test_docs_are_normalized(self, index: HybridIndex) -> None:
        docs = index.docs_of("LAB-10016")
        assert [(d.kind, d.text) for d in docs] == [
            ("name", "colesterol total fracoes"),
            ("alias", "lipidograma"),
            ("alias", "perfil lipidico"),
        ]
        assert docs[0].tokens == ("colesterol", "total", "fracoes")

    def test_unknown_code_has_no_docs(self, index: HybridIndex) -> None:
        assert index.docs_of("LAB-99999") == ()

    def test_every_doc_points_to_a_catalog_exam(self, catalog: Catalog, index: HybridIndex) -> None:
        assert {doc.code for doc in index.docs} == {exam.code for exam in catalog}


class TestExact:
    @pytest.mark.parametrize(
        ("text", "code", "kind"),
        [
            ("hemograma completo", "LAB-10001", "name"),
            ("pcr", "LAB-10037", "alias"),
            ("glic jejum", "LAB-10012", "alias"),
            ("t4 livre", "LAB-10044", "name"),
        ],
    )
    def test_finds_normalized_term(
        self, index: HybridIndex, text: str, code: str, kind: str
    ) -> None:
        doc = index.exact(text)
        assert doc is not None
        assert (doc.code, doc.kind) == (code, kind)

    @pytest.mark.parametrize("text", ["t4", "colesterol", "Hemograma completo", "", "hemogrma"])
    def test_no_match_for_non_normalized_or_partial_text(
        self, index: HybridIndex, text: str
    ) -> None:
        assert index.exact(text) is None


class TestTrigrams:
    def test_padding_marks_word_boundaries(self) -> None:
        assert trigrams("tsh") == {"  t", " ts", "tsh", "sh "}

    def test_empty_text_has_none(self) -> None:
        assert trigrams("") == frozenset()

    def test_typo_keeps_most_trigrams(self) -> None:
        original = trigrams("hemograma completo")
        typo = trigrams("hemogrma completo")
        assert len(original & typo) / len(original | typo) > 0.6


class TestRetrievers:
    def test_bm25_finds_docs_sharing_a_token(self, index: HybridIndex) -> None:
        codes = {index.docs[i].code for i in index.bm25_top(["colesterol"])}
        assert {"LAB-10016", "LAB-10017", "LAB-10018", "LAB-10019"} <= codes

    def test_bm25_ignores_unknown_tokens(self, index: HybridIndex) -> None:
        assert index.bm25_top(["xyzzy"]) == []
        assert index.bm25_top([]) == []

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            ("hemogrma completo", "LAB-10001"),
            ("glicemla jejum", "LAB-10012"),
            ("creatinlna", "LAB-10021"),
            ("urocultra", "LAB-10099"),
        ],
    )
    def test_trigrams_recover_ocr_errors(self, index: HybridIndex, text: str, code: str) -> None:
        top = index.trigram_top(text)
        assert index.docs[top[0]].code == code

    def test_trigrams_of_unrelated_text_are_empty(self, index: HybridIndex) -> None:
        assert index.trigram_top("") == []
        assert index.trigram_top("qqq") == []

    def test_limits_are_respected(self, index: HybridIndex) -> None:
        assert len(index.bm25_top(["de", "total", "colesterol", "teste"], limit=3)) <= 3
        assert len(index.trigram_top("a e i o u", limit=5)) <= 5
        assert len(index.trigram_top("ultrassonografia")) <= CANDIDATES_PER_RETRIEVER

    def test_results_are_deterministic(self, index: HybridIndex) -> None:
        assert index.trigram_top("t4") == index.trigram_top("t4")
        assert index.bm25_top(["t4"]) == index.bm25_top(["t4"])


class TestCandidates:
    def test_exact_exam_comes_first(self, index: HybridIndex) -> None:
        assert index.candidate_codes("pcr")[0] == "LAB-10037"

    def test_ambiguous_mention_brings_every_reading(self, index: HybridIndex) -> None:
        codes = index.candidate_codes("t4")
        assert {"LAB-10044", "LAB-10045"} <= set(codes)

    def test_codes_are_unique(self, index: HybridIndex) -> None:
        codes = index.candidate_codes("colesterol total")
        assert len(codes) == len(set(codes))

    def test_empty_query_has_no_candidates(self, index: HybridIndex) -> None:
        assert index.candidate_codes("") == []

    @given(st.text(max_size=60))
    def test_candidates_are_catalog_codes(self, index: HybridIndex, text: str) -> None:
        codes = index.candidate_codes(text)
        known = {doc.code for doc in index.docs}
        assert set(codes) <= known
        assert len(codes) <= 1 + 2 * CANDIDATES_PER_RETRIEVER


def test_empty_catalog_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty catalog"):
        HybridIndex(Catalog(version="2026.10.0", sha256="0" * 64, exams=()))

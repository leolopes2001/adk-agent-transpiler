import pytest
from hypothesis import given
from hypothesis import strategies as st

from lab_common.text import STOPWORDS, normalize_for_search, sanitize, tokens

pytestmark = pytest.mark.unit

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]
BIDI = ["\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069"]


class TestSanitize:
    @pytest.mark.parametrize("char", ZERO_WIDTH + BIDI)
    def test_removes_invisible_and_bidi_characters(self, char: str) -> None:
        result = sanitize(f"Hemo{char}grama", max_chars=100)
        assert result.text == "Hemograma"
        assert result.removed_chars == 1

    def test_removes_control_characters_but_keeps_newlines(self) -> None:
        result = sanitize("Glicose\x00\x07\x1b\nTSH", max_chars=100)
        assert result.text == "Glicose\nTSH"
        assert result.removed_chars == 3

    def test_removes_hangul_filler_that_renders_blank(self) -> None:
        assert sanitize("ignore\u3164previous", max_chars=100).text == "ignoreprevious"

    def test_preserves_accents(self) -> None:
        text = "Ácido úrico, Vitamina D, Coração, Pâncreas"
        assert sanitize(text, max_chars=100).text == text

    def test_composes_decomposed_accents(self) -> None:
        assert sanitize("A\u0301cido u\u0301rico", max_chars=100).text == "Ácido úrico"

    def test_applies_compatibility_normalization(self) -> None:
        assert (
            sanitize("\uff34\uff33\uff28 \ufb01brinogênio", max_chars=100).text
            == "TSH fibrinogênio"
        )

    def test_collapses_whitespace_and_trims_lines(self) -> None:
        raw = "  Hemograma \t\u00a0 completo  \r\n\n\n\n  Glicemia   de jejum  "
        assert sanitize(raw, max_chars=100).text == "Hemograma completo\n\nGlicemia de jejum"

    @pytest.mark.parametrize("separator", ["\r\n", "\r", "\u2028", "\u2029"])
    def test_normalizes_line_separators(self, separator: str) -> None:
        assert sanitize(f"Ureia{separator}Creatinina", max_chars=100).text == "Ureia\nCreatinina"

    def test_truncates_and_flags(self) -> None:
        result = sanitize("abcdefghij", max_chars=4)
        assert result.text == "abcd"
        assert result.truncated is True

    def test_does_not_flag_text_within_limit(self) -> None:
        result = sanitize("abcd", max_chars=4)
        assert result.text == "abcd"
        assert result.truncated is False
        assert result.removed_chars == 0

    def test_limit_applies_after_cleaning(self) -> None:
        assert sanitize("ab\u200b\u200bcd", max_chars=4).truncated is False

    def test_truncation_does_not_leave_trailing_space(self) -> None:
        assert sanitize("abc def", max_chars=4).text == "abc"

    @pytest.mark.parametrize("max_chars", [0, -1])
    def test_rejects_non_positive_limit(self, max_chars: int) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            sanitize("x", max_chars=max_chars)

    @given(st.text(), st.integers(min_value=1, max_value=500))
    def test_is_idempotent(self, text: str, max_chars: int) -> None:
        once = sanitize(text, max_chars)
        twice = sanitize(once.text, max_chars)
        assert twice.text == once.text
        assert twice.removed_chars == 0
        assert twice.truncated is False

    @given(st.text(), st.integers(min_value=1, max_value=500))
    def test_output_respects_limit_and_has_no_removable_characters(
        self, text: str, max_chars: int
    ) -> None:
        result = sanitize(text, max_chars).text
        assert len(result) <= max_chars
        assert not any(c in result for c in ZERO_WIDTH + BIDI + ["\r", "\x00"])


class TestNormalizeForSearch:
    def test_strips_accents_and_lowercases(self) -> None:
        assert normalize_for_search("ÁCIDO ÚRICO") == "acido urico"

    def test_punctuation_becomes_space(self) -> None:
        assert (
            normalize_for_search("T4-livre/TSH,(ultrassensível)") == "t4 livre tsh ultrassensivel"
        )

    def test_underscore_is_punctuation(self) -> None:
        assert normalize_for_search("hdl_colesterol") == "hdl colesterol"

    def test_removes_stopwords(self) -> None:
        assert normalize_for_search("Dosagem de Vitamina D") == "vitamina d"
        assert normalize_for_search("Exame da urina com sedimento") == "urina sedimento"

    def test_only_stopwords_yields_empty(self) -> None:
        assert normalize_for_search("exame de dosagem") == ""

    def test_case_folding_does_not_leave_marks(self) -> None:
        assert normalize_for_search("\u0130NSUL\u0130NA") == "insulina"

    def test_compatibility_characters(self) -> None:
        assert (
            normalize_for_search(
                "\uff28\uff45\uff4d\uff4f\uff47\uff52\uff41\uff4d\uff41 \ufb01brinogênio"
            )
            == "hemograma fibrinogenio"
        )

    @given(st.text())
    def test_is_idempotent(self, text: str) -> None:
        once = normalize_for_search(text)
        assert normalize_for_search(once) == once

    @given(st.text())
    def test_output_has_no_stopwords_or_extra_spaces(self, text: str) -> None:
        result = normalize_for_search(text)
        assert result == result.strip()
        assert "  " not in result
        assert STOPWORDS.isdisjoint(result.split())


class TestTokens:
    def test_keeps_order_and_repetitions(self) -> None:
        assert tokens("Glicose, glicose e Insulina") == ["glicose", "glicose", "insulina"]

    def test_empty_text(self) -> None:
        assert tokens("  ...  ") == []

    @given(st.text())
    def test_matches_normalized_form(self, text: str) -> None:
        assert " ".join(tokens(text)) == normalize_for_search(text)

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lab_common.pii import (
    CpfDetector,
    PiiEngineError,
    PiiEntity,
    PiiRedactor,
    PiiSpan,
    default_detectors,
    high_precision_detectors,
)
from lab_common.pii.redactor import resolve_overlaps

pytestmark = pytest.mark.unit

# Personal data printed on the sample images (scripts/make_sample_images.py).
SAMPLE_VALUES = [
    "Maria Fictícia da Silva",
    "202.393.613-69",
    "20239361369",
    "14/03/1985",
    "(31) 90000-0000",
    "90000-0000",
    "0000 1111 2222 3333",
    "João Exemplo",
    "CRM-MG 000000",
    "Rua Imaginária",
]
SAMPLE_ORDER = """\
Laboratório Exemplo (fictício)
Rua Imaginária, 100 — Belo Horizonte/MG
SOLICITAÇÃO DE EXAMES LABORATORIAIS
Paciente: Maria Fictícia da Silva
CPF: 202.393.613-69
Nascimento: 14/03/1985
Telefone: (31) 90000-0000
Convênio: Plano Exemplo Saúde (fictício)
Carteirinha: 0000 1111 2222 3333
Exames solicitados:
1. Hemograma completo
2. Glicemia de jejum
3. TSH
4. T4 livre
5. Colesterol total e frações
6. Urina tipo 1 (EAS)
7. PCR
Indicação clínica: avaliação de rotina.
Dr. João Exemplo — CRM-MG 000000
Belo Horizonte, 20/10/2026"""


@pytest.fixture(scope="module")
def redactor() -> PiiRedactor:
    return PiiRedactor(default_detectors())


class FixedDetector:
    def __init__(self, *spans: PiiSpan) -> None:
        self.name = "fixed"
        self._spans = list(spans)

    def detect(self, text: str) -> list[PiiSpan]:
        return [s for s in self._spans if s.end <= len(text)]


class FailingDetector:
    name = "failing"

    def detect(self, text: str) -> list[PiiSpan]:
        raise RuntimeError(f"cannot parse {text!r}")


def span(entity: PiiEntity, start: int, end: int, score: float = 1.0) -> PiiSpan:
    return PiiSpan(entity, start, end, score, "test")


class TestSampleOrder:
    def test_masks_every_personal_value(self, redactor: PiiRedactor) -> None:
        result = redactor.redact(SAMPLE_ORDER)

        for value in SAMPLE_VALUES:
            assert value not in result.text
            assert all(value not in line for line in result.lines)
        assert not result.has_residual

    def test_keeps_structure_and_exam_names(self, redactor: PiiRedactor) -> None:
        result = redactor.redact(SAMPLE_ORDER)

        assert result.lines[3] == "Paciente: [PESSOA_1]"
        assert result.lines[4] == "CPF: [CPF_1]"
        assert result.lines[-2] == "Dr. [PESSOA_2] — [CRM_1]"
        for exam in ["Hemograma completo", "T4 livre", "Urina tipo 1 (EAS)", "7. PCR"]:
            assert exam in result.text
        assert "Belo Horizonte, 20/10/2026" in result.text

    def test_counts_masked_spans_per_entity(self, redactor: PiiRedactor) -> None:
        result = redactor.redact(SAMPLE_ORDER)

        assert result.counts == {
            PiiEntity.ADDRESS: 1,
            PiiEntity.PERSON: 2,
            PiiEntity.CPF: 1,
            PiiEntity.BIRTH_DATE: 1,
            PiiEntity.PHONE: 1,
            PiiEntity.HEALTH_CARD: 1,
            PiiEntity.CRM: 1,
        }

    def test_spans_point_into_the_original_text(self, redactor: PiiRedactor) -> None:
        result = redactor.redact(SAMPLE_ORDER)

        [cpf] = [s for s in result.spans if s.entity is PiiEntity.CPF]
        assert SAMPLE_ORDER[cpf.start : cpf.end] == "202.393.613-69"
        assert [s.start for s in result.spans] == sorted(s.start for s in result.spans)


class TestPlaceholders:
    def test_same_value_reuses_its_placeholder(self, redactor: PiiRedactor) -> None:
        text = "Paciente: Maria Fictícia\nDr. João Exemplo\nResponsável: MARIA FICTÍCIA"
        assert redactor.redact(text).lines == (
            "Paciente: [PESSOA_1]",
            "Dr. [PESSOA_2]",
            "Responsável: [PESSOA_1]",
        )

    def test_formatting_does_not_change_the_placeholder(self, redactor: PiiRedactor) -> None:
        result = redactor.redact("CPF 202.393.613-69, confirmar CPF 20239361369")
        assert result.text == "CPF [CPF_1], confirmar CPF [CPF_1]"
        assert result.counts == {PiiEntity.CPF: 2}

    def test_indexes_are_per_entity_in_order_of_appearance(self, redactor: PiiRedactor) -> None:
        text = "Tel (31) 90000-0000\nCPF 202.393.613-69\nCel (31) 90000-0001"
        assert redactor.redact(text).text == "Tel [TELEFONE_1]\nCPF [CPF_1]\nCel [TELEFONE_2]"

    def test_lines_and_text_agree(self, redactor: PiiRedactor) -> None:
        result = redactor.redact(SAMPLE_ORDER)
        assert "\n".join(result.lines) == result.text
        assert len(result.lines) == SAMPLE_ORDER.count("\n") + 1

    def test_span_across_newline_is_masked_on_both_lines(self) -> None:
        text = "abc\ndef"
        result = PiiRedactor([FixedDetector(span(PiiEntity.PERSON, 1, 6))]).redact(text)
        assert result.lines == ("a[PESSOA_1]", "[PESSOA_1]f")

    def test_empty_text(self, redactor: PiiRedactor) -> None:
        result = redactor.redact("")
        assert (result.text, result.lines, result.counts) == ("", ("",), {})


class TestOverlaps:
    def test_highest_score_wins(self) -> None:
        low = span(PiiEntity.PHONE, 0, 10, 0.7)
        high = span(PiiEntity.CPF, 2, 8, 1.0)
        assert resolve_overlaps([low, high]) == [high]

    def test_longest_wins_on_equal_score(self) -> None:
        short = span(PiiEntity.CEP, 0, 4, 0.9)
        long = span(PiiEntity.HEALTH_CARD, 0, 9, 0.9)
        assert resolve_overlaps([short, long]) == [long]

    def test_adjacent_spans_are_both_kept(self) -> None:
        first, second = span(PiiEntity.CPF, 0, 3), span(PiiEntity.PHONE, 3, 6)
        assert resolve_overlaps([second, first]) == [first, second]

    def test_health_card_digits_are_not_split_into_cpf(self, redactor: PiiRedactor) -> None:
        assert redactor.redact("Carteirinha: 0000 1111 2222 3333").text == (
            "Carteirinha: [CARTEIRINHA_1]"
        )


class TestThresholdAndFailures:
    def test_spans_below_min_score_are_kept(self) -> None:
        detector = FixedDetector(span(PiiEntity.PERSON, 0, 3, 0.4))
        result = PiiRedactor([detector]).redact("abc")
        assert (result.text, result.counts) == ("abc", {})

    def test_invalid_cpf_is_masked_at_default_threshold(self, redactor: PiiRedactor) -> None:
        assert redactor.redact("CPF: 202.393.613-60").text == "CPF: [CPF_1]"

    def test_detector_failure_raises_without_text(self) -> None:
        redactor = PiiRedactor([CpfDetector(), FailingDetector()])

        with pytest.raises(PiiEngineError) as excinfo:
            redactor.redact("Paciente: Maria Fictícia da Silva")

        assert "Maria" not in str(excinfo.value)
        assert "RuntimeError" in str(excinfo.value)
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__suppress_context__

    def test_residual_after_masking_is_reported(self) -> None:
        # A detector that also fires on the placeholder it caused simulates a
        # masking bug; the re-scan must catch it.
        class Sticky:
            name = "sticky"

            def detect(self, text: str) -> list[PiiSpan]:
                return [span(PiiEntity.PERSON, 0, 1)] if text else []

        assert PiiRedactor([Sticky()]).redact("x").has_residual

    @pytest.mark.parametrize("min_score", [0.0, 1.5])
    def test_rejects_invalid_min_score(self, min_score: float) -> None:
        with pytest.raises(ValueError, match="min_score"):
            PiiRedactor(default_detectors(), min_score=min_score)

    def test_requires_detectors(self) -> None:
        with pytest.raises(ValueError, match="detector"):
            PiiRedactor([])


class TestHighPrecision:
    def test_only_reports_unambiguous_values(self) -> None:
        redactor = PiiRedactor(high_precision_detectors())
        text = "CPF 202.393.613-69 ou 202.393.613-60, Paciente: Maria Fictícia, tel (31) 90000-0000"

        assert redactor.redact(text).text == (
            "CPF [CPF_1] ou 202.393.613-60, Paciente: Maria Fictícia, tel [TELEFONE_1]"
        )


@given(st.text(max_size=300))
def test_masked_output_never_has_residual(text: str) -> None:
    result = PiiRedactor(default_detectors()).redact(text)
    assert not result.has_residual
    assert "\n".join(result.lines) == result.text

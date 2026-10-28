import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guard import (
    CnsDetector,
    CpfDetector,
    PhoneDetector,
    PiiDetector,
    PiiEntity,
    PiiSpan,
    default_detectors,
    is_valid_cns,
    is_valid_cpf,
)
from pii_guard.redactor import detect_all
from pii_guard.types import inside_hex_token

pytestmark = pytest.mark.unit

CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "pii_cases.jsonl"
# The fictional CPF printed on the sample images.
SAMPLE_CPF = "202.393.613-69"
MIN_SCORE = 0.5


def load_cases() -> list[dict[str, Any]]:
    lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


CASES = load_cases()
DETECTORS: dict[PiiEntity, PiiDetector] = {d.entity: d for d in default_detectors()}


def found(detector: PiiDetector, text: str) -> list[str]:
    return [text[s.start : s.end] for s in detector.detect(text) if s.score >= MIN_SCORE]


def cpf_from(digits: list[int]) -> str:
    for size in (9, 10):
        total = sum(d * w for d, w in zip(digits, range(size + 1, 1, -1), strict=True))
        digits = [*digits, (total * 10 % 11) % 10]
    return "".join(map(str, digits))


def cns_from(prefix: int, body: list[int]) -> str | None:
    """Provisional-style CNS: 14 digits plus whichever last digit closes the mod-11 sum."""
    digits = [prefix, *body]
    total = sum(d * w for d, w in zip(digits, range(15, 1, -1), strict=True))
    last = (-total) % 11
    return "".join(map(str, [*digits, last])) if last < 10 else None


class TestCorpus:
    def test_corpus_covers_every_entity_with_positives_and_negatives(self) -> None:
        entities = {PiiEntity(c["entity"]) for c in CASES}
        assert entities == set(DETECTORS)
        for entity in entities:
            expected = [c["expected"] for c in CASES if c["entity"] == entity]
            assert any(expected), f"{entity} has no positive case"
            assert not all(expected), f"{entity} has no negative case"

    def test_case_ids_are_unique(self) -> None:
        ids = [c["id"] for c in CASES]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
    def test_case(self, case: dict[str, Any]) -> None:
        detector = DETECTORS[PiiEntity(case["entity"])]
        spans = [s for s in detector.detect(case["text"]) if s.score >= MIN_SCORE]

        assert [case["text"][s.start : s.end] for s in spans] == case["expected"]
        if "score" in case:
            assert [s.score for s in spans] == [case["score"]]


class TestCpf:
    def test_sample_cpf_is_valid(self) -> None:
        assert is_valid_cpf(SAMPLE_CPF)
        assert is_valid_cpf("20239361369")

    @pytest.mark.parametrize("value", ["202.393.613-60", "000.000.000-00", "2023936136", ""])
    def test_rejects_invalid_values(self, value: str) -> None:
        assert not is_valid_cpf(value)

    def test_require_valid_drops_format_only_matches(self) -> None:
        detector = CpfDetector(require_valid=True)
        assert found(detector, "CPF 202.393.613-60") == []
        assert found(detector, f"CPF {SAMPLE_CPF}") == [SAMPLE_CPF]

    @given(st.lists(st.integers(0, 9), min_size=9, max_size=9).filter(lambda d: len(set(d)) > 1))
    def test_any_valid_cpf_is_detected_with_full_score(self, base: list[int]) -> None:
        digits = cpf_from(base)
        formatted = f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"
        for value in (digits, formatted):
            [span] = CpfDetector().detect(f"Paciente CPF {value} fim")
            assert span.score == 1.0
            assert len(span) == len(value)


class TestCns:
    @given(st.sampled_from([1, 2, 7, 8, 9]), st.lists(st.integers(0, 9), min_size=13, max_size=13))
    def test_any_valid_cns_is_detected(self, prefix: int, body: list[int]) -> None:
        value = cns_from(prefix, body)
        if value is None:
            return
        assert is_valid_cns(value)
        assert found(CnsDetector(), f"CNS {value}") == [value]


@given(st.text(alphabet=st.characters(exclude_categories=["Nd"]), max_size=200))
def test_text_without_digits_has_no_numeric_identifiers(text: str) -> None:
    for detector in (CpfDetector(), CnsDetector(), PhoneDetector()):
        assert detector.detect(text) == []


hex_ids = st.one_of(
    st.uuids().map(str),
    st.uuids().map(lambda u: str(u).upper()),
    st.binary(min_size=1, max_size=64).map(lambda b: hashlib.sha256(b).hexdigest()),
    st.binary(min_size=1, max_size=64).map(lambda b: hashlib.sha256(b).hexdigest().upper()),
    # A short id made only of digits ("1120000000") is a valid phone number, not a hex id.
    st.binary(min_size=4, max_size=16).map(bytes.hex).filter(lambda h: not h.isdigit()),
)


class TestHexTokens:
    """UUIDs, hashes and hex ids in logs never read as personal data."""

    @settings(max_examples=500)
    @given(value=hex_ids)
    def test_no_detector_fires_on_a_hex_id(self, value: str) -> None:
        assert detect_all(default_detectors(), value, min_score=0.0) == []

    @settings(max_examples=300)
    @given(value=hex_ids, prefix=st.sampled_from(["", "request_id=", "sha256:", "id ", "("]))
    def test_nor_inside_a_log_line(self, value: str, prefix: str) -> None:
        assert detect_all(default_detectors(), f"{prefix}{value} done", min_score=0.0) == []

    @pytest.mark.parametrize(
        "value",
        ["2f3b5fd6-2866-4383-bebd-02cf03c069af", "28eb7825865727be894151836eef70b1"],
    )
    def test_known_false_positives(self, value: str) -> None:
        assert detect_all(default_detectors(), value, min_score=0.0) == []

    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("Telefone: (31) 90000-0000 / id 2f3b5fd6", "(31) 90000-0000"),
            ("CPF20239361369", "20239361369"),
            ("Tel. 31-90000-0000", "90000-0000"),
        ],
    )
    def test_real_values_next_to_letters_are_still_found(self, text: str, value: str) -> None:
        spans = detect_all(default_detectors(), text, min_score=MIN_SCORE)
        assert value in [text[s.start : s.end] for s in spans]

    @pytest.mark.parametrize(
        ("text", "start", "end", "inside"),
        [
            ("ab12345678cd", 2, 10, True),
            ("x-12345678", 2, 10, False),
            ("af-12345678", 3, 11, True),
            ("12-12345678", 3, 11, False),
            ("Fone12345678", 4, 12, False),
            ("12345678-beef", 0, 8, True),
            ("12345678 beef", 0, 8, False),
        ],
    )
    def test_inside_hex_token(self, text: str, start: int, end: int, inside: bool) -> None:
        assert inside_hex_token(text, start, end) is inside


def test_spans_hold_offsets_not_values() -> None:
    text = f"CPF: {SAMPLE_CPF}"
    [span] = CpfDetector().detect(text)
    assert SAMPLE_CPF not in repr(span)
    assert text[span.start : span.end] == SAMPLE_CPF


@pytest.mark.parametrize(("start", "end", "score"), [(3, 3, 1.0), (-1, 2, 1.0), (0, 2, 1.5)])
def test_span_rejects_invalid_fields(start: int, end: int, score: float) -> None:
    with pytest.raises(ValueError, match=r"span offsets|score"):
        PiiSpan(PiiEntity.CPF, start, end, score, "cpf")


def test_placeholder_labels_are_distinct() -> None:
    labels = [entity.placeholder_label for entity in PiiEntity]
    assert len(labels) == len(set(labels))
    assert PiiEntity.PHONE.placeholder_label == "TELEFONE"

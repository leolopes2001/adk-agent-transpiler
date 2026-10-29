import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lab_common.contracts.ocr import PATIENT_REF_PATTERN
from ocr_mcp.identity import find_patient_ref, normalize_identifier, patient_ref
from pii_guard import PiiEntity, PiiRedactor, default_detectors
from scheduler_api.schemas import PATIENT_REF_PATTERN as API_PATIENT_REF_PATTERN

pytestmark = pytest.mark.unit

SECRET = b"s" * 32
OTHER_SECRET = b"t" * 32
# Patient data printed on the sample images (all fictional).
CPF = "202.393.613-69"
NAME = "Maria Fictícia da Silva"
# Format the scheduler API accepts (scheduler_api.schemas.PATIENT_REF_PATTERN).
REF_REGEX = re.compile(r"^pt_[A-Z2-7]{26}$")


def ref_of(text: str, secret: bytes = SECRET) -> str | None:
    spans = PiiRedactor(default_detectors()).redact(text).spans
    return find_patient_ref(secret, text, spans)


class TestFormat:
    def test_matches_the_api_pattern(self) -> None:
        ref = patient_ref(SECRET, PiiEntity.CPF, CPF)

        assert REF_REGEX.fullmatch(ref)
        assert re.fullmatch(API_PATIENT_REF_PATTERN, ref)
        assert PATIENT_REF_PATTERN == API_PATIENT_REF_PATTERN

    @given(
        st.binary(min_size=32, max_size=64),
        st.sampled_from([PiiEntity.CPF, PiiEntity.CNS, PiiEntity.PERSON]),
        st.text(min_size=1, max_size=40),
    )
    def test_any_input_yields_a_valid_ref(self, secret: bytes, kind: PiiEntity, value: str) -> None:
        assert REF_REGEX.fullmatch(patient_ref(secret, kind, value))


class TestDeterminism:
    def test_same_input_same_ref(self) -> None:
        assert patient_ref(SECRET, PiiEntity.CPF, CPF) == patient_ref(SECRET, PiiEntity.CPF, CPF)

    def test_depends_on_the_secret(self) -> None:
        assert patient_ref(SECRET, PiiEntity.CPF, CPF) != patient_ref(
            OTHER_SECRET, PiiEntity.CPF, CPF
        )

    def test_cpf_mask_does_not_matter(self) -> None:
        assert patient_ref(SECRET, PiiEntity.CPF, CPF) == patient_ref(
            SECRET, PiiEntity.CPF, "20239361369"
        )

    def test_name_case_accents_and_spacing_do_not_matter(self) -> None:
        assert patient_ref(SECRET, PiiEntity.PERSON, NAME) == patient_ref(
            SECRET, PiiEntity.PERSON, "MARIA  FICTICIA DA SILVA"
        )

    def test_kinds_do_not_collide(self) -> None:
        digits = "100000000000007"
        assert patient_ref(SECRET, PiiEntity.CNS, digits) != patient_ref(
            SECRET, PiiEntity.PERSON, digits
        )

    def test_ref_does_not_contain_the_identifier(self) -> None:
        ref = patient_ref(SECRET, PiiEntity.CPF, CPF)
        assert "20239361369" not in ref
        assert "MARIA" not in ref


class TestNormalization:
    def test_digits_only_for_documents(self) -> None:
        assert normalize_identifier(PiiEntity.CPF, CPF) == "20239361369"
        assert normalize_identifier(PiiEntity.CNS, "100 0000 0000 0007") == "100000000000007"

    def test_name_tokens(self) -> None:
        assert normalize_identifier(PiiEntity.PERSON, NAME) == "maria ficticia silva"

    def test_other_entities_are_not_identifiers(self) -> None:
        with pytest.raises(ValueError, match="not a patient identifier"):
            normalize_identifier(PiiEntity.PHONE, "(31) 90000-0000")


class TestFindPatientRef:
    def test_prefers_valid_cpf(self) -> None:
        text = f"Paciente: {NAME}\nCNS: 100 0000 0000 0007\nCPF: {CPF}"
        assert ref_of(text) == patient_ref(SECRET, PiiEntity.CPF, CPF)

    def test_same_patient_on_another_order_gets_the_same_ref(self) -> None:
        first = f"Paciente: {NAME}\nCPF: {CPF}\n1. TSH"
        second = "CPF 20239361369\nPaciente: MARIA FICTÍCIA DA SILVA\n1. Hemograma completo"
        assert ref_of(first) == ref_of(second)

    def test_falls_back_to_cns(self) -> None:
        text = f"Paciente: {NAME}\nCNS: 100 0000 0000 0007"
        assert ref_of(text) == patient_ref(SECRET, PiiEntity.CNS, "100000000000007")

    def test_skips_cpf_with_invalid_check_digits(self) -> None:
        text = f"Paciente: {NAME}\nCPF: 202.393.613-60"
        assert ref_of(text) == patient_ref(SECRET, PiiEntity.PERSON, NAME)

    def test_doctor_name_is_not_the_patient(self) -> None:
        assert ref_of("Dr. João Exemplo — CRM-MG 000000\n1. TSH") is None

    def test_no_identifier(self) -> None:
        assert ref_of("Exames solicitados:\n1. Hemograma completo") is None

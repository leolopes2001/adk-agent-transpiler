import json
from pathlib import Path

import pytest

from lab_common.text import tokens
from pii_guard import PersonDetector
from pii_guard.person import medical_terms

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG = REPO_ROOT / "data" / "exams_catalog.json"
# Header and signature lines as printed on samples/pedido_medico_ficticio.png.
SAMPLE_ORDER = """\
Laboratório Exemplo (fictício)
SOLICITAÇÃO DE EXAMES LABORATORIAIS
Paciente: Maria Fictícia da Silva
CPF: 202.393.613-69
Exames solicitados:
1. Hemograma completo
7. PCR
Indicação clínica: avaliação de rotina.
Dr. João Exemplo — CRM-MG 000000
"""


def names(text: str, detector: PersonDetector | None = None) -> list[str]:
    return [text[s.start : s.end] for s in (detector or PersonDetector()).detect(text)]


def test_finds_patient_and_doctor_on_the_sample_order() -> None:
    spans = PersonDetector().detect(SAMPLE_ORDER)

    assert [SAMPLE_ORDER[s.start : s.end] for s in spans] == [
        "Maria Fictícia da Silva",
        "João Exemplo",
    ]
    assert [s.detector for s in spans] == ["person.patient", "person.professional"]


@pytest.mark.parametrize(
    ("text", "role"),
    [
        ("Paciente: Maria Fictícia", "patient"),
        ("Nome do paciente: Maria Fictícia", "patient"),
        ("Sr. João Exemplo", "patient"),
        ("Dra. Maria Fictícia", "professional"),
        ("Médica responsável: Maria Fictícia", "professional"),
        ("Nome da mãe: Maria Fictícia", "other"),
        ("Acompanhante: João Exemplo", "other"),
    ],
)
def test_detector_name_records_the_label_role(text: str, role: str) -> None:
    [span] = PersonDetector().detect(text)
    assert span.detector == f"person.{role}"


def test_label_is_not_part_of_the_span() -> None:
    assert names("Paciente:   Maria Fictícia da Silva") == ["Maria Fictícia da Silva"]


def test_long_names_are_masked_whole() -> None:
    text = "Paciente: Maria Fictícia Exemplo de Souza Lima dos Santos Pereira"
    assert names(text) == [text.removeprefix("Paciente: ")]


def test_trailing_connector_is_left_out() -> None:
    assert names("Paciente: Maria Fictícia da\nSilva") == ["Maria Fictícia"]


def test_name_does_not_continue_on_the_next_line() -> None:
    assert names("Paciente: Maria Fictícia\nHemograma Completo") == ["Maria Fictícia"]


@pytest.mark.parametrize("field", ["CPF", "Telefone", "RG", "Nascimento", "Convênio"])
def test_name_stops_at_a_field_label(field: str) -> None:
    assert names(f"Paciente: Maria Fictícia {field} 000") == ["Maria Fictícia"]


def test_placeholders_are_not_names() -> None:
    assert names("Paciente: [PESSOA_1]\nDr. [PESSOA_2] — [CRM_1]") == []


def test_mixed_name_with_a_medical_word_is_still_a_name() -> None:
    # "Ferro" is an exam; a name containing it is still personal data.
    assert names("Paciente: Maria Ferro") == ["Maria Ferro"]


def test_extra_terms_extend_the_exclusion_list() -> None:
    assert names("Paciente: Exemplo") == ["Exemplo"]
    assert names("Paciente: Exemplo", PersonDetector(extra_terms=["exemplo"])) == []


def test_exclusion_list_contains_every_catalog_token() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog_tokens = {
        token
        for exam in catalog["exams"]
        for name in [exam["name"], *exam["aliases"]]
        for token in tokens(name)
        if not token.isdigit()
    }

    missing = sorted(catalog_tokens - medical_terms())
    assert not missing, f"add these catalog tokens to medical_terms.txt: {missing}"


def test_exclusion_list_has_no_common_given_names() -> None:
    assert not {"maria", "joao", "silva", "souza", "exemplo"} & medical_terms()


# Audit F01: a labeled field is personal data whatever the case of its value,
# and the value may come on the line after the label.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Paciente: maria fictícia da silva", ["maria fictícia da silva"]),
        ("Paciente: Maria FICTÍCIA da silva", ["Maria FICTÍCIA da silva"]),
        ("paciente: maria ficticia", ["maria ficticia"]),
        ("PACIENTE: Maria fictícia", ["Maria fictícia"]),
        ("Nome: joana ficcional teixeira", ["joana ficcional teixeira"]),
        ("Responsável: joana FICCIONAL teixeira", ["joana FICCIONAL teixeira"]),
        ("Médico solicitante: joão exemplo", ["joão exemplo"]),
        ("dra. beatriz Inventada — CRM-MG 000000", ["beatriz Inventada"]),
        ("Dr. joão exemplo", ["joão exemplo"]),
        ("Sra. maria fictícia", ["maria fictícia"]),
    ],
)
def test_labeled_value_is_masked_whatever_its_case(text: str, expected: list[str]) -> None:
    assert names(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Paciente:\nmaria fictícia da silva\nCPF: 000", ["maria fictícia da silva"]),
        ("Responsável:\n  joana FICCIONAL teixeira", ["joana FICCIONAL teixeira"]),
        ("Paciente:\nMaria Fictícia CPF 000", ["Maria Fictícia"]),
        ("Paciente:\r\nmaria fictícia", ["maria fictícia"]),
    ],
)
def test_value_on_the_line_after_the_label(text: str, expected: list[str]) -> None:
    assert names(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Paciente: maria silva cpf 000", ["maria silva"]),
        ("Paciente: maria silva Sexo: F", ["maria silva"]),
        ("Paciente: maria silva, 42 anos", ["maria silva"]),
        ("Paciente: maria silva — Data: 01/01/2026", ["maria silva"]),
    ],
)
def test_lowercase_value_ends_at_the_next_field(text: str, expected: list[str]) -> None:
    assert names(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Paciente: hemograma completo",
        "paciente: em jejum",
        "Paciente:\n1. Hemograma completo",
        "Médico solicitante:\nExames solicitados:",
        "Paciente:\n[PESSOA_1]",
        "Nome do exame: hemograma completo",
    ],
)
def test_lowercase_exams_and_labels_after_a_label_are_not_names(text: str) -> None:
    assert names(text) == []

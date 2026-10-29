import json
from pathlib import Path

import pytest

from pii_guard import catalog_terms, has_personal_label
from pii_guard.labels import PERSONAL_LABELS, SECTION_TITLES
from pii_guard.text import tokens

pytestmark = pytest.mark.unit

CATALOG = Path(__file__).resolve().parents[3] / "data" / "exams_catalog.json"


@pytest.mark.parametrize(
    "line",
    [
        "Paciente: [PESSOA_1]",
        "CPF: [CPF_1]",
        "CPE: 202.393.613-69",
        "“PATA DE NASCIMENTO. [DATA_NASC_1]",
        "Dr. [PESSOA_2] — [CRM_1]",
        "DR ROBERTO",
        "Convênio: Plano Exemplo Saúde",
        "Mãe: [PESSOA_3]",
        "OBS. PACIENTE EM JEJUM DE 8 HORAS",
        "Contato: [TELEFONE_1]",
    ],
)
def test_lines_with_a_personal_data_label(line: str) -> None:
    assert has_personal_label(line)


@pytest.mark.parametrize(
    "line",
    [
        "1. Hemograma completo",
        "Exames solicitados:",
        "SOLICITAÇÃO DE EXAMES LABORATORIAIS",
        "4. FERRLTINA",
        "1. PAI",
        "Indicação clínica: avaliação de rotina.",
    ],
)
def test_exam_and_section_lines_have_no_personal_data_label(line: str) -> None:
    assert not has_personal_label(line)


def test_no_label_is_a_word_of_the_catalog() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    words = {
        token
        for exam in catalog["exams"]
        for name in [exam["name"], *exam["aliases"]]
        for token in tokens(name)
    }

    assert not words & (PERSONAL_LABELS | SECTION_TITLES)


def test_catalog_terms_are_only_the_catalog_section() -> None:
    terms = catalog_terms()

    assert {"hemograma", "ferritina", "glicemia"} <= terms
    assert not {"paciente", "chagas", "coombs"} & terms

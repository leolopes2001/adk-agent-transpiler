"""Exam names must survive masking: a masked exam is an exam the agent cannot schedule.

The rules are checked here; the same sweep with the NER model, slower, is in
tests/security/test_catalog_ner.py.
"""

import json
from pathlib import Path

import pytest

from pii_guard import PersonDetector, PiiEntity, PiiRedactor, default_detectors

pytestmark = pytest.mark.unit

CATALOG = Path(__file__).resolve().parents[3] / "data" / "exams_catalog.json"

# How exam names appear on an order. None of them is a person label.
NEUTRAL_CONTEXTS = ["{}", "1. {}", "- {}", "Solicito: {}", "Exames: {}, Hemograma completo"]
# Labels that do introduce a name: an exam right after them still is not one.
LABEL_CONTEXTS = ["Paciente: {}", "Nome: {}", "Dr. {}", "Médico solicitante: {}", "Sra. {}"]


def catalog_mentions() -> list[str]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    return sorted({m for exam in catalog["exams"] for m in [exam["name"], *exam["aliases"]]})


MENTIONS = catalog_mentions()


@pytest.fixture(scope="module")
def redactor() -> PiiRedactor:
    return PiiRedactor(default_detectors())


def test_catalog_has_names_and_aliases() -> None:
    assert len(MENTIONS) > 120


@pytest.mark.parametrize("case", ["as_written", "upper"])
@pytest.mark.parametrize("context", NEUTRAL_CONTEXTS)
def test_no_detector_fires_on_catalog_mentions(
    redactor: PiiRedactor, context: str, case: str
) -> None:
    masked = {
        mention: result.text
        for mention in MENTIONS
        if (result := redactor.redact(context.format(_cased(mention, case)))).spans
    }
    assert masked == {}


@pytest.mark.parametrize("context", LABEL_CONTEXTS)
def test_catalog_mentions_after_person_labels_are_not_names(context: str) -> None:
    detector = PersonDetector()
    as_person = [m for m in MENTIONS if detector.detect(context.format(m))]
    assert as_person == []


@pytest.mark.parametrize("case", ["upper", "title"])
def test_case_variants_are_not_names(case: str) -> None:
    detector = PersonDetector()
    as_person = [m for m in MENTIONS if detector.detect(f"Paciente: {getattr(m, case)()}")]
    assert as_person == []


def test_whole_catalog_as_one_order_keeps_every_mention(redactor: PiiRedactor) -> None:
    order = "Paciente: Maria Fictícia da Silva\n" + "\n".join(
        f"{n}. {m}" for n, m in enumerate(MENTIONS, start=1)
    )
    result = redactor.redact(order)

    assert result.counts == {PiiEntity.PERSON: 1}
    assert result.lines[1:] == tuple(order.splitlines()[1:])


def _cased(mention: str, case: str) -> str:
    return mention.upper() if case == "upper" else mention

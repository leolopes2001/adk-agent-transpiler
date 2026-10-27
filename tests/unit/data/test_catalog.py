import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from lab_common.text import normalize_for_search

pytestmark = pytest.mark.unit

CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "exams_catalog.json"
CODE_PATTERN = re.compile(r"LAB-\d{5}")
FIELDS = {"code", "name", "aliases", "category", "specimen", "preparation"}

# Exams the sample images, expected outputs and later tests depend on.
REQUIRED_NAMES = [
    "Hemograma completo",
    "Glicemia de jejum",
    "Hemoglobina glicada",
    "TSH",
    "T4 livre",
    "T4 total",
    "T3 total",
    "Colesterol total e frações",
    "Triglicerídeos",
    "Creatinina",
    "Ureia",
    "TGO (AST)",
    "TGP (ALT)",
    "Gama GT",
    "Proteína C reativa",
    "RT-PCR para SARS-CoV-2",
    "Urina tipo 1",
    "Urocultura",
    "Parasitológico de fezes",
    "Vitamina D (25-OH)",
    "Vitamina B12",
    "Ferritina",
    "Ácido úrico",
    "Sódio",
    "Potássio",
    "Tempo de protrombina",
    "PSA total",
    "Beta-HCG",
    "Ultrassonografia de abdome total",
]

REQUIRED_ALIASES = {
    "glicose": "Glicemia de jejum",
    "glic jejum": "Glicemia de jejum",
    "HbA1c": "Hemoglobina glicada",
    "T4L": "T4 livre",
    "lipidograma": "Colesterol total e frações",
    "perfil lipídico": "Colesterol total e frações",
    "AST": "TGO (AST)",
    "ALT": "TGP (ALT)",
    "PCR": "Proteína C reativa",
    "RT-PCR covid": "RT-PCR para SARS-CoV-2",
    "EAS": "Urina tipo 1",
    "urina rotina": "Urina tipo 1",
    "EPF": "Parasitológico de fezes",
    "TP": "Tempo de protrombina",
    "INR": "Tempo de protrombina",
}

# Mentions that must stay ambiguous so the confidence gate has something to decide.
AMBIGUOUS_TERMS = ["T4", "T3", "colesterol"]


@pytest.fixture(scope="module")
def catalog() -> dict[str, Any]:
    with CATALOG_PATH.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


@pytest.fixture(scope="module")
def exams(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = catalog["exams"]
    return result


@pytest.fixture(scope="module")
def term_index(exams: list[dict[str, Any]]) -> dict[str, str]:
    """Normalized name or alias -> exam name (uniqueness is asserted separately)."""
    return {
        normalize_for_search(term): exam["name"]
        for exam in exams
        for term in [exam["name"], *exam["aliases"]]
    }


def test_has_version(catalog: dict[str, Any]) -> None:
    assert re.fullmatch(r"\d{4}\.\d{2}\.\d+", catalog["catalog_version"])


def test_has_at_least_100_exams(exams: list[dict[str, Any]]) -> None:
    assert len(exams) >= 100


def test_every_exam_has_exactly_the_expected_non_empty_fields(
    exams: list[dict[str, Any]],
) -> None:
    for exam in exams:
        assert set(exam) == FIELDS, exam["code"]
        for field in FIELDS - {"aliases"}:
            assert isinstance(exam[field], str), (exam["code"], field)
            assert exam[field].strip(), (exam["code"], field)
        assert isinstance(exam["aliases"], list), exam["code"]
        assert all(isinstance(a, str) and a.strip() for a in exam["aliases"]), exam["code"]


def test_codes_match_format(exams: list[dict[str, Any]]) -> None:
    invalid = [e["code"] for e in exams if not CODE_PATTERN.fullmatch(e["code"])]
    assert invalid == []


def test_codes_are_unique(exams: list[dict[str, Any]]) -> None:
    duplicates = [c for c, n in Counter(e["code"] for e in exams).items() if n > 1]
    assert duplicates == []


def test_names_and_aliases_are_unique_after_normalization(exams: list[dict[str, Any]]) -> None:
    owners: dict[str, set[str]] = {}
    occurrences: Counter[str] = Counter()
    for exam in exams:
        for term in [exam["name"], *exam["aliases"]]:
            key = normalize_for_search(term)
            owners.setdefault(key, set()).add(exam["code"])
            occurrences[key] += 1

    collisions = {key: sorted(owners[key]) for key, n in occurrences.items() if n > 1}
    assert collisions == {}


def test_no_name_or_alias_normalizes_to_empty(exams: list[dict[str, Any]]) -> None:
    empty = [
        term
        for exam in exams
        for term in [exam["name"], *exam["aliases"]]
        if not normalize_for_search(term)
    ]
    assert empty == []


def test_required_exams_are_present(exams: list[dict[str, Any]]) -> None:
    names = {e["name"] for e in exams}
    assert [n for n in REQUIRED_NAMES if n not in names] == []


@pytest.mark.parametrize(("alias", "name"), REQUIRED_ALIASES.items())
def test_required_alias_points_to_expected_exam(
    term_index: dict[str, str], alias: str, name: str
) -> None:
    assert term_index.get(normalize_for_search(alias)) == name


@pytest.mark.parametrize("term", AMBIGUOUS_TERMS)
def test_ambiguous_term_has_no_exact_match(term_index: dict[str, str], term: str) -> None:
    assert normalize_for_search(term) not in term_index

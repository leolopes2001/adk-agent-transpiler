import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lab_common.contracts.rag import Exam
from rag_mcp.catalog import MIN_EXAMS, CatalogError, load_catalog, parse_catalog

pytestmark = pytest.mark.unit

CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "exams_catalog.json"


def exam(index: int, **fields: Any) -> dict[str, Any]:
    return {
        "code": f"LAB-{20000 + index:05d}",
        "name": f"Analito {index} quantitativo",
        "aliases": [f"sigla {index}"],
        "category": "bioquimica",
        "specimen": "sangue",
        "preparation": "Não requer jejum.",
    } | fields


def catalog_bytes(exams: list[dict[str, Any]], version: str = "2026.10.0") -> bytes:
    return json.dumps({"catalog_version": version, "exams": exams}).encode("utf-8")


def valid_exams(count: int = MIN_EXAMS) -> list[dict[str, Any]]:
    return [exam(i) for i in range(count)]


class TestVersionedCatalog:
    def test_loads(self) -> None:
        catalog = load_catalog(CATALOG_PATH)

        assert catalog.version == "2026.10.0"
        assert len(catalog) == 120
        assert catalog.sha256 == hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()

    def test_lookup_by_code(self) -> None:
        catalog = load_catalog(CATALOG_PATH)

        hemograma = catalog.get("LAB-10001")
        assert hemograma is not None
        assert hemograma.name == "Hemograma completo"
        assert "LAB-10001" in catalog
        assert catalog.get("LAB-99999") is None
        assert "LAB-99999" not in catalog

    def test_info(self) -> None:
        catalog = load_catalog(CATALOG_PATH)

        info = catalog.info()
        assert (info.catalog_version, info.exam_count) == ("2026.10.0", 120)
        assert info.catalog_sha256 == catalog.sha256

    def test_iterates_in_file_order(self) -> None:
        catalog = load_catalog(CATALOG_PATH)

        codes = [e.code for e in catalog]
        assert codes[0] == "LAB-10001"
        assert all(isinstance(e, Exam) for e in catalog)


class TestValidCatalog:
    def test_minimum_size_is_accepted(self) -> None:
        catalog = parse_catalog(catalog_bytes(valid_exams()))
        assert len(catalog) == MIN_EXAMS

    def test_exam_without_aliases_is_accepted(self) -> None:
        exams = valid_exams()
        exams[0]["aliases"] = []
        assert parse_catalog(catalog_bytes(exams)).exams[0].aliases == []

    def test_hash_changes_with_the_file(self) -> None:
        exams = valid_exams()
        first = parse_catalog(catalog_bytes(exams))
        exams[0]["preparation"] = "Jejum de 8 horas."
        second = parse_catalog(catalog_bytes(exams))

        assert first.sha256 != second.sha256

    def test_min_exams_can_be_lowered(self) -> None:
        assert len(parse_catalog(catalog_bytes(valid_exams(3)), min_exams=3)) == 3


class TestInvalidCatalog:
    def test_too_few_exams(self) -> None:
        with pytest.raises(CatalogError, match="has 99 exams; at least 100 are required"):
            parse_catalog(catalog_bytes(valid_exams(MIN_EXAMS - 1)))

    def test_duplicate_code(self) -> None:
        exams = valid_exams()
        exams[5]["code"] = exams[2]["code"]
        with pytest.raises(CatalogError, match="duplicate exam code LAB-20002"):
            parse_catalog(catalog_bytes(exams))

    @pytest.mark.parametrize(
        "code", ["LAB-1234", "LAB-123456", "lab-12345", "XYZ-12345", "LAB-١٢٣٤٥"]
    )
    def test_invalid_code(self, code: str) -> None:
        exams = valid_exams()
        exams[3]["code"] = code
        with pytest.raises(CatalogError, match=r"invalid catalog: exams\.3\.code"):
            parse_catalog(catalog_bytes(exams))

    def test_alias_colliding_with_another_exams_alias(self) -> None:
        exams = valid_exams()
        exams[1]["aliases"] = ["SIGLA, 0"]
        expected = "LAB-20001: 'SIGLA, 0' collides with 'sigla 0' of LAB-20000"
        with pytest.raises(CatalogError, match=expected):
            parse_catalog(catalog_bytes(exams))

    def test_alias_colliding_with_another_exams_name(self) -> None:
        exams = valid_exams()
        exams[7]["aliases"].append("dosagem de analito 3 quantitativo")
        with pytest.raises(CatalogError, match="'Analito 3 quantitativo' of LAB-20003"):
            parse_catalog(catalog_bytes(exams))

    def test_names_colliding_after_normalization(self) -> None:
        exams = valid_exams()
        exams[9]["name"] = "ANÁLITO 4 (quantitativo)"
        with pytest.raises(CatalogError, match="both normalize to 'analito 4 quantitativo'"):
            parse_catalog(catalog_bytes(exams))

    def test_repeated_alias_within_one_exam(self) -> None:
        exams = valid_exams()
        exams[0]["aliases"] = ["sigla", "Sígla"]
        with pytest.raises(CatalogError, match="collides with 'sigla' of LAB-20000"):
            parse_catalog(catalog_bytes(exams))

    def test_alias_without_searchable_characters(self) -> None:
        exams = valid_exams()
        exams[4]["aliases"] = ["de - do"]
        with pytest.raises(CatalogError, match="LAB-20004: 'de - do' has no searchable"):
            parse_catalog(catalog_bytes(exams))

    @pytest.mark.parametrize("field", ["name", "category", "specimen", "preparation"])
    def test_empty_field(self, field: str) -> None:
        exams = valid_exams()
        exams[0][field] = ""
        with pytest.raises(CatalogError, match=rf"exams\.0\.{field}"):
            parse_catalog(catalog_bytes(exams))

    def test_missing_field(self) -> None:
        exams = valid_exams()
        del exams[0]["specimen"]
        with pytest.raises(CatalogError, match=r"exams\.0\.specimen: Field required"):
            parse_catalog(catalog_bytes(exams))

    def test_unknown_field(self) -> None:
        exams = valid_exams()
        exams[0]["price"] = "10"
        with pytest.raises(CatalogError, match=r"exams\.0\.price: Extra inputs"):
            parse_catalog(catalog_bytes(exams))

    @pytest.mark.parametrize("version", ["2026.10", "v1", ""])
    def test_invalid_version(self, version: str) -> None:
        with pytest.raises(CatalogError, match="catalog_version"):
            parse_catalog(catalog_bytes(valid_exams(), version=version))

    def test_many_errors_are_summarized(self) -> None:
        exams = [exam(i, code="bad") for i in range(MIN_EXAMS)]
        with pytest.raises(CatalogError, match=r"\(and 95 more\)$"):
            parse_catalog(catalog_bytes(exams))

    @pytest.mark.parametrize("raw", [b"{", b"\xff\xfe", b"[]", b"null"])
    def test_not_a_catalog_document(self, raw: bytes) -> None:
        with pytest.raises(CatalogError):
            parse_catalog(raw)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogError, match="cannot read catalog"):
            load_catalog(tmp_path / "missing.json")

    def test_errors_do_not_chain_the_cause(self) -> None:
        with pytest.raises(CatalogError) as info:
            parse_catalog(b"{")
        assert info.value.__cause__ is None
        assert info.value.__suppress_context__


def test_loading_does_not_mutate_input() -> None:
    exams = valid_exams()
    snapshot = copy.deepcopy(exams)
    parse_catalog(catalog_bytes(exams))
    assert exams == snapshot


def test_catalog_is_immutable() -> None:
    catalog = parse_catalog(catalog_bytes(valid_exams()))
    with pytest.raises(AttributeError):
        catalog.version = "x"  # type: ignore[misc]

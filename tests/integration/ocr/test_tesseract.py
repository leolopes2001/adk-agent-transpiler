"""Real Tesseract on the generated sample images.

Requires `tesseract` with Portuguese data on PATH (installed in CI and in the
OCR container). A missing binary fails these tests instead of skipping them,
so a broken environment cannot pass silently.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from rapidfuzz import fuzz

from ocr_mcp.engine import TesseractEngine
from ocr_mcp.errors import ErrorCode, OcrError
from ocr_mcp.preprocess import preprocess

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE = REPO_ROOT / "samples" / "pedido_medico_ficticio.png"
IMAGES = REPO_ROOT / "tests" / "data" / "images"
EXAMS = [
    "Hemograma completo",
    "Glicemia de jejum",
    "TSH",
    "T4 livre",
    "Colesterol total e frações",
    "Urina tipo 1 (EAS)",
    "PCR",
]
MIN_FUZZY_SCORE = 85


@pytest.fixture(scope="module")
def engine() -> TesseractEngine:
    if shutil.which("tesseract") is None:
        pytest.fail("tesseract is not installed (needs tesseract-ocr and tesseract-ocr-por)")
    return TesseractEngine()


def recognize(engine: TesseractEngine, path: Path) -> list[str]:
    pixels = np.asarray(Image.open(path).convert("L"))
    return [line.text for line in engine.recognize(preprocess(pixels).image, timeout_s=30)]


def test_portuguese_data_is_installed(engine: TesseractEngine) -> None:
    assert engine.version()
    lines = recognize(engine, SAMPLE)
    assert any("SOLICITAÇÃO" in line for line in lines), "accented output needs the 'por' model"


def test_sample_order_lists_all_exams(engine: TesseractEngine) -> None:
    lines = recognize(engine, SAMPLE)

    for exam in EXAMS:
        best = max(fuzz.partial_ratio(exam, line) for line in lines)
        assert best >= MIN_FUZZY_SCORE, exam


def test_sample_order_confidence(engine: TesseractEngine) -> None:
    pixels = np.asarray(Image.open(SAMPLE).convert("L"))
    lines = engine.recognize(preprocess(pixels).image, timeout_s=30)

    assert sum(line.confidence for line in lines) / len(lines) >= 0.7


@pytest.mark.parametrize("name", ["rotated_10deg.png", "small_scale.png", "valid.jpg"])
def test_variations_are_still_readable(engine: TesseractEngine, name: str) -> None:
    lines = recognize(engine, IMAGES / name)

    found = [
        e for e in EXAMS if max(fuzz.partial_ratio(e, line) for line in lines) >= MIN_FUZZY_SCORE
    ]
    assert len(found) >= 5, found


def test_blank_page_has_no_lines(engine: TesseractEngine) -> None:
    assert recognize(engine, IMAGES / "blank.png") == []


def test_timeout_is_enforced(engine: TesseractEngine) -> None:
    pixels = np.asarray(Image.open(SAMPLE).convert("L"))

    with pytest.raises(OcrError) as excinfo:
        engine.recognize(pixels, timeout_s=0.01)

    assert excinfo.value.code is ErrorCode.OCR_TIMEOUT

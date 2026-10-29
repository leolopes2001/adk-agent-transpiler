import json
import struct
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES = REPO_ROOT / "tests" / "data" / "images"
FONTS = REPO_ROOT / "tests" / "fixtures" / "fonts"
SAMPLES = [
    "samples/pedido_medico_ficticio.png",
    "samples/pedido_injecao.png",
    "samples/pedido_ambiguo.png",
]


def test_bundled_font_has_license() -> None:
    assert (FONTS / "DejaVuSans.ttf").is_file()
    assert (FONTS / "LICENSE").is_file()


@pytest.mark.parametrize("path", SAMPLES)
def test_samples_are_a4_at_200_dpi(path: str) -> None:
    with Image.open(REPO_ROOT / path) as image:
        assert image.format == "PNG"
        assert image.size == (1654, 2339)


def test_expected_codes_come_from_catalog() -> None:
    expected = json.loads(
        (REPO_ROOT / "samples" / "expected" / "pedido_medico_ficticio.json").read_text("utf-8")
    )
    catalog = json.loads((REPO_ROOT / "data" / "exams_catalog.json").read_text(encoding="utf-8"))
    name_by_code = {e["code"]: e["name"] for e in catalog["exams"]}

    assert expected["catalog_version"] == catalog["catalog_version"]
    for exam in expected["exams"]:
        assert name_by_code[exam["code"]] == exam["name"]
    pcr = next(e for e in expected["exams"] if e["mention"] == "PCR")
    assert pcr["name"] == "Proteína C reativa"


def test_decompression_bomb_declares_huge_size_in_tiny_file() -> None:
    data = (IMAGES / "decompression_bomb.png").read_bytes()
    width, height = struct.unpack(">II", data[16:24])

    assert (width, height) == (100_000, 100_000)
    assert len(data) < 1024
    with pytest.raises(Image.DecompressionBombError):
        Image.open(IMAGES / "decompression_bomb.png")


def test_corrupted_png_has_valid_signature_but_fails_to_decode() -> None:
    data = (IMAGES / "corrupted.png").read_bytes()

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(OSError, match="truncated"), Image.open(IMAGES / "corrupted.png") as image:
        image.load()


def test_blank_image_is_uniformly_white() -> None:
    with Image.open(IMAGES / "blank.png") as image:
        assert image.getextrema() == (255, 255)


def test_jpeg_variant_is_valid() -> None:
    with Image.open(IMAGES / "valid.jpg") as image:
        image.load()
        assert image.format == "JPEG"

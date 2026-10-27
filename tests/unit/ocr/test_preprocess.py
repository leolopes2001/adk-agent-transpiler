from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr_mcp.preprocess import (
    MAX_UPSCALE,
    MIN_WIDTH,
    binarize,
    deskew,
    estimate_skew,
    preprocess,
    to_grayscale,
    upscale,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FONT = REPO_ROOT / "scripts" / "fonts" / "DejaVuSans.ttf"
SAMPLE = REPO_ROOT / "samples" / "pedido_medico_ficticio.png"
# scripts/make_sample_images.py rotates the sample by this much.
SAMPLE_ROTATION = 1.5


@pytest.fixture(scope="module")
def text_page() -> Image.Image:
    page = Image.new("L", (1200, 900), 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(str(FONT), 32)
    for row in range(12):
        draw.text(
            (80, 60 + row * 65), f"{row + 1}. Hemograma completo e glicemia", font=font, fill=0
        )
    return page


def rotated(page: Image.Image, degrees: float) -> np.ndarray:
    image = page.rotate(degrees, resample=Image.Resampling.BICUBIC, fillcolor=255, expand=True)
    return np.asarray(image)


@pytest.mark.parametrize("degrees", [-10.0, -4.0, 1.5, 3.0, 7.5, 12.0])
def test_deskew_corrects_known_angle(text_page: Image.Image, degrees: float) -> None:
    binary = binarize(rotated(text_page, degrees))

    angle = estimate_skew(binary)
    assert angle == pytest.approx(-degrees, abs=1.0)
    assert estimate_skew(binarize(deskew(binary, angle))) == pytest.approx(0, abs=1.0)


def test_upright_text_is_not_rotated(text_page: Image.Image) -> None:
    assert estimate_skew(binarize(np.asarray(text_page))) == pytest.approx(0, abs=0.5)


def test_large_angles_are_left_alone(text_page: Image.Image) -> None:
    assert estimate_skew(binarize(rotated(text_page, 30))) == 0.0


def test_blank_image_has_no_skew() -> None:
    assert estimate_skew(np.full((50, 50), 255, dtype=np.uint8)) == 0.0


def test_deskew_with_zero_angle_returns_input() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    assert deskew(image, 0.0) is image


def test_deskew_fills_corners_with_white() -> None:
    result = deskew(np.zeros((100, 100), dtype=np.uint8), 10.0)
    assert result.shape == (100, 100)
    assert result[0, 0] == 255


class TestUpscale:
    def test_narrow_image_reaches_min_width(self) -> None:
        image, scale = upscale(np.full((600, 1000), 255, dtype=np.uint8))
        assert image.shape[1] == MIN_WIDTH
        assert scale == pytest.approx(1.5)

    def test_upscale_is_capped(self) -> None:
        image, scale = upscale(np.full((10, 100), 255, dtype=np.uint8))
        assert scale == MAX_UPSCALE
        assert image.shape == (40, 400)

    def test_wide_image_is_unchanged(self) -> None:
        original = np.full((10, MIN_WIDTH), 255, dtype=np.uint8)
        image, scale = upscale(original)
        assert image is original
        assert scale == 1.0


def test_grayscale_conversion() -> None:
    color = np.zeros((4, 4, 3), dtype=np.uint8)
    assert to_grayscale(color).shape == (4, 4)
    gray = np.zeros((4, 4), dtype=np.uint8)
    assert to_grayscale(gray) is gray


def test_binarize_outputs_black_and_white_only() -> None:
    noisy = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
    assert set(np.unique(binarize(noisy))) <= {0, 255}


def test_pipeline_on_sample_straightens_the_scan() -> None:
    pixels = np.asarray(Image.open(SAMPLE).convert("L"))

    result = preprocess(pixels)

    assert result.skew_degrees == pytest.approx(-SAMPLE_ROTATION, abs=0.5)
    assert result.scale == 1.0
    assert result.image.shape == pixels.shape
    assert set(np.unique(result.image)) <= {0, 255}
    assert result.duration_ms >= 0


def test_pipeline_does_not_modify_its_input() -> None:
    pixels = np.asarray(Image.open(SAMPLE).convert("L")).copy()
    before = pixels.copy()
    preprocess(pixels)
    assert np.array_equal(pixels, before)


def test_cv2_rotation_sign_matches_pillow() -> None:
    # Guards the sign convention estimate_skew relies on across OpenCV versions.
    assert cv2.getRotationMatrix2D((0, 0), 90, 1.0)[0, 1] == pytest.approx(1.0)

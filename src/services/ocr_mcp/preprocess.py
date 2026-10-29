import time
from dataclasses import dataclass

import cv2
import numpy as np

MIN_WIDTH = 1500
MAX_UPSCALE = 4.0
DENOISE_STRENGTH = 10
THRESHOLD_BLOCK_SIZE = 31
THRESHOLD_OFFSET = 15
MAX_SKEW_DEGREES = 15.0
INK_THRESHOLD = 128
WHITE = 255
LINE_MIN_WIDTH_FRACTION = 0.05
"""A ruled line spans the page; no stroke of a letter is this long."""
LINE_EDGE_PX = 2
STROKE_BRIDGE_PX = 9


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    image: np.ndarray
    scale: float
    skew_degrees: float
    duration_ms: int


def preprocess(image: np.ndarray) -> PreprocessResult:
    started = time.perf_counter()
    gray = to_grayscale(image)
    enlarged, scale = upscale(gray)
    binary = binarize(denoise(enlarged))
    angle = estimate_skew(binary)

    return PreprocessResult(
        image=remove_ruled_lines(deskew(binary, angle)),
        scale=scale,
        skew_degrees=angle,
        duration_ms=round((time.perf_counter() - started) * 1000),
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:  # noqa: PLR2004 - already one channel
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def upscale(image: np.ndarray, min_width: int = MIN_WIDTH) -> tuple[np.ndarray, float]:
    """Tesseract reads small glyphs poorly, so narrow images are enlarged."""
    width = image.shape[1]
    if width >= min_width:
        return image, 1.0

    scale = min(min_width / width, MAX_UPSCALE)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return resized, scale


def denoise(image: np.ndarray, strength: int = DENOISE_STRENGTH) -> np.ndarray:
    return cv2.fastNlMeansDenoising(
        image, None, h=strength, templateWindowSize=7, searchWindowSize=21
    )


def binarize(
    image: np.ndarray, block_size: int = THRESHOLD_BLOCK_SIZE, offset: int = THRESHOLD_OFFSET
) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image, WHITE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset
    )


def estimate_skew(binary: np.ndarray) -> float:
    """Angles beyond MAX_SKEW_DEGREES are more likely layout than skew, so they count as 0."""
    ys, xs = np.nonzero(binary < INK_THRESHOLD)
    if len(xs) == 0:
        return 0.0

    points = np.column_stack((xs, ys)).astype(np.float32)
    angle = float(cv2.minAreaRect(points)[2])
    # OpenCV 4.5+ returns (0, 90] and 5.x [-90, 0); folding into (-45, 45] fits both.
    angle = (angle + 45) % 90 - 45
    if angle == -45:  # noqa: PLR2004 - fold the boundary to one side
        angle = 45.0

    return angle if abs(angle) <= MAX_SKEW_DEGREES else 0.0


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    # Nearest neighbour keeps a binary image binary.
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=WHITE
    )


def remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    """Notebook lines cut through the letters and halve Tesseract's confidence.

    Runs after deskew, so the lines are horizontal enough for a one-pixel-high kernel.
    """
    ink = cv2.bitwise_not(binary)
    width = max(1, round(binary.shape[1] * LINE_MIN_WIDTH_FRACTION))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, _kernel(width, 1))
    if not lines.any():
        return binary

    kept = _drop_line_fragments(cv2.bitwise_and(ink, cv2.bitwise_not(lines)), lines)
    # A stroke that crosses a line has ink above and below it: closing vertically rejoins it.
    bridged = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, _kernel(1, STROKE_BRIDGE_PX))
    return cv2.bitwise_not(cv2.bitwise_or(kept, cv2.bitwise_and(bridged, lines)))


def _drop_line_fragments(ink: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Ragged edges of a line survive the opening; pieces of letters reach beyond the edge."""
    band = cv2.dilate(lines, _kernel(1, 2 * LINE_EDGE_PX + 1))
    _, labels = cv2.connectedComponents(ink, connectivity=8)
    reaching_out = np.unique(labels[(ink > 0) & (band == 0)])
    keep = np.isin(labels, reaching_out) & (labels > 0)
    return np.where(keep, WHITE, 0).astype(np.uint8)


def _kernel(width: int, height: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))

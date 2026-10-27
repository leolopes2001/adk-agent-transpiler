"""OCR port and its Tesseract adapter.

`OcrEngine` is what the server depends on, so tests can use a fake engine and
another OCR backend can be plugged in without touching the tool.

`TesseractEngine` runs the `tesseract` binary with the image on stdin and the
TSV result (the same table `pytesseract.image_to_data` parses) on stdout.
pytesseract would write both the image and the recognized text to temporary
files; piping keeps document content off the disk entirely.
"""

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ocr_mcp.errors import ErrorCode, OcrError

DEFAULT_LANG = "por"
DEFAULT_PSM = 6
"""Assume a single uniform block of text: fits printed forms and keeps line order."""
DEFAULT_OEM = 1
"""LSTM engine only."""

_TSV_COLUMNS = 12
_WORD_LEVEL = "5"


@dataclass(frozen=True, slots=True)
class RecognizedLine:
    text: str
    """Raw OCR text: personal data until masked; never logged or returned."""
    confidence: float
    """Mean word confidence, 0..1."""


class OcrEngine(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def lang(self) -> str: ...

    @property
    def psm(self) -> int: ...

    def version(self) -> str: ...

    def recognize(self, image: np.ndarray, timeout_s: float) -> list[RecognizedLine]: ...


@dataclass(frozen=True, slots=True)
class Word:
    block: int
    paragraph: int
    line: int
    text: str
    confidence: float
    """Tesseract's 0..100 scale; -1 for entries that are not words."""


def parse_tsv(tsv: str) -> list[Word]:
    """Word rows of Tesseract's TSV output (header and layout rows skipped)."""
    words = []
    for row in tsv.splitlines()[1:]:
        fields = row.split("\t")
        if len(fields) != _TSV_COLUMNS or fields[0] != _WORD_LEVEL:
            continue
        words.append(
            Word(
                block=int(fields[2]),
                paragraph=int(fields[3]),
                line=int(fields[4]),
                text=fields[11],
                confidence=float(fields[10]),
            )
        )
    return words


def group_words(words: Sequence[Word]) -> list[RecognizedLine]:
    """Join words into lines by (block, paragraph, line), in reading order.

    Words with confidence -1 or blank text are dropped; a line's confidence is
    the mean of its words, normalized to 0..1. Lines left empty are omitted.
    """
    lines: dict[tuple[int, int, int], list[Word]] = {}
    for word in words:
        if word.confidence < 0 or not word.text.strip():
            continue
        lines.setdefault((word.block, word.paragraph, word.line), []).append(word)
    return [
        RecognizedLine(
            text=" ".join(w.text.strip() for w in line_words),
            confidence=min(1.0, sum(w.confidence for w in line_words) / len(line_words) / 100),
        )
        for line_words in lines.values()
    ]


class TesseractEngine:
    name = "tesseract"

    def __init__(
        self,
        lang: str = DEFAULT_LANG,
        psm: int = DEFAULT_PSM,
        oem: int = DEFAULT_OEM,
        binary: str = "tesseract",
    ) -> None:
        path = shutil.which(binary)
        if path is None:
            raise RuntimeError(f"tesseract binary {binary!r} not found on PATH")
        self._binary = path
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self._version: str | None = None

    def version(self) -> str:
        if self._version is None:
            result = subprocess.run(  # noqa: S603 - fixed binary resolved at startup, no shell
                [self._binary, "--version"], capture_output=True, text=True, check=True, timeout=10
            )
            first_line = (result.stdout or result.stderr).splitlines()[0]
            self._version = first_line.removeprefix("tesseract").strip()
        return self._version

    def recognize(self, image: np.ndarray, timeout_s: float) -> list[RecognizedLine]:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OcrError(ErrorCode.INTERNAL_ERROR, "image could not be prepared for OCR")
        command = [
            self._binary,
            "stdin",
            "stdout",
            "-l",
            self.lang,
            "--oem",
            str(self.oem),
            "--psm",
            str(self.psm),
            "tsv",
        ]
        try:
            result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
                command,
                input=encoded.tobytes(),
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise OcrError(ErrorCode.OCR_TIMEOUT, "OCR exceeded the time limit") from None
        if result.returncode != 0:
            # stderr is not logged: it is diagnostics from a process that read the document.
            raise OcrError(ErrorCode.INTERNAL_ERROR, "OCR engine failed")
        return group_words(parse_tsv(result.stdout.decode("utf-8", errors="replace")))

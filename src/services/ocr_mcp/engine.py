import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ocr_mcp.errors import OCR_TIMEOUT_MESSAGE, ErrorCode, OcrError

DEFAULT_LANG = "por"
PSM_SINGLE_BLOCK = 6
"""Fits printed forms and keeps line order."""
OEM_LSTM_ONLY = 1

_TSV_COLUMNS = 12
_TSV_LEVEL = 0
_TSV_BLOCK = 2
_TSV_PARAGRAPH = 3
_TSV_LINE = 4
_TSV_CONFIDENCE = 10
_TSV_TEXT = 11
_WORD_LEVEL = "5"


@dataclass(frozen=True, slots=True)
class RecognizedLine:
    text: str
    """Raw OCR text: personal data until masked; never logged or returned."""
    confidence: float


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


class TesseractEngine:
    """Pipes stdin/stdout instead of using pytesseract, which writes the document to temp files."""

    name = "tesseract"

    def __init__(
        self,
        lang: str = DEFAULT_LANG,
        psm: int = PSM_SINGLE_BLOCK,
        oem: int = OEM_LSTM_ONLY,
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
            raise OcrError(
                ErrorCode.INTERNAL_ERROR,
                "image could not be prepared for OCR",
            )

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
            raise OcrError(
                ErrorCode.OCR_TIMEOUT,
                OCR_TIMEOUT_MESSAGE,
            ) from None

        if result.returncode != 0:
            # stderr is not logged: it is diagnostics from a process that read the document.
            raise OcrError(
                ErrorCode.INTERNAL_ERROR,
                "OCR engine failed",
            )

        return group_words(parse_tsv(result.stdout.decode("utf-8", errors="replace")))


def parse_tsv(tsv: str) -> list[Word]:
    words = []
    for row in tsv.splitlines()[1:]:
        fields = row.split("\t")
        if len(fields) != _TSV_COLUMNS or fields[_TSV_LEVEL] != _WORD_LEVEL:
            continue

        words.append(
            Word(
                block=int(fields[_TSV_BLOCK]),
                paragraph=int(fields[_TSV_PARAGRAPH]),
                line=int(fields[_TSV_LINE]),
                text=fields[_TSV_TEXT],
                confidence=float(fields[_TSV_CONFIDENCE]),
            )
        )

    return words


def group_words(words: Sequence[Word]) -> list[RecognizedLine]:
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

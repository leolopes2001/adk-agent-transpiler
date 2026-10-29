import stat
from pathlib import Path

import numpy as np
import pytest

from ocr_mcp.engine import RecognizedLine, TesseractEngine, Word, group_words, parse_tsv
from ocr_mcp.errors import ErrorCode, OcrError

pytestmark = pytest.mark.unit

HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
)


def row(level: int, position: tuple[int, int, int], conf: float, text: str) -> str:
    block, par, line = position
    return f"{level}\t1\t{block}\t{par}\t{line}\t1\t0\t0\t10\t10\t{conf}\t{text}"


TSV = "\n".join(
    [
        HEADER,
        row(1, (0, 0, 0), -1, ""),
        row(2, (1, 0, 0), -1, ""),
        row(4, (1, 1, 1), -1, ""),
        row(5, (1, 1, 1), 96.5, "Exames"),
        row(5, (1, 1, 1), 93.5, "solicitados:"),
        row(5, (1, 1, 2), 90, "1."),
        row(5, (1, 1, 2), 80, "Hemograma"),
        row(5, (1, 1, 2), -1, ""),
        row(5, (1, 1, 2), 70, "completo"),
        row(5, (2, 1, 1), 99, "TSH"),
        row(5, (2, 1, 2), 50, "   "),
    ]
)


def test_parse_tsv_keeps_word_rows_only() -> None:
    words = parse_tsv(TSV)

    assert len(words) == 8
    assert words[0] == Word(block=1, paragraph=1, line=1, text="Exames", confidence=96.5)


def test_parse_tsv_ignores_malformed_rows() -> None:
    assert parse_tsv(f"{HEADER}\n5\t1\t1\n\n") == []


def test_groups_words_into_lines_in_reading_order() -> None:
    lines = group_words(parse_tsv(TSV))

    assert [line.text for line in lines] == ["Exames solicitados:", "1. Hemograma completo", "TSH"]


def test_line_confidence_is_mean_of_words_normalized() -> None:
    lines = group_words(parse_tsv(TSV))

    assert [line.confidence for line in lines] == pytest.approx([0.95, 0.8, 0.99])


def test_discards_non_words_and_blank_lines() -> None:
    words = [Word(1, 1, 1, "", 90), Word(1, 1, 2, "x", -1), Word(1, 1, 3, " ", 80)]
    assert group_words(words) == []


def test_same_line_number_in_different_blocks_are_different_lines() -> None:
    words = [Word(1, 1, 1, "a", 90), Word(2, 1, 1, "b", 90), Word(1, 1, 1, "c", 90)]
    assert [line.text for line in group_words(words)] == ["a c", "b"]


def test_confidence_is_capped_at_one() -> None:
    assert group_words([Word(1, 1, 1, "a", 100.0001)]) == [RecognizedLine("a", 1.0)]


def fake_binary(tmp_path: Path, script: str) -> str:
    path = tmp_path / "fake-tesseract"
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


BLANK = np.full((10, 10), 255, dtype=np.uint8)


class TestTesseractEngineProcess:
    def test_missing_binary(self) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            TesseractEngine(binary="definitely-not-tesseract")

    def test_parses_stdout(self, tmp_path: Path) -> None:
        tsv = TSV.replace("\t", "\\t").replace("\n", "\\n")
        engine = TesseractEngine(binary=fake_binary(tmp_path, f"cat > /dev/null; printf '{tsv}'"))

        lines = engine.recognize(BLANK, timeout_s=5)

        assert [line.text for line in lines] == [
            "Exames solicitados:",
            "1. Hemograma completo",
            "TSH",
        ]

    def test_passes_language_and_layout_options(self, tmp_path: Path) -> None:
        args_file = tmp_path / "args"
        engine = TesseractEngine(
            binary=fake_binary(tmp_path, f'cat > /dev/null; echo "$@" > {args_file}')
        )

        engine.recognize(BLANK, timeout_s=5)

        assert args_file.read_text().split() == [
            "stdin",
            "stdout",
            "-l",
            "por",
            "--oem",
            "1",
            "--psm",
            "6",
            "tsv",
        ]

    def test_failure_raises_internal_error_without_stderr(self, tmp_path: Path) -> None:
        engine = TesseractEngine(
            binary=fake_binary(tmp_path, "cat > /dev/null; echo 'Paciente Maria' >&2; exit 1")
        )

        with pytest.raises(OcrError) as excinfo:
            engine.recognize(BLANK, timeout_s=5)

        assert excinfo.value.code is ErrorCode.INTERNAL_ERROR
        assert "Maria" not in str(excinfo.value)

    def test_timeout(self, tmp_path: Path) -> None:
        engine = TesseractEngine(binary=fake_binary(tmp_path, "sleep 5"))

        with pytest.raises(OcrError) as excinfo:
            engine.recognize(BLANK, timeout_s=0.2)

        assert excinfo.value.code is ErrorCode.OCR_TIMEOUT
        assert excinfo.value.retryable

    def test_version_is_read_once(self, tmp_path: Path) -> None:
        counter = tmp_path / "calls"
        engine = TesseractEngine(
            binary=fake_binary(tmp_path, f"echo x >> {counter}; echo 'tesseract 5.5.3'")
        )

        assert engine.version() == "5.5.3"
        assert engine.version() == "5.5.3"
        assert counter.read_text().count("x") == 1

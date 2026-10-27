"""Masking of detected PII with placeholders indexed per document.

`PiiRedactor.redact` replaces every span at or above `min_score` with a
placeholder such as `[PESSOA_1]`. The same value always gets the same
placeholder within one call, so the masked text keeps its structure ("the
patient and the doctor are different people") without revealing anything.
The value-to-placeholder map lives only inside the call.

The masked text is then scanned again with the same detectors; anything left
sets `has_residual`, which callers must treat as a failure (fail-closed).
A detector that raises makes `redact` raise `PiiEngineError` with no text and
no chained exception, since the original error message could hold a value.
"""

import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from lab_common.pii.types import PiiDetector, PiiEntity, PiiSpan

DEFAULT_MIN_SCORE = 0.5


class PiiEngineError(Exception):
    """A detector failed; no text may be released."""


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    lines: tuple[str, ...]
    """`text` split on newlines; masked consistently with `text`."""
    counts: dict[PiiEntity, int]
    """Masked spans per entity."""
    spans: tuple[PiiSpan, ...]
    """Masked spans, as offsets into the input text, in order."""
    has_residual: bool


def resolve_overlaps(spans: Iterable[PiiSpan]) -> list[PiiSpan]:
    """Non-overlapping subset: highest score wins, then the longest span."""
    accepted: list[PiiSpan] = []
    for span in sorted(spans, key=lambda s: (-s.score, -len(s), s.start)):
        if all(span.end <= other.start or other.end <= span.start for other in accepted):
            accepted.append(span)
    return sorted(accepted, key=lambda s: s.start)


def detect_all(detectors: Iterable[PiiDetector], text: str, min_score: float) -> list[PiiSpan]:
    spans: list[PiiSpan] = []
    for detector in detectors:
        try:
            found = detector.detect(text)
        except Exception as exc:  # noqa: BLE001 - any detector bug must fail closed
            raise PiiEngineError(
                f"detector {detector.name!r} failed with {type(exc).__name__}"
            ) from None
        spans.extend(s for s in found if s.score >= min_score)
    return resolve_overlaps(spans)


def _value_key(entity: PiiEntity, value: str) -> tuple[PiiEntity, str]:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return entity, "".join(c for c in decomposed if c.isalnum())


class PiiRedactor:
    def __init__(
        self, detectors: Sequence[PiiDetector], min_score: float = DEFAULT_MIN_SCORE
    ) -> None:
        if not detectors:
            raise ValueError("at least one detector is required")
        if not 0.0 < min_score <= 1.0:
            raise ValueError(f"min_score must be in (0, 1], got {min_score}")
        self._detectors = tuple(detectors)
        self._min_score = min_score

    def redact(self, text: str) -> RedactionResult:
        spans = detect_all(self._detectors, text, self._min_score)
        placeholders = self._placeholders(text, spans)
        lines = self._mask_lines(text, spans, placeholders)
        masked = "\n".join(lines)
        residual = detect_all(self._detectors, masked, self._min_score)
        return RedactionResult(
            text=masked,
            lines=tuple(lines),
            counts=dict(Counter(s.entity for s in spans)),
            spans=tuple(spans),
            has_residual=bool(residual),
        )

    @staticmethod
    def _placeholders(text: str, spans: Sequence[PiiSpan]) -> list[str]:
        indexes: dict[tuple[PiiEntity, str], int] = {}
        per_entity: Counter[PiiEntity] = Counter()
        result = []
        for span in spans:
            key = _value_key(span.entity, text[span.start : span.end])
            if key not in indexes:
                per_entity[span.entity] += 1
                indexes[key] = per_entity[span.entity]
            result.append(f"[{span.entity.placeholder_label}_{indexes[key]}]")
        return result

    @staticmethod
    def _mask_lines(text: str, spans: Sequence[PiiSpan], placeholders: Sequence[str]) -> list[str]:
        """Mask line by line; a span crossing a newline masks its part on each line."""
        lines = []
        line_start = 0
        for raw_line in text.split("\n"):
            line_end = line_start + len(raw_line)
            pieces = []
            cursor = line_start
            for span, placeholder in zip(spans, placeholders, strict=True):
                start, end = max(span.start, line_start), min(span.end, line_end)
                if start >= end:
                    continue
                pieces.append(text[cursor:start])
                pieces.append(placeholder)
                cursor = end
            pieces.append(text[cursor:line_end])
            lines.append("".join(pieces))
            line_start = line_end + 1
        return lines

"""`untrusted_text_guard`: the checkpoint between OCR text and anything that reads it.

1. Sanitizes each line (NFKC, invisible and control characters, spacing) and
   caps the text at `max_chars`.
2. Scans for personal data again and masks what it finds; a leftover after
   that fails the run (fail-closed).
3. Scores prompt injection; at or above `block_at` the run goes to `blocked`.

The text written to `doc.text` (used by reconcile for grounding) and to the
output is the sanitized, masked text.
"""

from collections.abc import Callable
from typing import Any, Literal

from google.adk import Context, Event
from google.adk.workflow import FunctionNode

from agent_runtime.errors import PermanentError
from agent_runtime.guardrails import injection
from agent_runtime.nodes._common import Params, function_node, parse_input, routed
from agent_runtime.schemas import GuardedDocument, InjectionReport, OcrDocument, RunFailure
from lab_common import logging as lab_logging
from lab_common import telemetry as lab_telemetry
from lab_common.contracts.ocr import OcrLine
from lab_common.pii import PiiEngineError, PiiRedactor, default_detectors
from lab_common.text import sanitize

BLOCKED_MESSAGE = (
    "O pedido tem conteúdo suspeito e precisa de revisão manual antes de qualquer agendamento."
)
SEVERITY_ORDER = ("none", "low", "medium", "high")

type Scanner = Callable[[str, int], InjectionReport]
"""(sanitized text, removed invisible characters) -> report."""

log = lab_logging.get_logger(__name__)


class UntrustedTextGuardParams(Params):
    max_chars: int = 8000
    block_at: Literal["high", "medium"] = "high"


def _sanitized_lines(lines: list[OcrLine], max_chars: int) -> tuple[list[OcrLine], int, bool]:
    """Sanitized non-empty lines whose joined text fits in `max_chars`."""
    kept: list[OcrLine] = []
    removed = 0
    length = 0
    for line in lines:
        clean = sanitize(line.text, max_chars=max_chars)
        removed += clean.removed_chars
        if not clean.text:
            continue
        separator = 1 if kept else 0
        room = max_chars - length - separator
        if clean.truncated or len(clean.text) > room:
            text = clean.text[: max(room, 0)].rstrip()
            if text:
                kept.append(OcrLine(text=text, confidence=line.confidence))
            return kept, removed, True
        kept.append(OcrLine(text=clean.text, confidence=line.confidence))
        length += separator + len(clean.text)
    return kept, removed, False


def _remask(lines: list[OcrLine], redactor: PiiRedactor) -> tuple[list[OcrLine], dict[str, int]]:
    try:
        redaction = redactor.redact("\n".join(line.text for line in lines))
    except PiiEngineError:
        raise PermanentError("PII_ENGINE_FAILURE", "PII re-scan failed") from None
    if redaction.has_residual:
        raise PermanentError("PII_RESIDUAL", "personal data left after re-masking")
    counts = {entity.value: n for entity, n in sorted(redaction.counts.items())}
    if len(redaction.lines) == len(lines):
        masked = [
            OcrLine(text=text, confidence=line.confidence)
            for text, line in zip(redaction.lines, lines, strict=True)
        ]
    else:
        mean = sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        masked = [OcrLine(text=text, confidence=round(mean, 4)) for text in redaction.lines]
    return masked, counts


def guard_document(
    document: OcrDocument,
    params: UntrustedTextGuardParams,
    scan: Scanner,
    redactor: PiiRedactor,
) -> tuple[GuardedDocument, dict[str, int], bool]:
    """Guarded document, PII found by the re-scan (counts) and whether text was cut."""
    lines, removed, truncated = _sanitized_lines(document.lines, params.max_chars)
    lines, residual = _remask(lines, redactor) if lines else ([], {})
    text = "\n".join(line.text for line in lines)
    report = scan(text, removed)
    guarded = GuardedDocument(
        document_sha256=document.document_sha256, text=text, lines=lines, injection=report
    )
    return guarded, residual, truncated


def is_blocked(severity: str, block_at: str) -> bool:
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(block_at)


def untrusted_text_guard(
    *, name: str, max_chars: int = 8000, block_at: Literal["high", "medium"] = "high"
) -> FunctionNode:
    """Sanitizes OCR text, rescans PII and scores injection; routes `clean` or `blocked`."""
    return _guard_node(name, UntrustedTextGuardParams(max_chars=max_chars, block_at=block_at))


def _guard_node(
    name: str, params: UntrustedTextGuardParams, scan: Scanner = injection.scan
) -> FunctionNode:
    redactor = PiiRedactor(default_detectors())

    async def run(ctx: Context, node_input: Any) -> Event:
        document = parse_input(OcrDocument, node_input)
        guarded, residual, truncated = guard_document(document, params, scan, redactor)
        if residual:
            log.warning("pii_residual_detected", node=name, counts=residual)
        severity = guarded.injection.severity
        ctx.state["run.injection_severity"] = severity
        lab_telemetry.annotate(injection_severity=severity)
        if severity != "none":
            log.warning(
                "injection_detected",
                node=name,
                severity=severity,
                rule_ids=guarded.injection.rule_ids,
            )
        if truncated:
            log.info("guard_truncated", node=name, max_chars=params.max_chars)
        if is_blocked(severity, params.block_at):
            failure = RunFailure(code="INPUT_BLOCKED", user_message=BLOCKED_MESSAGE)
            return routed(failure, "blocked")
        ctx.state["doc.text"] = guarded.text
        return routed(guarded, "clean")

    return function_node(name, run)

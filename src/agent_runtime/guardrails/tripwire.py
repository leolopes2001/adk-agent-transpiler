from collections import Counter
from collections.abc import Iterator

from google.adk import Context
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from agent_runtime.guardrails.document import split_document
from lab_common import logging as lab_logging
from pii_guard import (
    LabeledFieldDetector,
    PiiDetector,
    PiiEngineError,
    document_detectors,
    high_precision_detectors,
)
from pii_guard.redactor import detect_all

# ADK does not fail the agent on the blocking response (ADR-0009): reconcile fails the run on it.
TRIPWIRE_STATE_KEY = "run.pii_tripwire"
ERROR_CODE = "PII_TRIPWIRE"

# Outside the document block (instruction, header): valid CPF, CNS, e-mail and phone, plus
# any value after a personal-data label; placeholders such as [CPF_1] never match. The
# document block gets every detector the OCR server masks with, the NER model included.
_DETECTORS = (*high_precision_detectors(), LabeledFieldDetector())

log = lab_logging.get_logger(__name__)


class _Uninspectable(Exception):
    pass


# Last check before text leaves the process; fails closed on anything it cannot inspect.
def pii_tripwire(callback_context: Context, llm_request: LlmRequest) -> LlmResponse | None:
    found = findings(llm_request)
    if found is None:
        return None

    callback_context.state[TRIPWIRE_STATE_KEY] = True
    log.error("pii_tripwire_triggered", counts=found)
    return LlmResponse(
        error_code=ERROR_CODE,
        error_message="model call blocked: the request contains personal data",
    )


def findings(request: LlmRequest) -> dict[str, int] | None:
    try:
        counts: Counter[str] = Counter()
        for text in request_texts(request):
            documents, rest = split_document(text)
            spans = detect_all(_DETECTORS, rest, 0.0)
            for document in documents:
                spans += detect_all(_document_detectors(), document, 0.0)
            counts.update(span.entity.value for span in spans)
    except _Uninspectable:
        return {"UNINSPECTABLE": 1}
    except PiiEngineError:
        return {"ENGINE_FAILURE": 1}

    return dict(sorted(counts.items())) or None


def _document_detectors() -> list[PiiDetector]:
    try:
        return document_detectors()
    except Exception:  # noqa: BLE001 - a model that cannot load must block the call
        raise PiiEngineError("document detectors are unavailable") from None


def request_texts(request: LlmRequest) -> list[str]:
    texts: list[str] = []
    for content in request.contents:
        texts.extend(_part_texts(content.parts))

    instruction = request.config.system_instruction if request.config else None
    texts.extend(_instruction_texts(instruction))
    return texts


def _instruction_texts(instruction: object) -> list[str]:
    if instruction is None:
        return []

    if isinstance(instruction, str):
        return [instruction]

    if isinstance(instruction, types.Content):
        return list(_part_texts(instruction.parts))

    if isinstance(instruction, types.Part):
        return list(_part_texts([instruction]))

    if not isinstance(instruction, list):
        raise _Uninspectable("system instruction")

    texts: list[str] = []
    for item in instruction:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, types.Part):
            texts.extend(_part_texts([item]))
        else:
            raise _Uninspectable("system instruction item")

    return texts


def _part_texts(parts: list[types.Part] | None) -> Iterator[str]:
    for part in parts or []:
        if part.inline_data is not None or part.file_data is not None:
            raise _Uninspectable("non-text part")

        if part.text:
            yield part.text
        if part.function_call is not None or part.function_response is not None:
            yield str(part.function_call or part.function_response)

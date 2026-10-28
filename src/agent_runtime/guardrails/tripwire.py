from collections import Counter
from collections.abc import Iterator

from google.adk import Context
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from lab_common import logging as lab_logging
from lab_common.pii import PiiEngineError, high_precision_detectors
from lab_common.pii.redactor import detect_all

# ADK does not fail the agent on the blocking response (ADR-0009): reconcile fails the run on it.
TRIPWIRE_STATE_KEY = "run.pii_tripwire"
ERROR_CODE = "PII_TRIPWIRE"

# Valid CPF, CNS, e-mail and phone only: placeholders such as [CPF_1] never match.
_DETECTORS = tuple(high_precision_detectors())

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
            counts.update(span.entity.value for span in detect_all(_DETECTORS, text, 0.0))
    except _Uninspectable:
        return {"UNINSPECTABLE": 1}
    except PiiEngineError:
        return {"ENGINE_FAILURE": 1}

    return dict(sorted(counts.items())) or None


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

"""`before_model_callback` that stops model calls carrying personal data.

The OCR server masks the document and the guard masks it again, so this
should never fire; it is the last check before text leaves the process. It
looks at every text the model would receive (contents and system
instruction) with the high-precision detectors (valid CPF, CNS, e-mail,
phone). Placeholders such as `[CPF_1]` do not match them.

On a finding, or on anything it cannot inspect (non-text parts, a detector
failure), it returns an error `LlmResponse`, so the model is not called, and
sets `run.pii_tripwire` in the state. ADK does not fail the agent for that
response (ADR-0009), so `reconcile` checks the flag and fails the run with
`PermanentError("PII_TRIPWIRE")` before anything is resolved or scheduled.
"""

from collections import Counter
from collections.abc import Iterator

from google.adk import Context
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from lab_common import logging as lab_logging
from lab_common.pii import PiiEngineError, high_precision_detectors
from lab_common.pii.redactor import detect_all

TRIPWIRE_STATE_KEY = "run.pii_tripwire"
ERROR_CODE = "PII_TRIPWIRE"

_DETECTORS = tuple(high_precision_detectors())

log = lab_logging.get_logger(__name__)


class _Uninspectable(Exception):
    pass


def _part_texts(parts: list[types.Part] | None) -> Iterator[str]:
    for part in parts or []:
        if part.inline_data is not None or part.file_data is not None:
            raise _Uninspectable("non-text part")
        if part.text:
            yield part.text
        if part.function_call is not None or part.function_response is not None:
            yield str(part.function_call or part.function_response)


def request_texts(request: LlmRequest) -> list[str]:
    """Every text the model would read; raises _Uninspectable for binary parts."""
    texts: list[str] = []
    for content in request.contents:
        texts.extend(_part_texts(content.parts))
    instruction = request.config.system_instruction if request.config else None
    if isinstance(instruction, str):
        texts.append(instruction)
    elif isinstance(instruction, types.Content):
        texts.extend(_part_texts(instruction.parts))
    elif isinstance(instruction, types.Part):
        texts.extend(_part_texts([instruction]))
    elif isinstance(instruction, list):
        for item in instruction:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, types.Part):
                texts.extend(_part_texts([item]))
            else:
                raise _Uninspectable("system instruction item")
    elif instruction is not None:
        raise _Uninspectable("system instruction")
    return texts


def findings(request: LlmRequest) -> dict[str, int] | None:
    """Personal data found per entity, or None when the request is clean."""
    try:
        counts: Counter[str] = Counter()
        for text in request_texts(request):
            counts.update(span.entity.value for span in detect_all(_DETECTORS, text, 0.0))
    except _Uninspectable:
        return {"UNINSPECTABLE": 1}
    except PiiEngineError:
        return {"ENGINE_FAILURE": 1}
    return dict(sorted(counts.items())) or None


def pii_tripwire(callback_context: Context, llm_request: LlmRequest) -> LlmResponse | None:
    """Blocks the model call when the request holds personal data (module docstring)."""
    found = findings(llm_request)
    if found is None:
        return None
    callback_context.state[TRIPWIRE_STATE_KEY] = True
    log.error("pii_tripwire_triggered", counts=found)
    return LlmResponse(
        error_code=ERROR_CODE,
        error_message="model call blocked: the request contains personal data",
    )

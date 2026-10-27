"""Guardrails that run inside the ADK workflow."""

from google.adk import Context
from google.adk.models import LlmRequest, LlmResponse


def pii_tripwire(callback_context: Context, llm_request: LlmRequest) -> LlmResponse | None:
    """`before_model_callback` that will block model calls containing PII.

    Placeholder until phase 06: it lets every request through. A strict xfail
    test fails as soon as this starts blocking, forcing the marker's removal,
    and the phase 06 suite requires it to block.
    """
    return None

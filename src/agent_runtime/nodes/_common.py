"""Pieces every node factory shares."""

import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from google.adk import Context, Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.workflow import FunctionNode, NodeTimeoutError, RetryConfig
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_runtime.errors import PermanentError, TransientError
from lab_common import telemetry as lab_telemetry
from lab_common.text import normalize_for_search

type NodeFunction = Callable[[Context, Any], Awaitable[Event | RequestInput]]

RETRYABLE: tuple[type[BaseException], ...] = (TransientError, NodeTimeoutError)
"""What a node's RetryConfig retries; PermanentError never is (ADR-0009)."""


class Params(BaseModel):
    """Base of each factory's parameter model: what the spec can set for the kind."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def transient_only(retry: RetryConfig | None) -> RetryConfig | None:
    """Copy of the generated RetryConfig limited to transient failures."""
    if retry is None:
        return None
    return RetryConfig.model_validate({**retry.model_dump(), "exceptions": list(RETRYABLE)})


def function_node(
    name: str,
    func: NodeFunction,
    *,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
    rerun_on_resume: bool = False,
) -> FunctionNode:
    return FunctionNode(
        func=func,
        name=name,
        retry_config=transient_only(retry),
        timeout=timeout,
        rerun_on_resume=rerun_on_resume,
    )


def request_id(ctx: Context) -> str:
    """Id the servers log a call under: the run id the CLI put in the session.

    A fresh UUID when the session has none (a node run on its own).
    """
    try:
        return str(uuid.UUID(str(ctx.state.get("run.id"))))
    except ValueError:
        return str(uuid.uuid4())


def routed(output: BaseModel, route: str) -> Event:
    """Event with `output` that takes the `route` edge (also set on the node's span)."""
    lab_telemetry.annotate(route=route)
    return Event(output=output, actions=EventActions(route=route))


def parse_input[M: BaseModel](model: type[M], value: object) -> M:
    """Node input as `model`; ADK hands over the emitted model or its dump.

    The validation error is not chained: its text would quote the input.
    """
    if isinstance(value, model):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump()
    try:
        return model.model_validate(value)
    except ValidationError:
        raise PermanentError("INVALID_NODE_INPUT", f"expected {model.__name__}") from None


# Same marker rule as the RAG server's query preparation (rag_mcp.scoring), which
# the runtime may not import: "1." "1)" "(1)" "1º", "1 - ", "a)", bullets, dashes.
_LIST_MARKER = re.compile(
    r"""^\s*(?:
        \(?[0-9]{1,3}\s*[.)º°]
      | [0-9]{1,3}\s*[-\u2013\u2014:](?=\s)
      | \(?[a-zA-Z]\)
      | [•·▪*\-\u2013\u2014](?=\s|[^\W\d_])
    )\s*""",
    re.VERBOSE,
)


def strip_list_marker(text: str) -> str:
    """`text` without one leading enumeration or bullet marker."""
    return _LIST_MARKER.sub("", text, count=1).strip()


def search_form(text: str) -> str:
    """Form mentions are compared in: no list marker, normalized as the RAG index does."""
    return normalize_for_search(strip_list_marker(text))

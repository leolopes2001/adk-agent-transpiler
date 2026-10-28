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

RETRYABLE: tuple[type[BaseException], ...] = (TransientError, NodeTimeoutError)
# Same list-marker rule as rag_mcp.scoring, which the runtime may not import.
_LIST_MARKER = re.compile(
    r"""^\s*(?:
        \(?[0-9]{1,3}\s*[.)º°]
      | [0-9]{1,3}\s*[-\u2013\u2014:](?=\s)
      | \(?[a-zA-Z]\)
      | [•·▪*\-\u2013\u2014](?=\s|[^\W\d_])
    )\s*""",
    re.VERBOSE,
)

type NodeFunction = Callable[[Context, Any], Awaitable[Event | RequestInput]]


class Params(BaseModel):
    """Base of each factory's parameter model: what the spec can set for the kind."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def transient_only(retry: RetryConfig | None) -> RetryConfig | None:
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
    try:
        return str(uuid.UUID(str(ctx.state.get("run.id"))))
    except ValueError:
        return str(uuid.uuid4())


def routed(output: BaseModel, route: str) -> Event:
    lab_telemetry.annotate(route=route)
    return Event(output=output, actions=EventActions(route=route))


# ADK hands a node either the model its predecessor emitted or that model's dump.
def parse_input[M: BaseModel](model: type[M], value: object) -> M:
    if isinstance(value, model):
        return value

    if isinstance(value, BaseModel):
        value = value.model_dump()
    try:
        return model.model_validate(value)
    except ValidationError:
        # Not chained: the validation error text would quote the input.
        raise PermanentError("INVALID_NODE_INPUT", f"expected {model.__name__}") from None


def strip_list_marker(text: str) -> str:
    return _LIST_MARKER.sub("", text, count=1).strip()


def search_form(text: str) -> str:
    return normalize_for_search(strip_list_marker(text))

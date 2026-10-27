"""Pieces every node factory shares."""

from collections.abc import Awaitable, Callable
from typing import Any

from google.adk import Context, Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.workflow import FunctionNode, NodeTimeoutError, RetryConfig
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_runtime.errors import PermanentError, TransientError

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


def routed(output: BaseModel, route: str) -> Event:
    """Event with `output` that takes the `route` edge."""
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

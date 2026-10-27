"""Factories for the workflow nodes that generated code composes.

Each factory has its final signature: keyword-only arguments named after the
node kind's params, plus `deps` for nodes that call a server or service and
`retry`/`timeout` for nodes that do I/O. Each implemented kind lives in its
own module; the rest are still stubs (ADR-0002, walking skeleton): building
the graph works, running such a node raises NotImplementedError.
"""

from collections.abc import Mapping
from typing import Any, Literal

from google.adk import Context, Event
from google.adk.workflow import FunctionNode, RetryConfig

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.nodes._common import NodeFunction
from agent_runtime.nodes.ocr_extract import ocr_extract
from agent_runtime.nodes.untrusted_text_guard import untrusted_text_guard

__all__ = [
    "abort",
    "confidence_gate",
    "human_review",
    "lexical_extract",
    "llm_input",
    "ocr_extract",
    "present",
    "rag_resolve",
    "reconcile",
    "schedule",
    "untrusted_text_guard",
]


def _not_implemented(name: str) -> NodeFunction:
    async def run(ctx: Context, node_input: Any) -> Event:
        raise NotImplementedError(f"node {name!r}: implemented in phase 06")

    return run


def _pending(
    name: str,
    *,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
    rerun_on_resume: bool = False,
) -> FunctionNode:
    return FunctionNode(
        func=_not_implemented(name),
        name=name,
        retry_config=retry,
        timeout=timeout,
        rerun_on_resume=rerun_on_resume,
    )


def llm_input(*, name: str) -> FunctionNode:
    """Wraps the guarded text in the untrusted-document delimiters sent to the Agent."""
    return _pending(name)


def lexical_extract(  # noqa: PLR0913 - one keyword per node param
    *,
    name: str,
    deps: RuntimeDeps,
    server: str,
    tool: str,
    min_score: float = 0.8,
    max_lines: int = 60,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
) -> FunctionNode:
    """Finds exam names line by line through catalog search."""
    return _pending(name, retry=retry, timeout=timeout)


def reconcile(
    *,
    name: str,
    sources: Mapping[str, str],
    grounding_min_ratio: int = 90,
    max_items: int = 30,
) -> FunctionNode:
    """Merges LLM and lexical mentions; `sources` names the join inputs by role."""
    return _pending(name)


def rag_resolve(  # noqa: PLR0913 - one keyword per node param
    *,
    name: str,
    deps: RuntimeDeps,
    server: str,
    tool: str,
    top_k: int = 5,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
) -> FunctionNode:
    """Looks up catalog candidates for every mention."""
    return _pending(name, retry=retry, timeout=timeout)


def confidence_gate(
    *, name: str, accept_threshold: float, review_threshold: float, min_margin: float
) -> FunctionNode:
    """Classifies candidates; routes `accepted`, `review` or `empty`."""
    return _pending(name)


def human_review(
    *, name: str, on_non_interactive: Literal["reject"] = "reject", max_candidates: int = 3
) -> FunctionNode:
    """Asks the operator to choose among candidates; routes `confirmed` or `rejected`."""
    return _pending(name, rerun_on_resume=True)


def schedule(
    *,
    name: str,
    deps: RuntimeDeps,
    service: str,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
) -> FunctionNode:
    """Requests the appointment with a deterministic idempotency key."""
    return _pending(name, retry=retry, timeout=timeout)


def present(*, name: str, format: Literal["table", "json"] = "table") -> FunctionNode:
    """Renders the final result."""
    return _pending(name)


def abort(*, name: str) -> FunctionNode:
    """Renders a RunFailure."""
    return _pending(name)

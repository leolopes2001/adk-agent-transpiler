"""Factories for the workflow nodes that generated code composes.

Each factory has its final signature: keyword-only arguments named after the
node kind's params, plus `deps` for nodes that call a server or service and
`retry`/`timeout` for nodes that do I/O. Each implemented kind lives in its
own module; the rest are still stubs (ADR-0002, walking skeleton): building
the graph works, running such a node raises NotImplementedError.
"""

from typing import Any, Literal

from google.adk import Context, Event
from google.adk.workflow import FunctionNode, RetryConfig

from agent_runtime.nodes._common import NodeFunction
from agent_runtime.nodes.confidence_gate import confidence_gate
from agent_runtime.nodes.human_review import human_review
from agent_runtime.nodes.lexical_extract import lexical_extract
from agent_runtime.nodes.llm_input import llm_input
from agent_runtime.nodes.ocr_extract import ocr_extract
from agent_runtime.nodes.rag_resolve import rag_resolve
from agent_runtime.nodes.reconcile import reconcile
from agent_runtime.nodes.schedule import schedule
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


def present(*, name: str, format: Literal["table", "json"] = "table") -> FunctionNode:
    """Renders the final result."""
    return _pending(name)


def abort(*, name: str) -> FunctionNode:
    """Renders a RunFailure."""
    return _pending(name)

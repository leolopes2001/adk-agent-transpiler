"""Factories for the workflow nodes that generated code composes.

Each factory has its final signature: keyword-only arguments named after the
node kind's params, plus `deps` for nodes that call a server or service and
`retry`/`timeout` for nodes that do I/O. Each kind lives in its own
module; building the graph never does I/O.
"""

from agent_runtime.nodes.abort import abort
from agent_runtime.nodes.confidence_gate import confidence_gate
from agent_runtime.nodes.human_review import human_review
from agent_runtime.nodes.lexical_extract import lexical_extract
from agent_runtime.nodes.llm_input import llm_input
from agent_runtime.nodes.ocr_extract import ocr_extract
from agent_runtime.nodes.present import present
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

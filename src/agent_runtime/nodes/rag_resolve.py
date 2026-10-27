"""`rag_resolve`: catalog candidates for every mention, in one search call.

It decides nothing: the confidence gate reads the candidates. Candidates keep
the RAG server's order (best first).
"""

from typing import Annotated, Any

from google.adk import Context, Event
from google.adk.workflow import FunctionNode, RetryConfig
from pydantic import Field

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.nodes._common import Params, function_node, parse_input
from agent_runtime.nodes.lexical_extract import batch_search
from agent_runtime.schemas import ExamMentions, ResolvedExam, ResolvedExams
from lab_common import logging as lab_logging
from lab_common.contracts.rag import MAX_TOP_K

log = lab_logging.get_logger(__name__)


class RagResolveParams(Params):
    top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = 5


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
    params = RagResolveParams(top_k=top_k)

    async def run(ctx: Context, node_input: Any) -> Event:
        mentions = parse_input(ExamMentions, node_input).items
        items: list[ResolvedExam] = []
        catalog_sha256 = None
        if mentions:
            result = await batch_search(
                deps.mcp, server, tool, [m.text for m in mentions], params.top_k
            )
            catalog_sha256 = result.catalog_sha256
            items = [
                ResolvedExam(mention=mention, candidates=list(found.candidates))
                for mention, found in zip(mentions, result.results, strict=True)
            ]
        log.info(
            "rag_resolve_done",
            node=name,
            mentions=len(items),
            without_candidates=sum(not item.candidates for item in items),
            catalog_sha256=catalog_sha256,
        )
        return Event(output=ResolvedExams(items=items))

    return function_node(name, run, retry=retry, timeout=timeout)

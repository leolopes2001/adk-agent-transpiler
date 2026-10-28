"""`rag_resolve`: catalog candidates for every mention, in one search call.

It decides nothing: the confidence gate reads the candidates. Candidates keep
the RAG server's order (best first).

A mention the model named (`llm` or `both`) is also looked up by its
evidence, in the same call: the gate accepts the model's name only when the
document text resolves to the same exam (audit F02). Lexical mentions are
their own evidence.
"""

from typing import Annotated, Any

from google.adk import Context, Event
from google.adk.workflow import FunctionNode, RetryConfig
from pydantic import Field

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.nodes._common import Params, function_node, parse_input, request_id
from agent_runtime.nodes.lexical_extract import batch_search
from agent_runtime.schemas import ExamMention, ExamMentions, ResolvedExam, ResolvedExams
from lab_common import logging as lab_logging
from lab_common import telemetry as lab_telemetry
from lab_common.contracts.rag import MAX_TOP_K

log = lab_logging.get_logger(__name__)


def _named_by_model(mention: ExamMention) -> bool:
    return mention.source != "lexical"


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
            evidences = [m.evidence for m in mentions if _named_by_model(m)]
            result = await batch_search(
                deps.mcp,
                server,
                tool,
                [m.text for m in mentions] + evidences,
                params.top_k,
                request_id=request_id(ctx),
            )
            catalog_sha256 = result.catalog_sha256
            by_evidence = iter(result.results[len(mentions) :])
            items = [
                ResolvedExam(
                    mention=mention,
                    candidates=list(found.candidates),
                    evidence_candidates=(
                        list(next(by_evidence).candidates) if _named_by_model(mention) else None
                    ),
                )
                for mention, found in zip(mentions, result.results, strict=False)
            ]
        log.info(
            "rag_resolve_done",
            node=name,
            mentions=len(items),
            without_candidates=sum(not item.candidates for item in items),
            catalog_sha256=catalog_sha256,
        )
        lab_telemetry.annotate(
            mentions=len(items), without_candidates=sum(not item.candidates for item in items)
        )
        return Event(output=ResolvedExams(items=items))

    return function_node(name, run, retry=retry, timeout=timeout)

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

_DEFAULT_TOP_K = 5

log = lab_logging.get_logger(__name__)


class RagResolveParams(Params):
    top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = _DEFAULT_TOP_K


def rag_resolve(  # noqa: PLR0913 - one keyword per node param
    *,
    name: str,
    deps: RuntimeDeps,
    server: str,
    tool: str,
    top_k: int = _DEFAULT_TOP_K,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
) -> FunctionNode:
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

        without_candidates = sum(not item.candidates for item in items)
        log.info(
            "rag_resolve_done",
            node=name,
            mentions=len(items),
            without_candidates=without_candidates,
            catalog_sha256=catalog_sha256,
        )
        lab_telemetry.annotate(mentions=len(items), without_candidates=without_candidates)
        return Event(output=ResolvedExams(items=items))

    return function_node(name, run, retry=retry, timeout=timeout)


# Audit F02: the gate accepts a name from the model only if its evidence resolves the same.
def _named_by_model(mention: ExamMention) -> bool:
    return mention.source != "lexical"

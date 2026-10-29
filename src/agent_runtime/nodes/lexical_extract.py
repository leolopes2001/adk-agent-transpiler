import re
from typing import Annotated, Any

from google.adk import Context, Event
from google.adk.workflow import FunctionNode, RetryConfig
from pydantic import Field, ValidationError

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.errors import PermanentError
from agent_runtime.gateways.mcp import McpGateway
from agent_runtime.nodes._common import (
    Params,
    function_node,
    parse_input,
    request_id,
    search_form,
    strip_list_marker,
)
from agent_runtime.schemas import ExamMention, ExamMentions, GuardedDocument
from lab_common import logging as lab_logging
from lab_common import telemetry as lab_telemetry
from lab_common.contracts.rag import MAX_BATCH_QUERIES, BatchSearchResult

MIN_SEGMENT_CHARS = 2
# Off a list item, two letters are mostly initials or states ("MG" would match Magnésio).
MIN_UNLISTED_SEGMENT_CHARS = 3
MAX_SEGMENT_CHARS = 80
HEADER_LABELS = frozenset(
    {
        "paciente",
        "data",
        "assinatura",
        "crm",
        "convenio",
        "endereco",
        "nascimento",
        "cpf",
        "rg",
        "cns",
        "telefone",
        "carteirinha",
        "medico",
        "medica",
        "solicitante",
        "dr",
        "dra",
    }
)
# Never " e ": it is part of names such as "Colesterol total e frações".
_SEPARATORS = re.compile(r"[;•,]")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")
_DEFAULT_MIN_SCORE = 0.8
_DEFAULT_MAX_LINES = 60

log = lab_logging.get_logger(__name__)


class LexicalExtractParams(Params):
    min_score: Annotated[float, Field(ge=0, le=1)] = _DEFAULT_MIN_SCORE
    max_lines: Annotated[int, Field(ge=1)] = _DEFAULT_MAX_LINES


# Runs next to the model: an injection that makes the model omit an exam cannot hide it.
def lexical_extract(  # noqa: PLR0913 - one keyword per node param
    *,
    name: str,
    deps: RuntimeDeps,
    server: str,
    tool: str,
    min_score: float = _DEFAULT_MIN_SCORE,
    max_lines: int = _DEFAULT_MAX_LINES,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
) -> FunctionNode:
    params = LexicalExtractParams(min_score=min_score, max_lines=max_lines)

    async def run(ctx: Context, node_input: Any) -> Event:
        document = parse_input(GuardedDocument, node_input)
        candidates = segments(document.text, params.max_lines)
        mentions: list[ExamMention] = []
        if candidates:
            result = await batch_search(
                deps.mcp, server, tool, candidates, top_k=1, request_id=request_id(ctx)
            )
            mentions = [
                ExamMention(text=segment, source="lexical", evidence=segment)
                for segment, found in zip(candidates, result.results, strict=True)
                if found.candidates and found.candidates[0].score >= params.min_score
            ]

        log.info("lexical_extract_done", node=name, segments=len(candidates), found=len(mentions))
        lab_telemetry.annotate(segments=len(candidates), mentions=len(mentions))
        return Event(output=ExamMentions(items=mentions))

    return function_node(name, run, retry=retry, timeout=timeout)


def segments(text: str, max_lines: int) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for line in text.split("\n")[:max_lines]:
        listed = strip_list_marker(line) != line.strip()
        shortest = MIN_SEGMENT_CHARS if listed else MIN_UNLISTED_SEGMENT_CHARS
        for piece in _SEPARATORS.split(line):
            segment = strip_list_marker(piece.strip())
            if not shortest <= len(segment) <= MAX_SEGMENT_CHARS:
                continue

            if _is_header(segment) or _mostly_placeholders(segment):
                continue

            key = search_form(segment)
            if key and key not in seen:
                seen.add(key)
                found.append(segment)

    return found


async def batch_search(  # noqa: PLR0913 - the tool's arguments plus where to send them
    mcp: McpGateway,
    server: str,
    tool: str,
    queries: list[str],
    top_k: int,
    *,
    request_id: str,
) -> BatchSearchResult:
    results = []
    contract = None
    for start in range(0, len(queries), MAX_BATCH_QUERIES):
        chunk = queries[start : start + MAX_BATCH_QUERIES]
        args = {"queries": chunk, "top_k": top_k, "request_id": request_id}
        body = await mcp.call(server, tool, args)
        try:
            contract = BatchSearchResult.model_validate(body)
        except ValidationError:
            raise PermanentError("RAG_INVALID_RESULT", "result breaks the contract") from None

        if len(contract.results) != len(chunk):
            raise PermanentError("RAG_INVALID_RESULT", "search result count differs from queries")

        results.extend(contract.results)

    if contract is None:
        raise ValueError("batch_search needs at least one query")

    return contract.model_copy(update={"results": results})


def _is_header(segment: str) -> bool:
    if segment.endswith(":"):
        return True

    words = search_form(segment.split(":", 1)[0]).split()
    return bool(words) and words[0] in HEADER_LABELS


def _mostly_placeholders(segment: str) -> bool:
    placeholders = sum(len(m.group()) for m in _PLACEHOLDER.finditer(segment))
    rest = _PLACEHOLDER.sub("", segment)
    return placeholders * 2 >= len(segment) or sum(c.isalnum() for c in rest) < MIN_SEGMENT_CHARS

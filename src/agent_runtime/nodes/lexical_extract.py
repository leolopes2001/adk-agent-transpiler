"""`lexical_extract`: finds exam names in the document without the model.

It runs next to the LLM extractor, so an injection that makes the model omit
an exam does not make the exam disappear. The text is cut into candidate
segments (lines, then `;`, `•` and `,`; never " e ", which is part of names
such as "Colesterol total e frações"). Segments that are headers, mostly
placeholders or of implausible length are dropped; the rest go to the
catalog search in one batch, and segments whose best match scores at least
`min_score` become mentions.
"""

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
    search_form,
    strip_list_marker,
)
from agent_runtime.schemas import ExamMention, ExamMentions, GuardedDocument
from lab_common import logging as lab_logging
from lab_common.contracts.rag import MAX_BATCH_QUERIES, BatchSearchResult

MIN_SEGMENT_CHARS = 3
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
_SEPARATORS = re.compile(r"[;•,]")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")

log = lab_logging.get_logger(__name__)


class LexicalExtractParams(Params):
    min_score: Annotated[float, Field(ge=0, le=1)] = 0.8
    max_lines: Annotated[int, Field(ge=1)] = 60


def _is_header(segment: str) -> bool:
    if segment.endswith(":"):
        return True
    words = search_form(segment.split(":", 1)[0]).split()
    return bool(words) and words[0] in HEADER_LABELS


def _mostly_placeholders(segment: str) -> bool:
    placeholders = sum(len(m.group()) for m in _PLACEHOLDER.finditer(segment))
    rest = _PLACEHOLDER.sub("", segment)
    return placeholders * 2 >= len(segment) or sum(c.isalnum() for c in rest) < MIN_SEGMENT_CHARS


def segments(text: str, max_lines: int) -> list[str]:
    """Candidate exam mentions in document order, without duplicates."""
    found: list[str] = []
    seen: set[str] = set()
    for line in text.split("\n")[:max_lines]:
        for piece in _SEPARATORS.split(line):
            segment = strip_list_marker(piece.strip())
            if not MIN_SEGMENT_CHARS <= len(segment) <= MAX_SEGMENT_CHARS:
                continue
            if _is_header(segment) or _mostly_placeholders(segment):
                continue
            key = search_form(segment)
            if key and key not in seen:
                seen.add(key)
                found.append(segment)
    return found


async def batch_search(
    mcp: McpGateway, server: str, tool: str, queries: list[str], top_k: int
) -> BatchSearchResult:
    """One `batch_search` result for all `queries`, in order (chunked by the batch limit)."""
    results = []
    contract = None
    for start in range(0, len(queries), MAX_BATCH_QUERIES):
        chunk = queries[start : start + MAX_BATCH_QUERIES]
        body = await mcp.call(server, tool, {"queries": chunk, "top_k": top_k})
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
    params = LexicalExtractParams(min_score=min_score, max_lines=max_lines)

    async def run(ctx: Context, node_input: Any) -> Event:
        document = parse_input(GuardedDocument, node_input)
        candidates = segments(document.text, params.max_lines)
        mentions: list[ExamMention] = []
        if candidates:
            result = await batch_search(deps.mcp, server, tool, candidates, top_k=1)
            mentions = [
                ExamMention(text=segment, source="lexical", evidence=segment)
                for segment, found in zip(candidates, result.results, strict=True)
                if found.candidates and found.candidates[0].score >= params.min_score
            ]
        log.info("lexical_extract_done", node=name, segments=len(candidates), found=len(mentions))
        return Event(output=ExamMentions(items=mentions))

    return function_node(name, run, retry=retry, timeout=timeout)

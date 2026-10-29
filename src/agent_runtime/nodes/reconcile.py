import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from google.adk import Context, Event
from google.adk.workflow import FunctionNode
from pydantic import Field
from rapidfuzz import fuzz

from agent_runtime.errors import PermanentError
from agent_runtime.guardrails.tripwire import ERROR_CODE as TRIPWIRE_CODE
from agent_runtime.guardrails.tripwire import TRIPWIRE_STATE_KEY
from agent_runtime.nodes._common import Params, function_node, parse_input, search_form
from agent_runtime.schemas import ExamMention, ExamMentions, ExtractedExam, ExtractedExams
from lab_common import logging as lab_logging
from lab_common.text import normalize_for_search

# Near-identical names send the same query to the catalog, so one item absorbs the other.
DUPLICATE_MIN_RATIO = 90
# One character is in almost any document: shorter evidence grounds nothing.
MIN_EVIDENCE_CHARS = 2
# Letters, digits and the punctuation of exam names: no placeholders or markup.
_NAME_CHARSET = re.compile(r"^[^\W_]+(?:[ .,()/+%'\-][^\W_]*)*[.)%]?$")
_DEFAULT_GROUNDING_MIN_RATIO = 90
_DEFAULT_MAX_ITEMS = 30

log = lab_logging.get_logger(__name__)


class ReconcileParams(Params):
    sources: Mapping[str, str]
    grounding_min_ratio: Annotated[int, Field(ge=0, le=100)] = _DEFAULT_GROUNDING_MIN_RATIO
    max_items: Annotated[int, Field(ge=1)] = _DEFAULT_MAX_ITEMS


@dataclass(frozen=True, slots=True)
class _MergeCounts:
    llm: int
    lexical: int
    ungrounded: int
    invalid_name: int


def reconcile(
    *,
    name: str,
    sources: Mapping[str, str],
    grounding_min_ratio: int = _DEFAULT_GROUNDING_MIN_RATIO,
    max_items: int = _DEFAULT_MAX_ITEMS,
) -> FunctionNode:
    params = ReconcileParams(
        sources=dict(sources), grounding_min_ratio=grounding_min_ratio, max_items=max_items
    )
    if set(params.sources) != {"llm", "lexical"}:
        raise ValueError("reconcile sources must name the 'llm' and 'lexical' inputs")

    async def run(ctx: Context, node_input: Any) -> Event:
        # The tripwire's blocking answer does not fail the model step (ADR-0009); this does.
        if ctx.state.get(TRIPWIRE_STATE_KEY):
            raise PermanentError(TRIPWIRE_CODE, "the model call was blocked by the PII tripwire")

        joined: Mapping[str, Any] = node_input if isinstance(node_input, Mapping) else {}
        llm_output = joined.get(params.sources["llm"])
        lexical_output = joined.get(params.sources["lexical"])
        llm = parse_input(ExtractedExams, llm_output).exams if llm_output is not None else []
        lexical = (
            parse_input(ExamMentions, lexical_output).items if lexical_output is not None else []
        )
        document = str(ctx.state.get("doc.text") or "")

        items, counts = merge(llm, lexical, document, params)
        if counts.ungrounded or counts.invalid_name:
            log.warning(
                "grounding_rejected",
                node=name,
                ungrounded=counts.ungrounded,
                invalid_name=counts.invalid_name,
            )
        log.info(
            "reconcile_done",
            node=name,
            llm=counts.llm,
            lexical=counts.lexical,
            llm_missing=llm_output is None,
            items=len(items),
            both=sum(item.source == "both" for item in items),
        )
        return Event(output=ExamMentions(items=items))

    return function_node(name, run)


def merge(
    llm_items: list[ExtractedExam],
    lexical_items: list[ExamMention],
    document: str,
    params: ReconcileParams,
) -> tuple[list[ExamMention], _MergeCounts]:
    grounded_items, ungrounded, invalid = _grounded_llm_items(
        llm_items, normalize_for_search(document), params.grounding_min_ratio
    )
    mentions = _in_document_order(grounded_items, lexical_items, document)
    items = _one_per_search_form(mentions)
    counts = _MergeCounts(len(grounded_items), len(lexical_items), ungrounded, invalid)
    return items[: params.max_items], counts


def valid_name(name: str) -> bool:
    return bool(_NAME_CHARSET.match(name.strip()))


def grounded(evidence: str, document_form: str, min_ratio: int) -> bool:
    form = search_form(evidence)
    return len(form) >= MIN_EVIDENCE_CHARS and fuzz.partial_ratio(form, document_form) >= min_ratio


def _grounded_llm_items(
    llm_items: list[ExtractedExam], document_form: str, min_ratio: int
) -> tuple[list[ExtractedExam], int, int]:
    kept: list[ExtractedExam] = []
    ungrounded = invalid = 0
    for item in llm_items:
        if not valid_name(item.name):
            invalid += 1
        elif not grounded(item.evidence, document_form, min_ratio):
            ungrounded += 1
        else:
            kept.append(item)

    return kept, ungrounded, invalid


# A lexical item stays unless the names match: the model's name must never hide the document's.
def _in_document_order(
    llm_items: list[ExtractedExam], lexical_items: list[ExamMention], document: str
) -> list[ExamMention]:
    absorbed: set[int] = set()
    positioned: list[tuple[int, int, ExamMention]] = []
    for order, item in enumerate(llm_items):
        twins = [i for i, lex in enumerate(lexical_items) if _same_name(item, lex)]
        absorbed.update(twins)
        confirmed = bool(twins) or any(_same_segment(item, lex) for lex in lexical_items)
        mention = ExamMention(
            text=item.name.strip(), source="both" if confirmed else "llm", evidence=item.evidence
        )
        positioned.append((_document_position(item.evidence, document), order, mention))

    offset = len(llm_items)
    for index, lex in enumerate(lexical_items):
        if index not in absorbed:
            positioned.append((_document_position(lex.evidence, document), offset + index, lex))

    positioned.sort(key=lambda entry: (entry[0], entry[1]))
    return [mention for _, _, mention in positioned]


def _one_per_search_form(mentions: list[ExamMention]) -> list[ExamMention]:
    items: list[ExamMention] = []
    seen: dict[str, int] = {}
    for mention in mentions:
        key = search_form(mention.text)
        if key in seen:
            first = items[seen[key]]
            if first.source != mention.source:
                items[seen[key]] = first.model_copy(update={"source": "both"})
            continue

        seen[key] = len(items)
        items.append(mention)

    return items


def _same_name(llm: ExtractedExam, lexical: ExamMention) -> bool:
    ratio = fuzz.token_sort_ratio(search_form(llm.name), search_form(lexical.text))
    return ratio >= DUPLICATE_MIN_RATIO


def _same_segment(llm: ExtractedExam, lexical: ExamMention) -> bool:
    return search_form(llm.evidence) == search_form(lexical.text)


def _document_position(text: str, document: str) -> int:
    index = document.find(text)
    if index < 0:
        index = document.casefold().find(text.casefold())

    return index if index >= 0 else len(document)

"""`reconcile`: merges the LLM and lexical mentions into one list.

1. Fails the run if the PII tripwire blocked the model call (ADR-0009).
2. LLM items must be grounded: their evidence has to be in the guarded text
   (`partial_ratio >= grounding_min_ratio` on the search form). Items without
   grounding, or with a name outside the allowed characters (placeholders,
   markup), are dropped and counted in the log.
3. An LLM item absorbs a lexical item only when their names are
   near-identical (`token_sort_ratio >= 90`): both would send the same query
   to the catalog. When the LLM evidence is the lexical segment but the
   names differ ("Proteína C reativa" from "PCR"), the LLM item is marked
   `both` and the lexical item stays: the model's name must never hide what
   the document says (a model can be steered to name things wrongly). The
   gate later collapses mentions that resolve to the same code.
4. Items are ordered by where they appear in the document, then capped.

A missing source (the agent produced nothing) counts as an empty list.
"""

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

DUPLICATE_MIN_RATIO = 90
_NAME_CHARSET = re.compile(r"^[^\W_]+(?:[ .,()/+%'\-][^\W_]*)*[.)%]?$")

log = lab_logging.get_logger(__name__)


class ReconcileParams(Params):
    sources: Mapping[str, str]
    grounding_min_ratio: Annotated[int, Field(ge=0, le=100)] = 90
    max_items: Annotated[int, Field(ge=1)] = 30


@dataclass(frozen=True, slots=True)
class _Counts:
    llm: int
    lexical: int
    ungrounded: int
    invalid_name: int


def valid_name(name: str) -> bool:
    """Letters and digits with the punctuation exam names use; no brackets or markup."""
    return bool(_NAME_CHARSET.match(name.strip()))


def grounded(evidence: str, document_form: str, min_ratio: int) -> bool:
    form = search_form(evidence)
    return bool(form) and fuzz.partial_ratio(form, document_form) >= min_ratio


def _same_name(llm: ExtractedExam, lexical: ExamMention) -> bool:
    ratio = fuzz.token_sort_ratio(search_form(llm.name), search_form(lexical.text))
    return ratio >= DUPLICATE_MIN_RATIO


def _same_segment(llm: ExtractedExam, lexical: ExamMention) -> bool:
    return search_form(llm.evidence) == search_form(lexical.text)


def _position(text: str, document: str) -> int:
    index = document.find(text)
    if index < 0:
        index = document.casefold().find(text.casefold())
    return index if index >= 0 else len(document)


def merge(
    llm_items: list[ExtractedExam],
    lexical_items: list[ExamMention],
    document: str,
    params: ReconcileParams,
) -> tuple[list[ExamMention], _Counts]:
    document_form = normalize_for_search(document)
    accepted: list[ExtractedExam] = []
    ungrounded = invalid = 0
    for item in llm_items:
        if not valid_name(item.name):
            invalid += 1
        elif not grounded(item.evidence, document_form, params.grounding_min_ratio):
            ungrounded += 1
        else:
            accepted.append(item)

    absorbed: set[int] = set()
    merged: list[tuple[int, int, ExamMention]] = []
    for order, item in enumerate(accepted):
        twins = [i for i, lex in enumerate(lexical_items) if _same_name(item, lex)]
        absorbed.update(twins)
        confirmed = bool(twins) or any(_same_segment(item, lex) for lex in lexical_items)
        mention = ExamMention(
            text=item.name.strip(), source="both" if confirmed else "llm", evidence=item.evidence
        )
        merged.append((_position(item.evidence, document), order, mention))
    offset = len(accepted)
    for index, lex in enumerate(lexical_items):
        if index not in absorbed:
            merged.append((_position(lex.evidence, document), offset + index, lex))
    merged.sort(key=lambda entry: (entry[0], entry[1]))

    items: list[ExamMention] = []
    seen: dict[str, int] = {}
    for _, _, mention in merged:
        key = search_form(mention.text)
        if key in seen:
            first = items[seen[key]]
            if first.source != mention.source:
                items[seen[key]] = first.model_copy(update={"source": "both"})
            continue
        seen[key] = len(items)
        items.append(mention)
    counts = _Counts(len(accepted), len(lexical_items), ungrounded, invalid)
    return items[: params.max_items], counts


def reconcile(
    *,
    name: str,
    sources: Mapping[str, str],
    grounding_min_ratio: int = 90,
    max_items: int = 30,
) -> FunctionNode:
    """Merges LLM and lexical mentions; `sources` names the join inputs by role."""
    params = ReconcileParams(
        sources=dict(sources), grounding_min_ratio=grounding_min_ratio, max_items=max_items
    )
    if set(params.sources) != {"llm", "lexical"}:
        raise ValueError("reconcile sources must name the 'llm' and 'lexical' inputs")

    async def run(ctx: Context, node_input: Any) -> Event:
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

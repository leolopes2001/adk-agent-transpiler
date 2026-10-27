from collections.abc import Mapping
from typing import Any

import pytest
from runtime_fakes import FakeMcp, candidate, rag_body, run_node

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.errors import TransientError
from agent_runtime.nodes.rag_resolve import rag_resolve
from agent_runtime.schemas import ExamMention, ExamMentions, ResolvedExams

pytestmark = pytest.mark.unit


def _deps(fake: FakeMcp) -> RuntimeDeps:
    deps = RuntimeDeps.lazy()
    deps.use(mcp=fake)
    return deps


def _mentions(*texts: str) -> ExamMentions:
    return ExamMentions(items=[ExamMention(text=t, source="lexical", evidence=t) for t in texts])


async def test_one_call_for_all_mentions_with_candidates_kept_in_order() -> None:
    def respond(args: Mapping[str, Any]) -> dict[str, Any]:
        return rag_body(
            [candidate("LAB-10044", "T4 livre", 0.8), candidate("LAB-10045", "T4 total", 0.8)],
            [],
            queries=list(args["queries"]),
        )

    fake = FakeMcp()
    fake.script("rag", "batch_search", respond)
    node = rag_resolve(name="resolve", deps=_deps(fake), server="rag", tool="batch_search", top_k=3)

    run = await run_node(node, _mentions("T4", "Ressonância"))

    [(_, _, args)] = fake.calls
    assert args == {"queries": ["T4", "Ressonância"], "top_k": 3}
    items = ResolvedExams.model_validate(run.output).items
    assert [item.mention.text for item in items] == ["T4", "Ressonância"]
    assert [c.code for c in items[0].candidates] == ["LAB-10044", "LAB-10045"]
    assert items[1].candidates == []


async def test_model_items_also_resolve_their_evidence_in_the_same_call() -> None:
    by_query = {
        "Proteína C reativa": [candidate("LAB-10037", "Proteína C reativa", 1.0)],
        "TSH": [candidate("LAB-10043", "TSH", 1.0)],
        "7. PCR": [candidate("LAB-10037", "Proteína C reativa", 1.0)],
    }

    def respond(args: Mapping[str, Any]) -> dict[str, Any]:
        queries = list(args["queries"])
        return rag_body(*[by_query[q] for q in queries], queries=queries)

    fake = FakeMcp()
    fake.script("rag", "batch_search", respond)
    node = rag_resolve(name="resolve", deps=_deps(fake), server="rag", tool="batch_search")
    mentions = ExamMentions(
        items=[
            ExamMention(text="Proteína C reativa", source="both", evidence="7. PCR"),
            ExamMention(text="TSH", source="lexical", evidence="TSH"),
        ]
    )

    run = await run_node(node, mentions)

    [(_, _, args)] = fake.calls
    assert args["queries"] == ["Proteína C reativa", "TSH", "7. PCR"]
    model_item, lexical_item = ResolvedExams.model_validate(run.output).items
    assert model_item.evidence_candidates is not None
    assert [c.code for c in model_item.evidence_candidates] == ["LAB-10037"]
    assert lexical_item.evidence_candidates is None


async def test_no_mentions_no_call() -> None:
    fake = FakeMcp()
    node = rag_resolve(name="resolve", deps=_deps(fake), server="rag", tool="batch_search")

    run = await run_node(node, ExamMentions(items=[]))

    assert ResolvedExams.model_validate(run.output).items == []
    assert fake.calls == []


async def test_transient_errors_propagate_for_the_node_retry() -> None:
    fake = FakeMcp()
    fake.script("rag", "batch_search", TransientError("MCP_UNAVAILABLE"))
    node = rag_resolve(name="resolve", deps=_deps(fake), server="rag", tool="batch_search")

    with pytest.raises(TransientError):
        await run_node(node, _mentions("TSH"))


def test_top_k_is_limited_by_the_contract() -> None:
    with pytest.raises(ValueError, match="top_k"):
        rag_resolve(name="r", deps=RuntimeDeps.lazy(), server="rag", tool="batch_search", top_k=11)

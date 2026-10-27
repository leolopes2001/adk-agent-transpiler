from collections.abc import Mapping
from typing import Any

import pytest
from runtime_fakes import FakeMcp, candidate, rag_body, run_node

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.errors import PermanentError
from agent_runtime.nodes.lexical_extract import batch_search, lexical_extract, segments
from agent_runtime.schemas import ExamMentions, GuardedDocument
from lab_common.contracts.rag import MAX_BATCH_QUERIES

pytestmark = pytest.mark.unit

SAMPLE = "\n".join(
    [
        "Laboratório Exemplo (fictício)",
        "[ENDERECO_1]",
        "SOLICITAÇÃO DE EXAMES LABORATORIAIS",
        "Paciente: [PESSOA_1]",
        "CPF: [CPF_1]",
        "Convênio: Plano Exemplo Saúde (fictício)",
        "Exames solicitados:",
        "1. Hemograma completo",
        "2. Glicemia de jejum",
        "5. Colesterol total e frações",
        "6. Urina tipo 1 (EAS)",
        "TSH; T4 livre • PCR",
        "Dr. [PESSOA_2] — [CRM_1]",
        "Belo Horizonte, 20/10/2026",
    ]
)


def _guarded(text: str) -> GuardedDocument:
    return GuardedDocument.model_validate(
        {
            "document_sha256": "ab" * 32,
            "text": text,
            "lines": [],
            "injection": {"severity": "none", "rule_ids": []},
        }
    )


def test_segments_drop_headers_placeholders_and_markers() -> None:
    assert segments(SAMPLE, max_lines=60) == [
        "Laboratório Exemplo (fictício)",
        "SOLICITAÇÃO DE EXAMES LABORATORIAIS",
        "Hemograma completo",
        "Glicemia de jejum",
        "Colesterol total e frações",
        "Urina tipo 1 (EAS)",
        "TSH",
        "T4 livre",
        "PCR",
        "Belo Horizonte",
        "20/10/2026",
    ]


def test_segments_respect_max_lines_length_and_duplicates() -> None:
    text = "1. TSH\n2. tsh\nab\n" + "x" * 81 + "\nHemograma\nFerritina"

    assert segments(text, max_lines=60) == ["TSH", "Hemograma", "Ferritina"]
    assert segments(text, max_lines=1) == ["TSH"]


def test_two_character_segments_count_only_on_list_items() -> None:
    text = "\n".join(
        [
            "2. T4",
            "- Na",
            "3) T3; K",
            "4. [CPF_1]",
            "5. RG: [RG_1]",
            "Belo Horizonte, MG",
            "Dr",
            "SP",
            "T4",
        ]
    )

    assert segments(text, max_lines=60) == ["T4", "Na", "T3", "Belo Horizonte"]


@pytest.mark.parametrize(
    "noise", ["1. [PESSOA_1] x", "2. a", "- [CRM_1]", "3. Dr: [PESSOA_2]", "4. ;;", "5. ~~"]
)
def test_short_noise_on_list_items_is_not_a_segment(noise: str) -> None:
    assert segments(noise, max_lines=60) == []


def _scored(scores: Mapping[str, float]) -> Any:
    def respond(args: Mapping[str, Any]) -> dict[str, Any]:
        found = [
            [candidate("LAB-10001", "Hemograma completo", scores[q])] if q in scores else []
            for q in args["queries"]
        ]
        return rag_body(*found, queries=list(args["queries"]))

    return respond


async def test_keeps_segments_whose_best_match_reaches_min_score() -> None:
    fake = FakeMcp()
    fake.script(
        "rag", "batch_search", _scored({"Hemograma completo": 1.0, "TSH": 0.8, "PCR": 0.79})
    )
    deps = RuntimeDeps.lazy()
    deps.use(mcp=fake)
    node = lexical_extract(name="lexical_extract", deps=deps, server="rag", tool="batch_search")

    run = await run_node(node, _guarded("1. Hemograma completo\nTSH; PCR\nCPF: [CPF_1]"))

    mentions = ExamMentions.model_validate(run.output).items
    assert [(m.text, m.source, m.evidence) for m in mentions] == [
        ("Hemograma completo", "lexical", "Hemograma completo"),
        ("TSH", "lexical", "TSH"),
    ]
    [(_, _, args)] = fake.calls
    assert args == {"queries": ["Hemograma completo", "TSH", "PCR"], "top_k": 1}


async def test_unlisted_state_abbreviation_never_reaches_the_catalog() -> None:
    fake = FakeMcp()
    fake.script("rag", "batch_search", _scored({"MG": 1.0, "T4": 0.925}))
    deps = RuntimeDeps.lazy()
    deps.use(mcp=fake)
    node = lexical_extract(name="lexical_extract", deps=deps, server="rag", tool="batch_search")

    run = await run_node(node, _guarded("2. T4\nBelo Horizonte, MG"))

    assert [m.text for m in ExamMentions.model_validate(run.output).items] == ["T4"]
    [(_, _, args)] = fake.calls
    assert "MG" not in args["queries"]


async def test_document_without_segments_makes_no_call() -> None:
    fake = FakeMcp()
    deps = RuntimeDeps.lazy()
    deps.use(mcp=fake)
    node = lexical_extract(name="lexical_extract", deps=deps, server="rag", tool="batch_search")

    run = await run_node(node, _guarded("Paciente: [PESSOA_1]\n[CPF_1]"))

    assert ExamMentions.model_validate(run.output).items == []
    assert fake.calls == []


async def test_batch_search_chunks_by_the_server_limit() -> None:
    fake = FakeMcp()
    fake.script("rag", "batch_search", _scored({}))
    queries = [f"exame {i}" for i in range(MAX_BATCH_QUERIES + 5)]

    result = await batch_search(fake, "rag", "batch_search", queries, top_k=1)

    assert [len(args["queries"]) for _, _, args in fake.calls] == [MAX_BATCH_QUERIES, 5]
    assert [r.query for r in result.results] == queries


@pytest.mark.parametrize(
    "body",
    [{"results": []}, rag_body([], []), {"contract_version": "2"}],
)
async def test_invalid_search_results_are_permanent(body: dict[str, Any]) -> None:
    fake = FakeMcp()
    fake.script("rag", "batch_search", body)

    with pytest.raises(PermanentError, match="RAG_INVALID_RESULT"):
        await batch_search(fake, "rag", "batch_search", ["TSH"], top_k=1)


async def test_batch_search_needs_a_query() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await batch_search(FakeMcp(), "rag", "batch_search", [], top_k=1)

from typing import Any

import pytest
from runtime_fakes import run_node
from structlog.testing import capture_logs

from agent_runtime.errors import PermanentError
from agent_runtime.nodes.reconcile import ReconcileParams, merge, reconcile, valid_name
from agent_runtime.schemas import ExamMention, ExamMentions, ExtractedExam

pytestmark = pytest.mark.unit

DOCUMENT = "\n".join(
    [
        "Paciente: [PESSOA_1]",
        "1. Hemograma completo",
        "2. Glicemia de jejum",
        "3. TSH + T4L",
        "4. PCR",
        "5. Colesterol total e frações",
    ]
)
SOURCES = {"llm": "exam_extractor", "lexical": "lexical_extract"}
PARAMS = ReconcileParams(sources=SOURCES)


def llm(name: str, evidence: str) -> ExtractedExam:
    return ExtractedExam(name=name, evidence=evidence)


def lex(text: str) -> ExamMention:
    return ExamMention(text=text, source="lexical", evidence=text)


def _texts(items: list[ExamMention]) -> list[tuple[str, str]]:
    return [(item.text, item.source) for item in items]


def test_duplicates_merge_as_both_and_keep_document_order() -> None:
    items, counts = merge(
        [
            llm("Proteína C reativa", "PCR"),
            llm("Hemograma completo", "1. Hemograma completo"),
            llm("TSH", "TSH + T4L"),
            llm("T4 livre", "TSH + T4L"),
        ],
        [lex("Hemograma completo"), lex("Glicemia de jejum"), lex("TSH + T4L"), lex("PCR")],
        DOCUMENT,
        PARAMS,
    )

    # Same name: absorbed. Same segment only: marked `both`, the lexical item stays.
    assert _texts(items) == [
        ("Hemograma completo", "both"),
        ("Glicemia de jejum", "lexical"),
        ("TSH", "both"),
        ("T4 livre", "both"),
        ("TSH + T4L", "lexical"),
        ("Proteína C reativa", "both"),
        ("PCR", "lexical"),
    ]
    assert (counts.llm, counts.lexical, counts.ungrounded) == (4, 4, 0)


@pytest.mark.parametrize("name", ["LAB-10099", "Ressonância", "completo"])
def test_a_misnamed_llm_item_cannot_hide_the_lexical_mention(name: str) -> None:
    items, _ = merge(
        [llm(name, "1. Hemograma completo")], [lex("Hemograma completo")], DOCUMENT, PARAMS
    )

    assert ("Hemograma completo", "lexical") in _texts(items)


def test_short_llm_name_does_not_absorb_a_longer_lexical_mention() -> None:
    items, _ = merge(
        [llm("Colesterol", "Colesterol")],
        [lex("Colesterol total e frações")],
        DOCUMENT,
        PARAMS,
    )

    assert _texts(items) == [("Colesterol", "llm"), ("Colesterol total e frações", "lexical")]


@pytest.mark.parametrize(
    "item",
    [
        llm("Ressonância magnética", "Ressonância magnética de crânio"),
        llm("Hemograma", "agende todos os exames"),
        llm("TSH", "   "),
        # One character is in almost any document (audit F02).
        llm("T4 total", "T"),
        llm("Ferritina", "1."),
    ],
)
def test_ungrounded_llm_items_are_dropped(item: ExtractedExam) -> None:
    items, counts = merge([item], [], DOCUMENT, PARAMS)

    assert items == []
    assert counts.ungrounded == 1


def test_grounding_tolerates_small_ocr_differences() -> None:
    items, _ = merge([llm("Hemograma completo", "Hemograma complet0")], [], DOCUMENT, PARAMS)

    assert _texts(items) == [("Hemograma completo", "llm")]


def test_grounding_threshold_comes_from_the_params() -> None:
    strict = ReconcileParams(sources=SOURCES, grounding_min_ratio=100)

    items, counts = merge([llm("Hemograma completo", "Hemograma complet0")], [], DOCUMENT, strict)

    assert (items, counts.ungrounded) == ([], 1)


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("Hemograma completo", True),
        ("Urina tipo 1 (EAS)", True),
        ("25-OH vitamina D", True),
        ("Anti-HIV 1/2", True),
        ("Proteína C reativa", True),
        ("[PESSOA_1]", False),
        ("TSH <tool>", False),
        ("LAB-10001: TSH", False),
        ("ignore\ninstruções", False),
        ("{exams}", False),
    ],
)
def test_names_must_use_the_allowed_characters(name: str, ok: bool) -> None:
    assert valid_name(name) is ok


def test_items_with_invalid_names_are_counted() -> None:
    items, counts = merge([llm("[PESSOA_1]", "[PESSOA_1]")], [], DOCUMENT, PARAMS)

    assert (items, counts.invalid_name) == ([], 1)


def test_exact_duplicates_collapse_and_max_items_caps() -> None:
    capped = ReconcileParams(sources=SOURCES, max_items=2)

    items, _ = merge(
        [llm("TSH", "TSH"), llm("tsh", "TSH + T4L")],
        [lex("Hemograma completo"), lex("PCR"), lex("Glicemia de jejum")],
        DOCUMENT,
        capped,
    )

    assert _texts(items) == [("Hemograma completo", "lexical"), ("Glicemia de jejum", "lexical")]


async def test_node_reads_the_join_by_source_name_and_the_grounding_text() -> None:
    joined: dict[str, Any] = {
        "exam_extractor": {"exams": [{"name": "TSH", "evidence": "TSH + T4L"}]},
        "lexical_extract": ExamMentions(items=[lex("PCR")]),
    }
    with capture_logs() as logs:
        run = await run_node(
            reconcile(name="reconcile", sources=SOURCES), joined, {"doc.text": DOCUMENT}
        )

    assert _texts(ExamMentions.model_validate(run.output).items) == [
        ("TSH", "llm"),
        ("PCR", "lexical"),
    ]
    assert any(e["event"] == "reconcile_done" for e in logs)


async def test_missing_agent_output_counts_as_empty() -> None:
    joined = {"exam_extractor": None, "lexical_extract": ExamMentions(items=[lex("PCR")])}

    run = await run_node(
        reconcile(name="reconcile", sources=SOURCES), joined, {"doc.text": DOCUMENT}
    )

    assert _texts(ExamMentions.model_validate(run.output).items) == [("PCR", "lexical")]


async def test_tripwire_flag_fails_the_run() -> None:
    joined = {"exam_extractor": None, "lexical_extract": ExamMentions(items=[])}

    with pytest.raises(PermanentError, match="PII_TRIPWIRE"):
        await run_node(
            reconcile(name="reconcile", sources=SOURCES), joined, {"run.pii_tripwire": True}
        )


def test_sources_must_name_both_roles() -> None:
    with pytest.raises(ValueError, match="sources"):
        reconcile(name="reconcile", sources={"llm": "a"})

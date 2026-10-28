"""The generated workflow end to end: real OCR (Tesseract), RAG and scheduler API
(Postgres in a container), FakeLlm as the model."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from runtime_stack import REPO_ROOT, Stack, run_agent

pytestmark = pytest.mark.integration

SAMPLE = "samples/pedido_medico_ficticio.png"
EXPECTED = json.loads(
    (REPO_ROOT / "samples" / "expected" / "pedido_medico_ficticio.json").read_text("utf-8")
)
EXPECTED_CODES = [exam["code"] for exam in EXPECTED["exams"]]
FONT = REPO_ROOT / "tests" / "fixtures" / "fonts" / "DejaVuSans.ttf"


def _page(path: Path, lines: list[str]) -> None:
    page = Image.new("L", (1654, 1200), 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(str(FONT), 40)
    for index, line in enumerate(lines):
        draw.text((140, 140 + index * 70), line, font=font, fill=0)
    page.save(path)


async def test_sample_order_schedules_the_seven_expected_exams(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = stack.count("appointments")

    run = await run_agent(stack, monkeypatch, SAMPLE)

    assert run.exit_code == 0
    report = run.report
    assert report["outcome"] == "scheduled"
    assert [e["code"] for e in report["exams"] if e["status"] == "accepted"] == EXPECTED_CODES
    assert report["appointment"]["status"] == "scheduled"
    assert report["appointment"]["id"].startswith("apt_")
    assert report["path"] == [
        "ocr",
        "guard",
        "extraction",
        "resolve",
        "gate",
        "schedule",
        "present",
    ]
    assert report["pii_masked"]["CPF"] == 1
    assert report["injection_severity"] == "none"
    assert stack.count("appointments") == before + 1
    assert len(run.llm_requests) == 1


async def test_second_run_is_an_idempotent_replay(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await run_agent(stack, monkeypatch, SAMPLE)
    count = stack.count("appointments")

    second = await run_agent(stack, monkeypatch, SAMPLE)

    assert first.exit_code == second.exit_code == 0
    assert second.report["appointment"]["status"] == "replayed"
    assert second.report["appointment"]["id"] == first.report["appointment"]["id"]
    assert stack.count("appointments") == count


async def test_injection_sample_is_blocked_before_the_model(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await run_agent(stack, monkeypatch, "samples/pedido_injecao.png")

    assert (run.exit_code, run.report["outcome"]) == (3, "INPUT_BLOCKED")
    assert run.report["path"] == ["ocr", "guard", "abort"]
    assert run.report["injection_severity"] == "high"
    assert run.llm_requests == []


async def test_blank_page_is_low_quality(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copy(REPO_ROOT / "tests" / "data" / "images" / "blank.png", stack.work / "blank.png")

    run = await run_agent(stack, monkeypatch, "blank.png")

    assert (run.exit_code, run.report["outcome"]) == (4, "OCR_LOW_QUALITY")
    assert run.report["path"] == ["ocr", "abort"]


async def test_document_without_exams_resolves_nothing(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    _page(
        stack.work / "atestado.png",
        ["Atestado de comparecimento", "Compareceu à consulta nesta data.", "Retorno em 30 dias."],
    )

    run = await run_agent(stack, monkeypatch, "atestado.png", fixture="empty.json")

    assert (run.exit_code, run.report["outcome"]) == (5, "NO_EXAMS_RESOLVED")
    assert run.report["path"][-2:] == ["gate", "abort"]


async def test_ambiguous_item_is_rejected_when_non_interactive(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await run_agent(
        stack, monkeypatch, "samples/pedido_ambiguo.png", fixture="pedido_ambiguo.json"
    )

    assert run.exit_code == 0
    exams = {e["mention"]: e for e in run.report["exams"]}
    assert exams["T4"]["status"] == "rejected"
    assert exams["T4"]["reason"] == "needs_human_review"
    # Found by the model and by the lexical extraction (two-character list item).
    assert exams["T4"]["source"] == "both"
    assert exams["Hemograma completo"]["code"] == "LAB-10001"
    assert "review" in run.report["path"]
    assert run.report["appointment"]["status"] in {"scheduled", "replayed"}


async def test_ambiguous_item_is_settled_by_the_operator(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    offered: list[list[str]] = []

    def pick_t4_free(payload: Mapping[str, Any]) -> dict[str, Any]:
        [item] = payload["items"]
        offered.append([o["name"] for o in item["options"]])
        free = next(o["code"] for o in item["options"] if o["name"] == "T4 livre")
        return {"choices": {str(item["index"]): free}}

    run = await run_agent(
        stack,
        monkeypatch,
        "samples/pedido_ambiguo.png",
        fixture="pedido_ambiguo.json",
        interactive=True,
        review=pick_t4_free,
    )

    assert run.exit_code == 0
    assert {"T4 livre", "T4 total"} <= set(offered[0])
    exams = {e["mention"]: e for e in run.report["exams"]}
    assert (exams["T4"]["status"], exams["T4"]["code"]) == ("accepted", "LAB-10044")
    assert run.report["appointment"]["status"] == "scheduled"

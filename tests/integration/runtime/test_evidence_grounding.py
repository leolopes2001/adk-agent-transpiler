"""The model's exam name must resolve to the exam its evidence names (audit F02).

Real OCR, RAG and scheduler; the FakeLlm answers with a name the evidence
does not support. The synthetic pages carry no patient id, so they resolve
without scheduling.
"""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from runtime_stack import REPO_ROOT, Stack, run_agent

pytestmark = pytest.mark.integration

FONT = REPO_ROOT / "tests" / "fixtures" / "fonts" / "DejaVuSans.ttf"


def _page(path: Path, lines: list[str]) -> None:
    page = Image.new("L", (1654, 1200), 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(str(FONT), 40)
    for index, line in enumerate(lines):
        draw.text((140, 140 + index * 70), line, font=font, fill=0)
    page.save(path)


def _accepted(report: dict[str, object]) -> list[str]:
    exams = report["exams"]
    assert isinstance(exams, list)
    return [e["code"] for e in exams if e["status"] == "accepted"]


async def test_name_the_evidence_does_not_support_adds_no_code(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    _page(stack.work / "hemograma.png", ["Pedido de exames", "1. Hemograma completo"])

    run = await run_agent(stack, monkeypatch, "hemograma.png", fixture="nome_divergente.json")

    assert _accepted(run.report) == ["LAB-10001"]
    ferritina = next(e for e in run.report["exams"] if e["mention"] == "Ferritina")
    assert ferritina["status"] != "accepted"


async def test_specific_name_for_an_ambiguous_evidence_goes_to_review(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await run_agent(
        stack, monkeypatch, "samples/pedido_ambiguo.png", fixture="pedido_ambiguo_resolvido.json"
    )

    assert run.exit_code == 0
    assert _accepted(run.report) == ["LAB-10001"]
    assert "review" in run.report["path"]
    t4_total = next(e for e in run.report["exams"] if e["mention"] == "T4 total")
    assert (t4_total["status"], t4_total["reason"]) == ("rejected", "needs_human_review")


async def test_one_character_evidence_never_accepts(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    _page(
        stack.work / "curta.png",
        ["Pedido de exames", "1. Hemograma completo", "Retorno em 30 dias"],
    )

    run = await run_agent(stack, monkeypatch, "curta.png", fixture="evidencia_curta.json")

    assert _accepted(run.report) == ["LAB-10001"]

from pathlib import Path

import pytest
from google.adk.models import LlmRequest
from google.genai import types

from agent_runtime.errors import ConfigurationError
from agent_runtime.models import FakeLlm, resolve_model
from agent_runtime.schemas import ExtractedExams

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[2] / "fixtures" / "fake_llm"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AGENT_LLM_BACKEND", "APP_ENV", "FAKE_LLM_FIXTURE"):
        monkeypatch.delenv(name, raising=False)


def test_default_backend_passes_the_name_to_adk() -> None:
    assert resolve_model("gemini-flash-latest") == "gemini-flash-latest"


@pytest.mark.parametrize("app_env", [None, "", "production", "dev", "TEST"])
def test_fake_backend_outside_test_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, app_env: str | None
) -> None:
    monkeypatch.setenv("AGENT_LLM_BACKEND", "fake")
    if app_env is not None:
        monkeypatch.setenv("APP_ENV", app_env)

    with pytest.raises(ConfigurationError, match="FAKE_LLM_OUTSIDE_TEST"):
        resolve_model("gemini-flash-latest")


def test_unknown_backend_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LLM_BACKEND", "openai")

    with pytest.raises(ConfigurationError, match="INVALID_ENV"):
        resolve_model("x")


def _request(text: str) -> LlmRequest:
    return LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text=text)])])


async def test_fake_answers_from_the_fixture_and_records_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_LLM_BACKEND", "fake")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("FAKE_LLM_FIXTURE", str(FIXTURES / "pedido_ambiguo.json"))
    model = resolve_model("gemini-flash-latest")
    assert isinstance(model, FakeLlm)

    [response] = [r async for r in model.generate_content_async(_request("Documento"))]

    assert response.content is not None
    assert response.content.parts is not None
    answer = ExtractedExams.model_validate_json(response.content.parts[0].text or "")
    assert [exam.name for exam in answer.exams] == ["Hemograma completo", "T4"]
    assert len(model.requests) == 1
    assert model.requests[0].contents[0].parts[0].text == "Documento"  # type: ignore[index]
    assert model.model == "fake/gemini-flash-latest"


async def test_fake_refuses_to_answer_outside_test(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeLlm(fixture=str(FIXTURES / "empty.json"))

    with pytest.raises(ConfigurationError, match="FAKE_LLM_OUTSIDE_TEST"):
        [r async for r in model.generate_content_async(_request("x"))]
    assert model.requests == []


@pytest.mark.parametrize("fixture", ["missing.json", None])
async def test_fake_needs_a_readable_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixture: str | None
) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    model = FakeLlm(fixture=str(tmp_path / fixture) if fixture else None)

    with pytest.raises(ConfigurationError):
        [r async for r in model.generate_content_async(_request("x"))]


async def test_fake_validates_the_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    bad = tmp_path / "bad.json"
    bad.write_text('{"exams": [{"name": "x"}]}')

    with pytest.raises(ConfigurationError, match="INVALID_FAKE_LLM_FIXTURE"):
        [r async for r in FakeLlm(fixture=str(bad)).generate_content_async(_request("x"))]


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.name)
def test_versioned_fixtures_are_valid(path: Path) -> None:
    ExtractedExams.model_validate_json(path.read_text("utf-8"))

import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from google.adk.models import BaseLlm, LlmCapabilities, LlmRequest, LlmResponse
from google.genai import types
from pydantic import Field, ValidationError

from agent_runtime.errors import ConfigurationError
from agent_runtime.schemas import ExtractedExams

BACKEND_ENV = "AGENT_LLM_BACKEND"
APP_ENV = "APP_ENV"
FIXTURE_ENV = "FAKE_LLM_FIXTURE"
BACKENDS = ("gemini", "fake")


class FakeLlm(BaseLlm):
    """Test model: answers with the ExtractedExams of a fixture read at each call."""

    model: str = "fake"
    fixture: str | None = None
    requests: list[LlmRequest] = Field(default_factory=list)

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=False)

    def answer(self) -> ExtractedExams:
        path = self.fixture or os.environ.get(FIXTURE_ENV)
        if not path:
            raise ConfigurationError("MISSING_ENV", FIXTURE_ENV)

        try:
            return ExtractedExams.model_validate(json.loads(Path(path).read_text("utf-8")))
        except (OSError, ValueError, ValidationError):
            raise ConfigurationError("INVALID_FAKE_LLM_FIXTURE", "fixture is unreadable") from None

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        _require_test_env()
        self.requests.append(llm_request.model_copy(deep=True))
        text = self.answer().model_dump_json()
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def resolve_model(name: str) -> str | BaseLlm:
    backend = os.environ.get(BACKEND_ENV) or "gemini"
    if backend not in BACKENDS:
        raise ConfigurationError("INVALID_ENV", f"{BACKEND_ENV} must be one of {list(BACKENDS)}")

    if backend == "fake":
        _require_test_env()
        return FakeLlm(model=f"fake/{name}")

    return name


def _require_test_env() -> None:
    if os.environ.get(APP_ENV) != "test":
        raise ConfigurationError(
            "FAKE_LLM_OUTSIDE_TEST",
            f"{BACKEND_ENV}=fake requires {APP_ENV}=test",
        )

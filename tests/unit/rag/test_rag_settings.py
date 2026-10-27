from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_mcp.settings import RagSettings

pytestmark = pytest.mark.unit

TOKEN = "t" * 32


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "MCP_AUTH_TOKEN",
        "RAG_HOST",
        "RAG_PORT",
        "CATALOG_PATH",
        "RAG_MAX_BATCH",
        "LOG_LEVEL",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_reads_documented_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("RAG_PORT", "9002")
    monkeypatch.setenv("CATALOG_PATH", "/app/data/catalog.json")
    monkeypatch.setenv("RAG_MAX_BATCH", "20")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = RagSettings()  # type: ignore[call-arg]

    assert settings.port == 9002
    assert settings.catalog_path == Path("/app/data/catalog.json")
    assert settings.max_batch == 20
    assert settings.log_level == "DEBUG"
    assert settings.mcp_auth_token.get_secret_value() == TOKEN


def test_defaults() -> None:
    settings = RagSettings(mcp_auth_token=TOKEN)

    assert (settings.host, settings.port) == ("0.0.0.0", 8102)  # noqa: S104
    assert settings.catalog_path == Path("data/exams_catalog.json")
    assert settings.max_batch == 50
    assert settings.log_level == "INFO"


def test_token_is_required() -> None:
    with pytest.raises(ValidationError, match="MCP_AUTH_TOKEN"):
        RagSettings()  # type: ignore[call-arg]


def test_short_token_is_rejected_without_echoing_it() -> None:
    short = "short-token-value"
    with pytest.raises(ValidationError) as info:
        RagSettings(mcp_auth_token=short)
    assert short not in str(info.value)


def test_token_is_hidden_in_repr() -> None:
    assert TOKEN not in repr(RagSettings(mcp_auth_token=TOKEN))


@pytest.mark.parametrize("value", ["0", "51"])
def test_max_batch_is_bounded(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RAG_MAX_BATCH", value)
    with pytest.raises(ValidationError, match="RAG_MAX_BATCH"):
        RagSettings(mcp_auth_token=TOKEN)

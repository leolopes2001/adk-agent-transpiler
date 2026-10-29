"""Fixtures of the end-to-end tests; the stack itself is `e2e_stack` in tests/conftest.py."""

from collections.abc import Iterator

import pytest

from e2e.helpers import LLM_KEY_ENV, google_api_key


@pytest.fixture(scope="session")
def llm_key(monkeypatch_session: pytest.MonkeyPatch) -> str:
    """Puts the real Gemini key in this process' environment, for `run_agent(pass_env=...)`."""
    key = google_api_key()
    if not key:
        pytest.skip(f"{LLM_KEY_ENV} is not set (environment or .env)")
    monkeypatch_session.setenv(LLM_KEY_ENV, key)
    return LLM_KEY_ENV


@pytest.fixture(scope="session")
def monkeypatch_session() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as patch:
        yield patch

"""`.env.example` must work as is: `cp .env.example .env` plus the Gemini key."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SECRETS = ["POSTGRES_PASSWORD", "SCHEDULER_API_KEY", "MCP_AUTH_TOKEN", "PII_HMAC_SECRET"]
MIN_SECRET_LENGTH = 32
"""The API and the MCP servers refuse shorter keys."""


def _values() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            values[name] = value
    return values


@pytest.mark.parametrize("name", SECRETS)
def test_secrets_have_valid_local_values(name: str) -> None:
    value = _values()[name]

    assert len(value) >= MIN_SECRET_LENGTH
    # POSTGRES_PASSWORD goes inside DATABASE_URL; the others share the rule for simplicity.
    assert re.fullmatch(r"[A-Za-z0-9_-]+", value)


def test_gemini_key_is_left_for_the_user() -> None:
    assert _values()["GOOGLE_API_KEY"] == ""


def test_declares_every_variable_compose_requires() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()
    required = set(re.findall(r"\$\{([A-Z_]+):\?", compose))

    assert required
    assert required <= _values().keys()

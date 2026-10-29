import pytest
from pydantic import ValidationError

from ocr_mcp.settings import OcrSettings

pytestmark = pytest.mark.unit

TOKEN = "t" * 32
SECRET = "s" * 32


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["MCP_AUTH_TOKEN", "PII_HMAC_SECRET", "OCR_PORT", "OCR_HOST", "LOG_LEVEL"]:
        monkeypatch.delenv(name, raising=False)


def test_reads_documented_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("PII_HMAC_SECRET", SECRET)
    monkeypatch.setenv("OCR_PORT", "9000")
    monkeypatch.setenv("OCR_MAX_CONCURRENCY", "4")

    settings = OcrSettings()

    assert settings.port == 9000
    assert settings.max_concurrency == 4
    assert settings.mcp_auth_token.get_secret_value() == TOKEN


def test_defaults() -> None:
    settings = OcrSettings(mcp_auth_token=TOKEN, pii_hmac_secret=SECRET)

    assert (settings.host, settings.port) == ("0.0.0.0", 8101)  # noqa: S104
    assert settings.max_image_bytes == 8 * 1024 * 1024
    assert settings.max_pixels == 40_000_000
    assert settings.timeout_s == 30
    assert settings.max_concurrency == 2
    assert settings.log_level == "INFO"


@pytest.mark.parametrize("missing", ["mcp_auth_token", "pii_hmac_secret"])
def test_secrets_are_required(missing: str) -> None:
    values = {"mcp_auth_token": TOKEN, "pii_hmac_secret": SECRET}
    del values[missing]
    with pytest.raises(ValidationError, match=f"(?i){missing}"):
        OcrSettings(**values)


@pytest.mark.parametrize("field", ["mcp_auth_token", "pii_hmac_secret"])
def test_short_secrets_are_rejected(field: str) -> None:
    short = "short-secret-" + "x" * 18
    values = {"mcp_auth_token": TOKEN, "pii_hmac_secret": SECRET, field: short}
    with pytest.raises(ValidationError, match=field) as excinfo:
        OcrSettings(**values)
    assert short not in str(excinfo.value)


def test_secrets_are_hidden_in_repr() -> None:
    settings = OcrSettings(mcp_auth_token=TOKEN, pii_hmac_secret=SECRET)
    assert TOKEN not in repr(settings)
    assert SECRET not in repr(settings)

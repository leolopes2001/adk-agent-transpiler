"""Runtime configuration, read from environment variables.

Variable names follow the deployment docs (`OCR_*`, plus the shared
`MCP_AUTH_TOKEN`, `PII_HMAC_SECRET` and `LOG_LEVEL`); secrets are `SecretStr`
so they never show up in a repr or a log line.
"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

MIN_SECRET_LENGTH = 32


class OcrSettings(BaseSettings):
    # hide_input_in_errors: a too-short secret must not be echoed in the
    # validation error that aborts startup.
    model_config = SettingsConfigDict(
        frozen=True, extra="ignore", populate_by_name=True, hide_input_in_errors=True
    )

    host: str = Field(default="0.0.0.0", validation_alias="OCR_HOST")  # noqa: S104 - container bind
    port: int = Field(default=8101, ge=1, le=65535, validation_alias="OCR_PORT")
    mcp_auth_token: SecretStr = Field(
        min_length=MIN_SECRET_LENGTH, validation_alias="MCP_AUTH_TOKEN"
    )
    pii_hmac_secret: SecretStr = Field(
        min_length=MIN_SECRET_LENGTH, validation_alias="PII_HMAC_SECRET"
    )
    max_image_bytes: int = Field(
        default=8 * 1024 * 1024, ge=1, le=64 * 1024 * 1024, validation_alias="OCR_MAX_IMAGE_BYTES"
    )
    max_pixels: int = Field(
        default=40_000_000, ge=1, le=80_000_000, validation_alias="OCR_MAX_PIXELS"
    )
    timeout_s: float = Field(default=30, gt=0, le=300, validation_alias="OCR_TIMEOUT_S")
    max_concurrency: int = Field(default=2, ge=1, le=32, validation_alias="OCR_MAX_CONCURRENCY")
    log_level: LogLevel = Field(default="INFO", validation_alias="LOG_LEVEL")

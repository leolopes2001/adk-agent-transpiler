"""Runtime configuration, read from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Each field maps to the upper-case environment variable of the same name.

    Secrets are `SecretStr` so they never show up in a repr or a log line.
    """

    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    database_url: SecretStr
    scheduler_api_key: SecretStr = Field(min_length=32)
    catalog_path: Path
    idempotency_ttl_hours: int = Field(default=24, ge=1, le=24 * 30)
    log_level: LogLevel = "INFO"

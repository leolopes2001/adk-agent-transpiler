from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from lab_common.contracts.rag import MAX_BATCH_QUERIES

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

MIN_SECRET_LENGTH = 32


class RagSettings(BaseSettings):
    # A too-short token must not be echoed in the validation error that aborts startup.
    model_config = SettingsConfigDict(
        frozen=True, extra="ignore", populate_by_name=True, hide_input_in_errors=True
    )

    host: str = Field(default="0.0.0.0", validation_alias="RAG_HOST")  # noqa: S104 - container bind
    port: int = Field(default=8102, ge=1, le=65535, validation_alias="RAG_PORT")
    catalog_path: Path = Field(
        default=Path("data/exams_catalog.json"), validation_alias="CATALOG_PATH"
    )
    mcp_auth_token: SecretStr = Field(
        min_length=MIN_SECRET_LENGTH, validation_alias="MCP_AUTH_TOKEN"
    )
    max_batch: int = Field(
        default=MAX_BATCH_QUERIES, ge=1, le=MAX_BATCH_QUERIES, validation_alias="RAG_MAX_BATCH"
    )
    log_level: LogLevel = Field(default="INFO", validation_alias="LOG_LEVEL")

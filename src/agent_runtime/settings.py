import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_runtime.errors import ConfigurationError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AgentSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_root: Path
    generated_root: Path
    log_level: LogLevel | None = None

    @classmethod
    def from_env(cls) -> "AgentSettings":
        cwd = Path.cwd()
        try:
            return cls(
                input_root=Path(os.environ.get("INPUT_ROOT") or cwd).resolve(),
                generated_root=Path(
                    os.environ.get("GENERATED_ROOT") or cwd / "generated"
                ).resolve(),
                log_level=(os.environ.get("LOG_LEVEL") or "").upper() or None,  # type: ignore[arg-type]
            )
        except ValidationError:
            raise ConfigurationError("INVALID_ENV", "LOG_LEVEL") from None


def missing(names: list[str]) -> list[str]:
    return [name for name in names if not os.environ.get(name)]

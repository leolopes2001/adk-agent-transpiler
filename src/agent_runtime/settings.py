"""Configuration of the `lab-agent` CLI, read from the environment.

Only what the CLI itself needs lives here. The servers and services a
workflow uses declare their own variables in the generated code
(`RuntimeDeps.required_env()`), and the CLI checks them before running, so a
missing variable is reported once, with every name, instead of failing in
the middle of a run.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_runtime.errors import ConfigurationError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AgentSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_root: Path
    """INPUT_ROOT: the run's image must be under it (default: working directory)."""
    generated_root: Path
    """GENERATED_ROOT: the generated package must be under it (default: ./generated)."""
    log_level: LogLevel | None = None
    """LOG_LEVEL: unset leaves the choice to the CLI (quiet unless --verbose)."""

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
    """Names in `names` that are unset or empty."""
    return [name for name in names if not os.environ.get(name)]

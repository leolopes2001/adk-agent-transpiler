from collections.abc import Mapping
from typing import Any


# ADK stores and logs str(exc) of a failed node: messages hold codes, never document text.
class AgentRuntimeError(Exception):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        self.message = message or code
        super().__init__(f"{code}: {self.message}" if message else code)


class TransientError(AgentRuntimeError):
    pass


class PermanentError(AgentRuntimeError):
    pass


class InvalidInputError(PermanentError):
    pass


class ConfigurationError(AgentRuntimeError):
    pass


class ToolError(AgentRuntimeError):
    tool_code: str
    retryable: bool


class TransientToolError(ToolError, TransientError):
    pass


class PermanentToolError(ToolError, PermanentError):
    pass


def tool_error(server: str, tool: str, body: Mapping[str, Any]) -> ToolError:
    raw_code = body.get("code")
    tool_code = raw_code if isinstance(raw_code, str) and raw_code.isidentifier() else "UNKNOWN"
    retryable = body.get("retryable") is True
    cls = TransientToolError if retryable else PermanentToolError
    # Only the code: the server contracts fix the code set, its free-text message is not ours.
    error = cls("MCP_TOOL_ERROR", f"{server}.{tool} returned {tool_code}")
    error.tool_code = tool_code
    error.retryable = retryable
    return error


def for_http_status(status: int, code: str) -> AgentRuntimeError:
    if status >= 500 or status in (408, 429):  # noqa: PLR2004 - HTTP status classes
        return TransientError(code, f"HTTP {status}")
    return PermanentError(code, f"HTTP {status}")

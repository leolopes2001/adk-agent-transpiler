"""Error taxonomy of the runtime.

There is one retry level: the node's `RetryConfig`, which the factories limit
to `TransientError` (and node timeouts). Gateways and HTTP clients never retry
by themselves; they classify failures:

- `TransientError`: network, timeout, 5xx, a tool error marked retryable.
  Raised so the node retries.
- `PermanentError`: 4xx, broken contract, a tool error not marked retryable.
  Nodes turn it into a typed output when their kind can express it (schedule
  -> `failed`); otherwise it ends the run (CLI exit 6).

ADK stores `str(exc)` of a failed node in the session and logs its traceback,
so messages hold codes and names chosen here, never document text, tool
payloads or the text of an underlying exception.
"""

from collections.abc import Mapping
from typing import Any


class AgentRuntimeError(Exception):
    """Base class; `code` is a stable, machine-readable identifier."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        self.message = message or code
        super().__init__(f"{code}: {self.message}" if message else code)


class TransientError(AgentRuntimeError):
    """The same call may succeed later; the node's retry policy handles it."""


class PermanentError(AgentRuntimeError):
    """Retrying the same call cannot succeed."""


class InvalidInputError(PermanentError):
    """The run input (image path, format, size) is unacceptable (CLI exit 2)."""


class ConfigurationError(AgentRuntimeError):
    """Missing or invalid configuration (CLI exit 8)."""


class ToolError(AgentRuntimeError):
    """An MCP tool answered with `isError: true` and a `{code, message, retryable}` body."""

    tool_code: str
    retryable: bool


class TransientToolError(ToolError, TransientError):
    pass


class PermanentToolError(ToolError, PermanentError):
    pass


def tool_error(server: str, tool: str, body: Mapping[str, Any]) -> ToolError:
    """Error for a tool result body; a body without a usable code is permanent.

    Only the tool's error code goes into the message: the code set is fixed by
    the server contracts, while the free-text message is not ours to vouch for.
    """
    raw_code = body.get("code")
    tool_code = raw_code if isinstance(raw_code, str) and raw_code.isidentifier() else "UNKNOWN"
    retryable = body.get("retryable") is True
    cls = TransientToolError if retryable else PermanentToolError
    error = cls("MCP_TOOL_ERROR", f"{server}.{tool} returned {tool_code}")
    error.tool_code = tool_code
    error.retryable = retryable
    return error


def for_http_status(status: int, code: str) -> AgentRuntimeError:
    """5xx (and 408/429) are transient; any other error status is permanent."""
    if status >= 500 or status in (408, 429):  # noqa: PLR2004 - HTTP status classes
        return TransientError(code, f"HTTP {status}")
    return PermanentError(code, f"HTTP {status}")

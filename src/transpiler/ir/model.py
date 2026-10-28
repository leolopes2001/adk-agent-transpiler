from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from transpiler.registry import NodeKind

type ParamValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class RetryIR:
    max_attempts: int
    initial_delay_s: float
    max_delay_s: float


@dataclass(frozen=True, slots=True)
class McpServerIR:
    key: str
    url_env: str
    auth_token_env: str
    allowed_tools: tuple[str, ...]
    connect_timeout_s: float
    read_timeout_s: float


@dataclass(frozen=True, slots=True)
class HttpServiceIR:
    key: str
    base_url_env: str
    api_key_env: str
    timeout_s: float


@dataclass(frozen=True, slots=True)
class ModelIR:
    key: str
    provider: str
    name: str
    temperature: float
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class LlmAgentIR:
    key: str
    model: ModelIR
    instruction: str
    instruction_sha256: str
    output_schema: str


@dataclass(frozen=True, slots=True)
class NodeIR:
    id: str
    kind: NodeKind
    params: Mapping[str, ParamValue]
    retry: RetryIR | None
    """None unless the kind accepts retry; policy defaults are already applied."""
    timeout_s: float | None
    server: McpServerIR | None = None
    tool: str | None = None
    service: HttpServiceIR | None = None
    agent: LlmAgentIR | None = None
    subgraph: GraphIR | None = None


@dataclass(frozen=True, slots=True)
class EdgeIR:
    source: str
    target: str
    route: str | None


@dataclass(frozen=True, slots=True)
class GraphIR:
    name: str
    nodes: tuple[NodeIR, ...]
    edges: tuple[EdgeIR, ...]

    def node(self, node_id: str) -> NodeIR:
        return next(node for node in self.nodes if node.id == node_id)


@dataclass(frozen=True, slots=True)
class ProgramIR:
    agent_name: str
    description: str
    version: str
    root: GraphIR
    servers: tuple[McpServerIR, ...]
    services: tuple[HttpServiceIR, ...]
    needs_resumability: bool
    """True when a node may pause for human input (human_review)."""
    spec_sha256: str

    def digest(self) -> str:
        """Leaves out spec_sha256: reformatting the spec must not move the digest."""
        canonical = json.dumps(
            _plain(self, skip=frozenset({"spec_sha256"})),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def _plain(value: Any, skip: frozenset[str] = frozenset()) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value) if f.name not in skip}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value

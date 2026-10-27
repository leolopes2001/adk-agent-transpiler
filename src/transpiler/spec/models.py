"""Pydantic models for agent spec v1.0.

The models are the schema pass of the transpiler: anything that can be decided
by looking at one value in isolation (shape, ranges, identifier syntax) is
rejected here, so later passes can assume well-formed input. Cross-references
and graph rules belong to the semantic pass.

Identifiers end up as Python variable names in generated code, which is why
they are restricted to a lowercase regex and cannot shadow keywords, builtins
or names the template defines.
"""

import builtins
import keyword
import re
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from lab_common.contracts.rag import MAX_TOP_K

IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_]{0,62}$"
ENV_VAR_PATTERN = r"^[A-Z][A-Z0-9_]{1,63}$"
NODE_REF_PATTERN = r"^(START|[a-z][a-z0-9_]{0,62})$"
ROUTE_PATTERN = r"^[a-z][a-z0-9_]{0,62}$"

START = "START"

# Names the generated module binds at top level; a node called `deps` would
# silently replace the runtime dependencies object.
TEMPLATE_NAMES = frozenset(
    {
        "start",
        "root_agent",
        "app",
        "deps",
        "workflow",
        "agent",
        "nodes",
        "schemas",
        "guardrails",
        "types",
        "resolve_model",
        "annotations",
    }
)
RESERVED_IDENTIFIERS = TEMPLATE_NAMES | frozenset(
    name for name in dir(builtins) if name.islower() and not name.startswith("_")
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _not_reserved(value: str) -> str:
    if keyword.iskeyword(value) or keyword.issoftkeyword(value) or value in RESERVED_IDENTIFIERS:
        raise PydanticCustomError(
            "reserved_identifier",
            '"{value}" é uma palavra reservada e não pode ser usado como identificador',
            {"value": value},
        )
    return value


Identifier = Annotated[str, Field(pattern=IDENTIFIER_PATTERN), AfterValidator(_not_reserved)]
EnvVarName = Annotated[str, Field(pattern=ENV_VAR_PATTERN)]
NodeRef = Annotated[str, Field(pattern=NODE_REF_PATTERN)]
RouteName = Annotated[str, Field(pattern=ROUTE_PATTERN)]
Threshold = Annotated[float, Field(ge=0, le=1)]
TimeoutSeconds = Annotated[float, Field(gt=0, le=300)]


class _Model(BaseModel):
    # strict: JSON strings are never coerced into numbers or booleans.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AgentInfo(_Model):
    name: Identifier
    description: Annotated[str, Field(min_length=1, max_length=500)]
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]


class ModelSpec(_Model):
    provider: Literal["gemini"]
    name: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9.\-]{0,62}$")]
    temperature: Annotated[float, Field(ge=0, le=2)]
    max_output_tokens: Annotated[int, Field(ge=1, le=65536)]


def _unique(values: list[str]) -> list[str]:
    if len(set(values)) != len(values):
        raise PydanticCustomError("duplicate_item", "a lista contém itens repetidos")
    return values


class McpServerSpec(_Model):
    transport: Literal["sse"]
    url_env: EnvVarName
    auth_token_env: EnvVarName
    allowed_tools: Annotated[
        list[Identifier], Field(min_length=1, max_length=32), AfterValidator(_unique)
    ]
    connect_timeout_s: TimeoutSeconds
    read_timeout_s: TimeoutSeconds


class HttpServiceSpec(_Model):
    base_url_env: EnvVarName
    api_key_env: EnvVarName
    timeout_s: TimeoutSeconds


class LlmAgentSpec(_Model):
    model: Identifier
    instruction_file: Annotated[str, Field(min_length=1, max_length=255, pattern=r"^[^\x00\\]+$")]
    output_schema: Literal["ExtractedExams"]
    tools: Annotated[list[Identifier], AfterValidator(_unique)] = []
    input_trust: Literal["untrusted", "trusted"] = "untrusted"


class RetrySpec(_Model):
    max_attempts: Annotated[int, Field(ge=1, le=5)]
    initial_delay_s: Annotated[float, Field(ge=0.1, le=10)]
    max_delay_s: Annotated[float, Field(gt=0, le=30)]

    @model_validator(mode="after")
    def _delays_ordered(self) -> Self:
        if self.max_delay_s < self.initial_delay_s:
            raise PydanticCustomError(
                "retry_delay_order", "max_delay_s deve ser maior ou igual a initial_delay_s"
            )
        return self


# --- node params (defaults here are the defaults of the node kind) ---


class OcrExtractParams(_Model):
    min_mean_confidence: Threshold = 0.55
    max_image_bytes: Annotated[int, Field(ge=1, le=MAX_IMAGE_BYTES)] = MAX_IMAGE_BYTES


class UntrustedTextGuardParams(_Model):
    max_chars: Annotated[int, Field(ge=100, le=100_000)] = 8000
    block_at: Literal["high", "medium"] = "high"


class LexicalExtractParams(_Model):
    min_score: Threshold = 0.8
    max_lines: Annotated[int, Field(ge=1, le=500)] = 60


class ReconcileParams(_Model):
    grounding_min_ratio: Annotated[int, Field(ge=0, le=100)] = 90
    max_items: Annotated[int, Field(ge=1, le=100)] = 30


class RagResolveParams(_Model):
    top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = 5
    """Capped by the RAG server's contract, which rejects larger values."""


class ConfidenceGateParams(_Model):
    accept_threshold: Threshold
    review_threshold: Threshold
    min_margin: Annotated[float, Field(ge=0, le=0.5)]

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> Self:
        if self.review_threshold > self.accept_threshold:
            raise PydanticCustomError(
                "threshold_order", "review_threshold deve ser menor ou igual a accept_threshold"
            )
        return self


class HumanReviewParams(_Model):
    on_non_interactive: Literal["reject"] = "reject"
    max_candidates: Annotated[int, Field(ge=1, le=10)] = 3


class PresentParams(_Model):
    format: Literal["table", "json"] = "table"


# --- nodes ---


class _Node(_Model):
    id: Identifier


class _IoNode(_Node):
    """Nodes that call a server, a service or a model; only these retry or time out."""

    retry: RetrySpec | None = None
    timeout_s: TimeoutSeconds | None = None


class OcrExtractNode(_IoNode):
    kind: Literal["ocr_extract"]
    server: Identifier
    tool: Identifier
    params: OcrExtractParams = Field(default_factory=OcrExtractParams)


class UntrustedTextGuardNode(_Node):
    kind: Literal["untrusted_text_guard"]
    params: UntrustedTextGuardParams = Field(default_factory=UntrustedTextGuardParams)


class LlmExtractNode(_IoNode):
    kind: Literal["llm_extract"]
    agent: Identifier


class LexicalExtractNode(_IoNode):
    kind: Literal["lexical_extract"]
    server: Identifier
    tool: Identifier
    params: LexicalExtractParams = Field(default_factory=LexicalExtractParams)


class JoinNodeSpec(_Node):
    kind: Literal["join"]


class ReconcileNode(_Node):
    kind: Literal["reconcile"]
    params: ReconcileParams = Field(default_factory=ReconcileParams)


class RagResolveNode(_IoNode):
    kind: Literal["rag_resolve"]
    server: Identifier
    tool: Identifier
    params: RagResolveParams = Field(default_factory=RagResolveParams)


class ConfidenceGateNode(_Node):
    kind: Literal["confidence_gate"]
    params: ConfidenceGateParams


class HumanReviewNode(_Node):
    kind: Literal["human_review"]
    params: HumanReviewParams = Field(default_factory=HumanReviewParams)


class ScheduleNode(_IoNode):
    kind: Literal["schedule"]
    service: Identifier


class PresentNode(_Node):
    kind: Literal["present"]
    params: PresentParams = Field(default_factory=PresentParams)


class AbortNode(_Node):
    kind: Literal["abort"]


class SubworkflowNode(_Node):
    kind: Literal["subworkflow"]
    ref: Identifier


NodeSpec = Annotated[
    OcrExtractNode
    | UntrustedTextGuardNode
    | LlmExtractNode
    | LexicalExtractNode
    | JoinNodeSpec
    | ReconcileNode
    | RagResolveNode
    | ConfidenceGateNode
    | HumanReviewNode
    | ScheduleNode
    | PresentNode
    | AbortNode
    | SubworkflowNode,
    Field(discriminator="kind"),
]


class EdgeSpec(_Model):
    source: NodeRef = Field(alias="from")
    target: NodeRef = Field(alias="to")
    route: RouteName | None = None


class GraphSpec(_Model):
    nodes: Annotated[list[NodeSpec], Field(min_length=1, max_length=200)]
    edges: Annotated[list[EdgeSpec], Field(max_length=1000)]


class PiiPolicy(_Model):
    fail_closed: bool


def _default_retry() -> RetrySpec:
    return RetrySpec(max_attempts=1, initial_delay_s=0.5, max_delay_s=2)


class NodeDefaults(_Model):
    retry: RetrySpec = Field(default_factory=_default_retry)
    timeout_s: TimeoutSeconds = 30


class Policies(_Model):
    pii: PiiPolicy
    defaults: NodeDefaults = Field(default_factory=NodeDefaults)


class AgentSpec(_Model):
    spec_version: Literal["1.0"]
    agent: AgentInfo
    models: Annotated[dict[Identifier, ModelSpec], Field(max_length=32)] = {}
    mcp_servers: Annotated[dict[Identifier, McpServerSpec], Field(max_length=32)] = {}
    http_services: Annotated[dict[Identifier, HttpServiceSpec], Field(max_length=32)] = {}
    llm_agents: Annotated[dict[Identifier, LlmAgentSpec], Field(max_length=32)] = {}
    workflow: GraphSpec
    subworkflows: Annotated[dict[Identifier, GraphSpec], Field(max_length=32)] = {}
    policies: Policies


# --- inline secrets ---

_SECRET_KEY = re.compile(r"(?:^|_)(?:secret|token|password|api_key)(?:_|$)")
# Keys of these maps are names chosen by the spec author, not field names.
_NAMED_MAPS = frozenset({"models", "mcp_servers", "http_services", "llm_agents", "subworkflows"})

type JsonPath = tuple[str | int, ...]


def is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY.search(key)) and not key.endswith("_env")


def find_inline_secrets(raw: object) -> list[JsonPath]:
    """Paths of object keys that look like a secret given inline instead of by env var name.

    Runs on the raw JSON before model validation: those keys are not fields, so
    validation alone would report a generic "unknown field" and hide the real
    problem.
    """
    found: list[JsonPath] = []

    def walk(value: object, path: JsonPath, keys_are_names: bool) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*path, key)
                if not keys_are_names and is_secret_key(key):
                    found.append(child_path)
                walk(child, child_path, keys_are_names=not path and key in _NAMED_MAPS)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, index), keys_are_names=False)

    walk(raw, (), keys_are_names=False)
    return found

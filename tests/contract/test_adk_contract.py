"""Contract with the installed google-adk: the behaviors the runtime relies on.

Each test pins one assumption of the runtime design (ADR-0009). When an ADK
upgrade breaks one of them, the failing test names the assumption, and the
fallback recorded in the ADR says what to change.
"""

import inspect
import json
from collections.abc import AsyncGenerator
from typing import Any

import google.adk
import pytest
from google.adk import Agent, Context, Event, Runner, Workflow
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.request_input import RequestInput
from google.adk.models import BaseLlm, LlmCapabilities, LlmRequest, LlmResponse
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_tool import McpTool
from google.adk.workflow import FunctionNode, JoinNode, NodeTimeoutError, RetryConfig
from google.genai import _extra_utils, types
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel, Field

pytestmark = pytest.mark.contract

ADK_VERSION = "2.13.0"
USER = "contract"


class Payload(BaseModel):
    value: int


class Mentions(BaseModel):
    exams: list[str]


class ScriptedLlm(BaseLlm):
    """Answers every request with the same text and keeps the requests."""

    model: str = "scripted"
    answer: str = '{"exams": ["Hemograma completo"]}'
    requests: list[LlmRequest] = Field(default_factory=list)

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.answer)]))


class Retryable(Exception):
    """Stands for the runtime's TransientError."""


class Final(Exception):
    """Stands for the runtime's PermanentError."""


def _node(name: str, func: Any, **kwargs: Any) -> FunctionNode:
    return FunctionNode(func=func, name=name, **kwargs)


async def _emit(ctx: Context, node_input: Any) -> Event:
    return Event(output=Payload(value=1))


async def _run(
    root: Workflow,
    *,
    state: dict[str, Any] | None = None,
    message: str = "{}",
    resumable: bool = False,
) -> tuple[list[Event], Session, Runner]:
    service = InMemorySessionService()
    app = App(
        name="contract",
        root_agent=root,
        resumability_config=ResumabilityConfig(is_resumable=True) if resumable else None,
    )
    runner = Runner(app=app, session_service=service)
    session = await service.create_session(app_name="contract", user_id=USER, state=state)
    content = types.Content(role="user", parts=[types.Part(text=message)])
    events = [
        event
        async for event in runner.run_async(
            user_id=USER, session_id=session.id, new_message=content
        )
    ]
    final = await service.get_session(app_name="contract", user_id=USER, session_id=session.id)
    assert final is not None
    return events, final, runner


def _outputs(events: list[Event]) -> dict[str, Any]:
    """Last output of each node, keyed by node name."""
    return {e.node_info.name: e.output for e in events if e.output is not None}


def test_installed_version_is_the_one_the_spike_checked() -> None:
    assert google.adk.__version__ == ADK_VERSION


def test_function_node_takes_retry_timeout_and_rerun_on_resume() -> None:
    retry = RetryConfig(max_attempts=3, initial_delay=0.5, max_delay=4.0)

    node = FunctionNode(
        func=_emit, name="ocr", retry_config=retry, timeout=45.0, rerun_on_resume=True
    )

    assert (node.name, node.timeout, node.rerun_on_resume) == ("ocr", 45.0, True)
    assert node.retry_config is retry


async def test_first_node_receives_the_user_message_as_content() -> None:
    seen: list[Any] = []

    async def first(ctx: Context, node_input: Any) -> Event:
        seen.append(node_input)
        return Event(output={"ok": True})

    await _run(Workflow(name="w", edges=[("START", _node("first", first))]), message='{"a": 1}')

    assert isinstance(seen[0], types.Content)
    assert seen[0].parts is not None
    assert json.loads(seen[0].parts[0].text or "") == {"a": 1}


async def test_event_output_model_reaches_the_next_node_and_serializes() -> None:
    seen: list[Any] = []

    async def second(ctx: Context, node_input: Any) -> Event:
        seen.append(node_input)
        return Event(output={"done": True})

    first, nxt = _node("first", _emit), _node("second", second)
    _, session, _ = await _run(Workflow(name="w", edges=[("START", first), (first, nxt)]))

    # In memory the model arrives as is; nodes must accept a model or its dump.
    assert Payload.model_validate(seen[0]) == Payload(value=1)
    assert '"value":1' in session.model_dump_json()


async def test_routes_select_the_edge_and_two_routes_may_share_a_target() -> None:
    async def router(ctx: Context, node_input: Any) -> Event:
        return Event(output={"r": "b"}, route="b")

    async def sink(ctx: Context, node_input: Any) -> Event:
        return Event(output={"sink": node_input})

    route, target, other = _node("router", router), _node("target", sink), _node("other", sink)
    root = Workflow(
        name="w",
        edges=[("START", route), (route, {"a": target, "b": target, "c": other})],
    )

    events, _, _ = await _run(root)

    assert _outputs(events)["target"] == {"sink": {"r": "b"}}
    assert "other" not in _outputs(events)


async def test_state_writes_become_session_state() -> None:
    async def write(ctx: Context, node_input: Any) -> Event:
        ctx.state["run.flag"] = ctx.state.get("run.interactive")
        return Event(output={"ok": True})

    _, session, _ = await _run(
        Workflow(name="w", edges=[("START", _node("write", write))]),
        state={"run.interactive": False},
    )

    assert session.state["run.flag"] is False


async def test_join_keys_inputs_by_predecessor_name_including_an_agent() -> None:
    seen: list[Any] = []

    async def to_text(ctx: Context, node_input: Any) -> Event:
        return Event(output="Documento: HMG")

    async def lexical(ctx: Context, node_input: Any) -> Event:
        return Event(output=Payload(value=3))

    async def merge_reader(ctx: Context, node_input: Any) -> Event:
        seen.append(node_input)
        return Event(output={"ok": True})

    model = ScriptedLlm()
    agent = Agent(
        name="extractor", model=model, mode="single_turn", instruction="i", output_schema=Mentions
    )
    text, lex, read = _node("text", to_text), _node("lex", lexical), _node("read", merge_reader)
    merge = JoinNode(name="merge")
    root = Workflow(
        name="w",
        edges=[
            ("START", text),
            (text, agent),
            ("START", lex),
            (agent, merge),
            (lex, merge),
            (merge, read),
        ],
    )

    await _run(root)

    joined = seen[0]
    assert set(joined) == {"extractor", "lex"}
    # The agent's output arrives as a dict validated against output_schema.
    assert joined["extractor"] == {"exams": ["Hemograma completo"]}
    assert Payload.model_validate(joined["lex"]).value == 3


async def test_nested_workflow_output_is_its_terminal_node_output() -> None:
    seen: list[Any] = []

    async def inner(ctx: Context, node_input: Any) -> Event:
        return Event(output={"inner": True})

    async def after(ctx: Context, node_input: Any) -> Event:
        seen.append(node_input)
        return Event(output={"after": True})

    sub = Workflow(name="sub", edges=[("START", _node("inner", inner))])
    nxt = _node("after", after)

    await _run(Workflow(name="w", edges=[("START", sub), (sub, nxt)]))

    assert seen == [{"inner": True}]


async def test_single_turn_agent_sends_only_the_node_input_and_instruction() -> None:
    async def to_text(ctx: Context, node_input: Any) -> Event:
        return Event(output="Documento: HMG")

    calls: list[dict[str, Any]] = []

    def callback(*, callback_context: Context, llm_request: LlmRequest) -> None:
        calls.append({"context": callback_context, "request": llm_request})

    model = ScriptedLlm()
    agent = Agent(
        name="extractor",
        model=model,
        mode="single_turn",
        instruction="Extraia exames.",
        output_schema=Mentions,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=2)),
        ),
        before_model_callback=callback,
        timeout=10.0,
    )
    text = _node("text", to_text)

    events, _, _ = await _run(Workflow(name="w", edges=[("START", text), (text, agent)]))

    assert _outputs(events)["extractor"] == {"exams": ["Hemograma completo"]}
    request = model.requests[0]
    assert [c.role for c in request.contents] == ["user"]
    assert request.contents[0].parts is not None
    assert request.contents[0].parts[0].text == "Documento: HMG"
    assert request.config is not None
    assert request.config.system_instruction == "Extraia exames."
    assert request.config.temperature == 0.0
    # HttpRetryOptions travel with the request; the Gemini client applies them.
    assert request.config.http_options is not None
    assert request.config.http_options.retry_options is not None
    assert request.config.http_options.retry_options.attempts == 2
    # AFC off travels with the request, and the Gemini client then skips its
    # automatic function calling loop without a warning.
    assert request.config.automatic_function_calling is not None
    assert request.config.automatic_function_calling.disable is True
    assert _extra_utils.should_disable_afc(request.config) is True
    # The callback gets keyword arguments and the request the model would get.
    assert calls[0]["request"] is request


async def test_callback_response_skips_the_model_but_does_not_fail_the_agent() -> None:
    def block(*, callback_context: Context, llm_request: LlmRequest) -> LlmResponse:
        return LlmResponse(error_code="BLOCKED", error_message="blocked")

    async def to_text(ctx: Context, node_input: Any) -> Event:
        return Event(output="Documento")

    async def after(ctx: Context, node_input: Any) -> Event:
        return Event(output={"after": node_input})

    model = ScriptedLlm()
    agent = Agent(
        name="extractor",
        model=model,
        mode="single_turn",
        instruction="i",
        output_schema=Mentions,
        before_model_callback=block,
    )
    text, nxt = _node("text", to_text), _node("after", after)

    events, _, _ = await _run(
        Workflow(name="w", edges=[("START", text), (text, agent), (agent, nxt)])
    )

    assert model.requests == []
    assert any(e.error_code == "BLOCKED" for e in events)
    # The run goes on with no agent output: blocking must be enforced downstream.
    assert _outputs(events)["after"] == {"after": None}


async def test_exception_in_a_node_fails_the_run() -> None:
    async def boom(ctx: Context, node_input: Any) -> Event:
        raise Final("PII_TRIPWIRE")

    with pytest.raises(Final, match="PII_TRIPWIRE"):
        await _run(Workflow(name="w", edges=[("START", _node("boom", boom))]))


async def test_retry_config_retries_only_the_listed_exceptions() -> None:
    attempts = {"transient": 0, "permanent": 0}

    async def transient(ctx: Context, node_input: Any) -> Event:
        attempts["transient"] += 1
        if attempts["transient"] < 3:
            raise Retryable("try again")
        return Event(output={"attempt": ctx.attempt_count})

    async def permanent(ctx: Context, node_input: Any) -> Event:
        attempts["permanent"] += 1
        raise Final("do not retry")

    retry = RetryConfig(
        max_attempts=3,
        initial_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
        exceptions=[Retryable, NodeTimeoutError],
    )
    ok = _node("transient", transient, retry_config=retry)
    events, _, _ = await _run(Workflow(name="w", edges=[("START", ok)]))
    assert _outputs(events)["transient"] == {"attempt": 3}
    # Each failed attempt leaves an error event named after the exception class.
    assert [e.error_code for e in events if e.error_code] == ["Retryable", "Retryable"]

    failing = _node("permanent", permanent, retry_config=retry)
    with pytest.raises(Final):
        await _run(Workflow(name="w", edges=[("START", failing)]))
    assert attempts["permanent"] == 1


async def test_node_timeout_raises_node_timeout_error() -> None:
    async def slow(ctx: Context, node_input: Any) -> Event:
        import asyncio  # noqa: PLC0415 - only this test sleeps

        await asyncio.sleep(1)
        return Event(output={})

    with pytest.raises(NodeTimeoutError):
        await _run(Workflow(name="w", edges=[("START", _node("slow", slow, timeout=0.05))]))


async def test_request_input_interrupts_and_resume_reruns_the_node() -> None:
    runs: list[dict[str, Any]] = []

    async def review(ctx: Context, node_input: Any) -> Event | RequestInput:
        runs.append(dict(ctx.resume_inputs))
        if "review-1" in ctx.resume_inputs:
            return Event(output=ctx.resume_inputs["review-1"], route="confirmed")
        return RequestInput(interrupt_id="review-1", message="?", payload={"options": ["a"]})

    async def done(ctx: Context, node_input: Any) -> Event:
        return Event(output={"done": node_input})

    first = _node("first", _emit)
    ask = _node("review", review, rerun_on_resume=True)
    end = _node("done", done)
    root = Workflow(
        name="w", edges=[("START", first), (first, ask), (ask, {"confirmed": end, "rejected": end})]
    )

    events, session, runner = await _run(root, resumable=True)

    interrupt = next(e for e in events if e.long_running_tool_ids)
    assert interrupt.long_running_tool_ids == {"review-1"}
    assert interrupt.content is not None
    assert interrupt.content.parts is not None
    call = interrupt.content.parts[0].function_call
    assert call is not None
    assert (call.name, call.id) == ("adk_request_input", "review-1")
    assert call.args is not None
    assert call.args["payload"] == {"options": ["a"]}

    answer = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call.id, name=call.name, response={"choice": "a"}
                )
            )
        ],
    )
    resumed = [
        e
        async for e in runner.run_async(
            user_id=USER,
            session_id=session.id,
            invocation_id=interrupt.invocation_id,
            new_message=answer,
        )
    ]

    assert runs == [{}, {"review-1": {"choice": "a"}}]
    assert _outputs(resumed)["done"] == {"done": {"choice": "a"}}

    # The resume yields the finished nodes' events again, as new events with
    # new ids. The CLI deduplicates them by content (ADR-0009).
    original = next(e for e in events if e.node_info.name == "first" and e.output is not None)
    replayed = next(e for e in resumed if e.node_info.name == "first" and e.output is not None)
    assert replayed.id != original.id
    assert Payload.model_validate(replayed.output) == Payload.model_validate(original.output)


def test_mcp_toolset_builds_offline_with_sse_params_and_filter() -> None:
    toolset = McpToolset(
        connection_params=SseConnectionParams(
            url="http://127.0.0.1:9/sse",
            headers={"Authorization": "Bearer x"},
            timeout=5.0,
            sse_read_timeout=40.0,
        ),
        tool_filter=["extract_text"],
    )

    assert isinstance(toolset, McpToolset)


def test_mcp_tool_invocation_needs_an_agent_tool_context() -> None:
    """Why the runtime calls MCP through `mcp.ClientSession` (ADR-0009).

    `McpTool.run_async` requires a `tool_context`, which only exists inside an
    agent or node run, and session setup is retried internally once, which
    would add a second retry level under the node's RetryConfig.
    """
    params = inspect.signature(McpTool.run_async).parameters

    assert params["tool_context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["tool_context"].default is inspect.Parameter.empty
    assert hasattr(McpTool._create_session, "__wrapped__")


def test_mcp_sdk_sse_client_takes_the_spec_timeouts() -> None:
    sse = inspect.signature(sse_client).parameters
    session = inspect.signature(ClientSession.call_tool).parameters

    assert {"url", "headers", "timeout", "sse_read_timeout"} <= set(sse)
    assert "read_timeout_seconds" in session


def test_message_content_capture_is_switched_off_by_the_variables_the_cli_sets() -> None:
    """ADK reads the variables `cli.CONTENT_CAPTURE_OFF` sets to keep content off spans."""
    from google.adk.telemetry import context, tracing  # noqa: PLC0415

    from agent_runtime.cli import CONTENT_CAPTURE_OFF  # noqa: PLC0415

    assert tracing.ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS in CONTENT_CAPTURE_OFF
    assert tracing.OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT in CONTENT_CAPTURE_OFF
    source = inspect.getsource(context)
    for name in CONTENT_CAPTURE_OFF:
        assert name in source, name


def test_mcp_sdk_call_tool_takes_request_meta() -> None:
    """The run's `traceparent` reaches the MCP servers in the request's `_meta`."""
    assert "meta" in inspect.signature(ClientSession.call_tool).parameters

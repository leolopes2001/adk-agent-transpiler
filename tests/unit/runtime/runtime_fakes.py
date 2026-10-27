"""Helpers to run one node inside a real ADK workflow, with fake gateways."""

import json
import warnings
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from google.adk import Context, Event, Runner, Workflow
from google.adk.apps import App, ResumabilityConfig
from google.adk.models import BaseLlm, LlmCapabilities, LlmRequest, LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import BaseNode, FunctionNode
from google.genai import types
from pydantic import Field

USER = "unit"

type Response = Mapping[str, Any] | Exception | Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass
class FakeMcp:
    """McpGateway double: scripted answers per (server, tool), every call recorded."""

    responses: dict[tuple[str, str], list[Response]] = field(default_factory=dict)
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    closed: bool = False

    def script(self, server: str, tool: str, *responses: Response) -> None:
        self.responses.setdefault((server, tool), []).extend(responses)

    async def call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((server, tool, dict(args)))
        queue = self.responses[(server, tool)]
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(args)
        return response

    async def aclose(self) -> None:
        self.closed = True


class ScriptedLlm(BaseLlm):
    """Model double: the same answer to every request, requests recorded."""

    model: str = "scripted"
    answer: str = '{"exams": []}'
    requests: list[LlmRequest] = Field(default_factory=list)

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.answer)]))


@dataclass
class NodeRun:
    output: Any
    route: Any
    state: dict[str, Any]
    events: list[Event]


@dataclass
class NodeSession:
    """A one-node workflow run that can be resumed after an interrupt."""

    node: BaseNode
    runner: Runner
    service: InMemorySessionService
    session_id: str
    events: list[Event] = field(default_factory=list)

    async def result(self) -> NodeRun:
        final = await self.service.get_session(
            app_name="unit_app", user_id=USER, session_id=self.session_id
        )
        assert final is not None
        mine = [e for e in self.events if e.node_info.name == self.node.name]
        outputs = [e for e in mine if e.output is not None]
        routes = [e.actions.route for e in mine if e.actions.route is not None]
        return NodeRun(
            output=outputs[-1].output if outputs else None,
            route=routes[-1] if routes else None,
            state=dict(final.state),
            events=list(self.events),
        )

    def interrupt(self) -> Event | None:
        return next((e for e in reversed(self.events) if e.long_running_tool_ids), None)

    async def resume(self, response: dict[str, Any]) -> NodeRun:
        event = self.interrupt()
        assert event is not None
        assert event.content is not None
        assert event.content.parts
        call = event.content.parts[0].function_call
        assert call is not None
        answer = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response=response
                    )
                )
            ],
        )
        self.events = [
            e
            async for e in self.runner.run_async(
                user_id=USER,
                session_id=self.session_id,
                invocation_id=event.invocation_id,
                new_message=answer,
            )
        ]
        return await self.result()


async def start_node(
    node: BaseNode, node_input: Any = None, state: dict[str, Any] | None = None
) -> NodeSession:
    """Run `node` with `node_input` (resumable); exceptions from the node propagate."""

    async def feed(ctx: Context, node_input: Any) -> Event:
        return Event(output=feed_value)

    feed_value = node_input
    feeder = FunctionNode(func=feed, name="feeder")
    workflow = Workflow(name="unit_workflow", edges=[("START", feeder), (feeder, node)])
    service = InMemorySessionService()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = App(
            name="unit_app",
            root_agent=workflow,
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
    runner = Runner(app=app, session_service=service)
    session = await service.create_session(app_name="unit_app", user_id=USER, state=state)
    message = types.Content(role="user", parts=[types.Part(text=json.dumps({}))])
    run = NodeSession(node, runner, service, session.id)
    run.events = [
        e async for e in runner.run_async(user_id=USER, session_id=session.id, new_message=message)
    ]
    return run


async def run_node(
    node: BaseNode, node_input: Any = None, state: dict[str, Any] | None = None
) -> NodeRun:
    """Run `node` with `node_input`; exceptions from the node propagate."""
    return await (await start_node(node, node_input, state)).result()


CATALOG_SHA = "28" * 32


def candidate(code: str, name: str, score: float, matched_on: str = "name") -> dict[str, Any]:
    return {
        "code": code,
        "name": name,
        "score": score,
        "matched_on": matched_on,
        "category": "Hematologia",
    }


def rag_body(*results: list[dict[str, Any]], queries: list[str] | None = None) -> dict[str, Any]:
    """`batch_search` body with one result per candidate list."""
    names = queries or [f"q{i}" for i in range(len(results))]
    return {
        "contract_version": "1",
        "catalog_version": "2026.10.0",
        "catalog_sha256": CATALOG_SHA,
        "results": [
            {
                "contract_version": "1",
                "catalog_version": "2026.10.0",
                "catalog_sha256": CATALOG_SHA,
                "query": query,
                "normalized_query": query.lower(),
                "candidates": found,
            }
            for query, found in zip(names, results, strict=True)
        ],
    }

import socket
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

from lab_common import telemetry
from telemetry_capture import Exported

pytestmark = pytest.mark.unit


@pytest.fixture
def listening_port() -> Iterator[int]:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        yield server.getsockname()[1]


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class TestCollectorEndpoint:
    def test_unset_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(telemetry.ENDPOINT_ENV, raising=False)
        assert telemetry.collector_endpoint() is None

    def test_name_that_does_not_resolve_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # What every service sees without the `observability` profile.
        monkeypatch.setenv(telemetry.ENDPOINT_ENV, "http://observability.invalid:4318")
        assert telemetry.collector_endpoint() is None

    def test_nothing_listening_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(telemetry.ENDPOINT_ENV, f"http://127.0.0.1:{_closed_port()}")
        assert telemetry.collector_endpoint() is None

    def test_listening_collector_is_on(
        self, monkeypatch: pytest.MonkeyPatch, listening_port: int
    ) -> None:
        monkeypatch.setenv(telemetry.ENDPOINT_ENV, f"http://127.0.0.1:{listening_port}/")
        assert telemetry.collector_endpoint() == f"http://127.0.0.1:{listening_port}"


def test_setup_installs_nothing_without_a_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(telemetry.ENDPOINT_ENV, raising=False)

    def refuse(provider: object) -> None:
        raise AssertionError("a tracer provider was installed")

    monkeypatch.setattr(trace, "set_tracer_provider", refuse)
    assert telemetry.setup("agent", instance_id="run") is None


def _span(**overrides: object) -> ReadableSpan:
    fields: dict[str, object] = {
        "name": "call_llm",
        "attributes": {
            "gen_ai.request.model": "gemini",
            "gen_ai.usage.input_tokens": 12,
            "gcp.vertex.agent.llm_request": '{"contents": "Paciente: Maria"}',
            "gcp.vertex.agent.llm_response": "Hemograma",
            "lab.run_id": "run",
        },
        "events": [
            Event(
                "exception",
                {"exception.type": "ValueError", "exception.message": "input was 'Maria'"},
            )
        ],
        "status": Status(StatusCode.ERROR, "ValueError: input was 'Maria'"),
        **overrides,
    }
    return ReadableSpan(**fields)  # type: ignore[arg-type]


class TestSafeSpanExporter:
    def test_keeps_only_allowed_attributes(self) -> None:
        exported = InMemorySpanExporter()
        telemetry.SafeSpanExporter(exported).export([_span()])

        [span] = exported.get_finished_spans()
        assert dict(span.attributes or {}) == {
            "gen_ai.request.model": "gemini",
            "gen_ai.usage.input_tokens": 12,
            "lab.run_id": "run",
        }

    def test_events_keep_the_exception_type_only(self) -> None:
        span = telemetry.safe_span(_span())
        [event] = span.events
        assert (event.name, dict(event.attributes or {})) == (
            "exception",
            {"exception.type": "ValueError"},
        )

    def test_status_loses_its_description(self) -> None:
        status = telemetry.safe_span(_span()).status
        assert (status.status_code, status.description) == (StatusCode.ERROR, None)


class TestPropagation:
    def test_server_span_joins_the_callers_trace(self, exported: Exported) -> None:
        tracer = telemetry.tracer(__name__)
        with tracer.start_as_current_span("client"):
            meta = telemetry.trace_meta()
        with telemetry.server_span("tools/call batch_search", {**meta, "other": 1}, run_id="r"):
            pass

        client, server = exported.spans()
        assert server.kind == SpanKind.SERVER
        assert server.context.trace_id == client.context.trace_id
        assert server.parent is not None
        assert server.parent.span_id == client.context.span_id
        assert dict(server.attributes or {}) == {"lab.run_id": "r"}

    def test_server_span_without_meta_starts_a_trace(self, exported: Exported) -> None:
        with telemetry.server_span("tools/call extract_text", None):
            pass
        [span] = exported.spans()
        assert span.parent is None

    def test_no_traceparent_outside_a_span(self) -> None:
        assert telemetry.trace_meta() == {}

    def test_request_meta_outside_a_request_is_none(self) -> None:
        assert telemetry.request_meta(None) is None


def test_annotate_and_fail_set_codes_never_messages(exported: Exported) -> None:
    with telemetry.tracer(__name__).start_as_current_span("node"):
        telemetry.annotate(route="ok", mentions=3, skipped=None)
        telemetry.fail("MCP_TIMEOUT")

    [span] = exported.spans()
    assert dict(span.attributes or {}) == {
        "lab.route": "ok",
        "lab.mentions": 3,
        "lab.error_code": "MCP_TIMEOUT",
    }
    assert span.status.status_code == StatusCode.ERROR

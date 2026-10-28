import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from scheduler_api import telemetry
from scheduler_api.settings import Settings
from telemetry_capture import Exported

pytestmark = pytest.mark.unit

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_ID = "00f067aa0ba902b7"


def _headers(settings: Settings, **extra: str) -> dict[str, str]:
    return {"X-API-Key": settings.scheduler_api_key.get_secret_value(), **extra}


def test_request_span_continues_the_callers_trace(
    app: FastAPI, settings: Settings, exported: Exported
) -> None:
    traceparent = f"00-{TRACE_ID}-{PARENT_ID}-01"
    with TestClient(app) as client:
        client.get(
            "/v1/exams/LAB-10001",
            headers=_headers(settings, traceparent=traceparent, **{"X-Request-ID": "req-1"}),
        )

    [span] = exported.spans()
    assert (span.name, span.kind) == ("GET /v1/exams/{code}", SpanKind.SERVER)
    assert f"{span.context.trace_id:032x}" == TRACE_ID
    assert span.parent is not None
    assert f"{span.parent.span_id:016x}" == PARENT_ID
    assert dict(span.attributes or {}) == {
        "http.request.method": "GET",
        "http.route": "/v1/exams/{code}",
        "http.response.status_code": 200,
        "lab.request_id": "req-1",
    }


def test_unmatched_paths_are_not_named_after_the_path(
    app: FastAPI, settings: Settings, exported: Exported
) -> None:
    with TestClient(app) as client:
        client.get("/v1/Maria-da-Silva", headers=_headers(settings))

    [span] = exported.spans()
    assert span.name == "GET <unmatched>"
    assert "Maria" not in str(span.attributes)


@pytest.mark.parametrize("path", ["/health/live", "/health/ready"])
def test_health_checks_are_not_traced(app: FastAPI, exported: Exported, path: str) -> None:
    with TestClient(app) as client:
        client.get(path)

    assert exported.spans() == []


def test_server_errors_mark_the_span_failed(app: FastAPI, exported: Exported) -> None:
    def broken() -> None:
        raise RuntimeError("boom")

    app.add_api_route("/v1/broken", broken)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.get("/v1/broken")

    [span] = exported.spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.events == ()


def test_setup_installs_nothing_without_a_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(telemetry.ENDPOINT_ENV, raising=False)
    monkeypatch.setattr(telemetry, "_provider", None)

    def refuse(provider: object) -> None:
        raise AssertionError("a tracer provider was installed")

    monkeypatch.setattr(trace, "set_tracer_provider", refuse)
    assert telemetry.setup() is None

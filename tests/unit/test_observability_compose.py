"""The optional `observability` service: local only and wired to the services."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = (ROOT / "docker-compose.yml").read_text("utf-8")


def _service(name: str) -> str:
    match = re.search(rf"^  {name}:\n(.*?)(?=^  \S|^\S)", COMPOSE, re.MULTILINE | re.DOTALL)
    assert match, name
    return match.group(1)


def test_dashboard_image_is_pinned_by_tag_and_digest() -> None:
    [image] = re.findall(r"image: (\S+)", _service("observability"))

    assert re.fullmatch(
        r"mcr\.microsoft\.com/dotnet/aspire-dashboard:[\d.]+@sha256:[0-9a-f]{64}", image
    )


def test_only_the_ui_is_published_and_on_loopback() -> None:
    service = _service("observability")

    assert re.findall(r"ports: (.*)", service) == ['["127.0.0.1:18888:18888"]']
    assert "profiles: [observability]" in service
    # OTLP/gRPC and the dashboard's MCP endpoint are switched off.
    assert 'DOTNET_DASHBOARD_OTLP_ENDPOINT_URL: ""' in service
    assert 'DOTNET_DASHBOARD_MCP_ENDPOINT_URL: ""' in service


def test_services_send_otlp_http_to_the_dashboard() -> None:
    assert "x-otlp-endpoint: &otlp-endpoint http://observability:18890" in COMPOSE
    for name in ("scheduler-api", "ocr-mcp", "rag-mcp", "agent"):
        assert "OTEL_EXPORTER_OTLP_ENDPOINT: *otlp-endpoint" in _service(name), name


def test_no_service_needs_the_dashboard() -> None:
    # Optional dependencies only: without the profile, compose ignores them.
    assert COMPOSE.count("observability: *after-collector") == 3
    assert re.search(r"x-after-collector: &after-collector\n(?:  .*\n)*  required: false", COMPOSE)

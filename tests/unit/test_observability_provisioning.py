"""The Grafana provisioning in observability/grafana matches what the code exports."""

import json
import re
from pathlib import Path

import pytest

from agent_runtime.cli import RUN_SPAN, SERVICE

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
GRAFANA = ROOT / "observability" / "grafana"
DASHBOARD = GRAFANA / "dashboards" / "lab-exam-agent.json"
SERVICES = {"agent", "ocr-mcp", "rag-mcp", "scheduler-api"}


def _dashboard() -> dict[str, object]:
    data: dict[str, object] = json.loads(DASHBOARD.read_text("utf-8"))
    return data


def _queries() -> list[str]:
    panels = _dashboard()["panels"]
    assert isinstance(panels, list)
    return [t.get("expr") or t.get("query") for p in panels for t in p.get("targets", [])]


def test_every_mounted_file_exists() -> None:
    compose = (ROOT / "docker-compose.yml").read_text("utf-8")
    mounts = re.findall(r"- \./(observability/[^:]+):", compose)

    assert len(mounts) == 3
    for path in mounts:
        assert (ROOT / path).exists(), path


def test_dashboard_is_the_home_page() -> None:
    compose = (ROOT / "docker-compose.yml").read_text("utf-8")
    [home] = re.findall(r"GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH: (\S+)", compose)

    assert home == f"/otel-lgtm/lab-dashboards/{DASHBOARD.name}"
    assert (_dashboard()["uid"], _dashboard()["title"]) == ("lab-exam-agent", "Lab Exam Agent")


def test_run_id_variable_lists_the_agent_runs() -> None:
    templating = _dashboard()["templating"]
    assert isinstance(templating, dict)
    [variable] = templating["list"]

    # The CLI's resource carries the run id as service.instance.id (Loki: service_instance_id).
    assert variable["name"] == "run_id"
    assert variable["query"]["label"] == "service_instance_id"
    assert variable["query"]["stream"] == f'{{service_name="{SERVICE}"}}'


def test_trace_queries_use_the_run_span_and_its_run_id() -> None:
    traceql = [q for q in _queries() if "span.lab.run_id" in q]

    assert len(traceql) == 2
    for query in traceql:
        assert f'name="{RUN_SPAN}"' in query


def test_log_queries_cover_the_four_services() -> None:
    selectors = [m for q in _queries() for m in re.findall(r'service_name=~"([^"]+)"', q)]

    assert selectors
    assert all(set(selector.split("|")) == SERVICES for selector in selectors)


def test_logs_link_to_traces_and_back() -> None:
    datasources = (GRAFANA / "provisioning" / "datasources" / "datasources.yaml").read_text()

    assert "matcherRegex: trace_id" in datasources
    assert "datasourceUid: tempo" in datasources
    assert 'trace_id="$${__trace.traceId}"' in datasources

"""The e2e stack is up as the evaluator gets it: healthy services, the API on
loopback only, no internet for the API, the generated code in the volume."""

import json

import pytest

from e2e.helpers import REPO_ROOT, ComposeStack

pytestmark = pytest.mark.e2e

SERVICES = {"postgres", "scheduler-api", "ocr-mcp", "rag-mcp"}
PACKAGES = ("exam_scheduling_agent", "exam_lookup_only")


def test_every_service_is_healthy(e2e_stack: ComposeStack) -> None:
    output = e2e_stack.compose("ps", "--format", "json").stdout
    containers = [json.loads(line) for line in output.splitlines() if line.strip()]

    health = {c["Service"]: (c["State"], c["Health"]) for c in containers}
    assert health == dict.fromkeys(SERVICES, ("running", "healthy"))


def test_api_is_published_on_loopback_and_answers(e2e_stack: ComposeStack) -> None:
    # On Docker Engine for Linux this also proves that the `edge` network without
    # IP masquerading still forwards connections to the published port.
    host, _, port = e2e_stack.api_url().removeprefix("http://").rpartition(":")

    status, body = e2e_stack.api_get("/health/ready")

    assert host == "127.0.0.1"
    assert int(port) > 0
    assert status == 200, body


def test_api_rejects_a_wrong_key(e2e_stack: ComposeStack) -> None:
    status, body = e2e_stack.api_get("/v1/exams/LAB-10001", api_key="wrong-" + "k" * 32)

    assert status == 401
    assert body["type"].endswith("/unauthorized")


def test_api_container_has_no_internet(e2e_stack: ComposeStack) -> None:
    # An address, not a name: the probe measures routing, not the resolver.
    probe = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=5).close()\n"
        "except OSError:\n"
        "    raise SystemExit(1)\n"
    )

    result = e2e_stack.compose(
        "exec", "-T", "scheduler-api", "python", "-c", probe, check=False, timeout=60
    )

    assert result.returncode == 1, result.stderr


@pytest.mark.parametrize("package", PACKAGES)
def test_generated_code_in_the_volume_matches_the_golden_files(
    e2e_stack: ComposeStack, package: str
) -> None:
    for name in ("agent.py", "_manifest.json"):
        result = e2e_stack.compose(
            "run", "--rm", "-T", "--entrypoint", "cat", "agent", f"/app/generated/{package}/{name}"
        )
        assert result.stdout == (REPO_ROOT / "tests" / "golden" / package / name).read_text(
            "utf-8"
        ), f"{package}/{name}"

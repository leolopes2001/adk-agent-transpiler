"""The Docker Compose stack of the end-to-end tests, driven from the host.

The stack runs under its own project name (`lab-agent-e2e`) with the e2e
override: FakeLlm in the agent, the API on a random loopback port. Secrets are
generated for each session and handed to Compose through the environment, with
an empty env file, so the developer's `.env` never takes part. Everything the
tests read (CLI output, logs, database rows) is collected here, on the host:
the containers have read-only file systems and never write to the repository.
"""

import json
import os
import secrets
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = "lab-agent-e2e"
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.e2e.yml")
EVIDENCE_DIR = REPO_ROOT / "evidence" / "e2e"

MAIN_SPEC = "/app/agent_specs/exam_scheduling_agent.json"
LOOKUP_SPEC = "/app/agent_specs/examples/exam_lookup_only.json"
LOOKUP_AGENT_DIR = "/app/generated/exam_lookup_only"
SAMPLES = "/app/inputs/samples"
TEST_IMAGES = "/app/inputs/test-images"
FAKE_LLM_FIXTURES = "/app/tests/fixtures/fake_llm"

SECRET_NAMES = ("POSTGRES_PASSWORD", "SCHEDULER_API_KEY", "MCP_AUTH_TOKEN", "PII_HMAC_SECRET")
LLM_KEY_ENV = "GOOGLE_API_KEY"

BUILD_TIMEOUT_S = 900
RUN_TIMEOUT_S = 180


@dataclass(frozen=True)
class AgentRun:
    """One `lab-agent run` in a container: exit code, JSON report (if any), raw output."""

    args: tuple[str, ...]
    exit_code: int
    report: dict[str, Any] | None
    stdout: str
    stderr: str


@dataclass
class ComposeStack:
    env: dict[str, str]
    env_file: Path
    runs: list[AgentRun] = field(default_factory=list)

    @property
    def api_key(self) -> str:
        return self.env["SCHEDULER_API_KEY"]

    def compose(
        self,
        *args: str,
        check: bool = True,
        timeout: float = RUN_TIMEOUT_S,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = ["docker", "compose", "-p", PROJECT, "--env-file", str(self.env_file)]
        for name in COMPOSE_FILES:
            command += ["-f", name]
        result = subprocess.run(  # noqa: S603 - fixed program, arguments built here
            [*command, *args],
            cwd=REPO_ROOT,
            env={**self.env, **(extra_env or {})},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            raise AssertionError(
                f"docker compose {' '.join(args)} exited {result.returncode}:\n{result.stderr}"
            )
        return result

    def up(self) -> None:
        # `up --build` skips the agent (profile `cli`), and `run` never rebuilds a
        # stale image, so the agent image is built explicitly first.
        self.compose("build", "agent", timeout=BUILD_TIMEOUT_S)
        self.compose(
            "up", "-d", "--build", "--wait", "--wait-timeout", "300", timeout=BUILD_TIMEOUT_S
        )

    def transpile(self, spec: str = MAIN_SPEC) -> str:
        return self.compose("run", "--rm", "transpile", spec).stdout

    def down(self) -> None:
        self.compose("--profile", "*", "down", "-v", "--remove-orphans", check=False)

    def logs(self, *services: str) -> str:
        return self.compose("--profile", "*", "logs", "--no-color", *services).stdout

    def run_agent(
        self,
        image: str,
        *extra: str,
        env: Mapping[str, str] | None = None,
        pass_env: Sequence[str] = (),
        entrypoint: str | None = None,
    ) -> AgentRun:
        """`lab-agent run --image <image> <extra>`; the report is parsed when stdout is JSON.

        `env` sets container variables (values on the command line, so never
        secrets); `pass_env` forwards variables from the host environment by name.
        """
        options = ["run", "--rm", "-T"]
        for name, value in (env or {}).items():
            options += ["-e", f"{name}={value}"]
        for name in pass_env:
            options += ["-e", name]
        if entrypoint is not None:
            options += ["--entrypoint", entrypoint]
        args = ("run", "--image", image, *extra)
        forwarded = {name: os.environ.get(name, "") for name in pass_env}
        result = self.compose(*options, "agent", *args, check=False, extra_env=forwarded)
        try:
            report = json.loads(result.stdout)
        except ValueError:
            report = None
        run = AgentRun(
            args=args,
            exit_code=result.returncode,
            report=report if isinstance(report, dict) else None,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        self.runs.append(run)
        return run

    def api_url(self, path: str = "") -> str:
        published = self.compose("port", "scheduler-api", "8000").stdout.strip()
        return f"http://{published}{path}"

    def api_get(self, path: str, *, api_key: str | None = None) -> tuple[int, Any]:
        """GET on the published API port, with the API key unless `api_key` is given."""
        request = urllib.request.Request(  # noqa: S310 - http on loopback, built here
            self.api_url(path), headers={"X-API-Key": self.api_key if api_key is None else api_key}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"null")

    def db_rows(self, sql: str) -> list[list[str]]:
        """Rows of `sql` from psql (`-At`, unaligned, `|` between columns)."""
        output = self.compose(
            "exec", "-T", "postgres", "psql", "-U", "scheduler", "-d", "scheduler", "-At",
            "-v", "ON_ERROR_STOP=1", "-c", sql,
        ).stdout  # fmt: skip
        return [line.split("|") for line in output.splitlines() if line]

    def count(self, table: str) -> int:
        return int(self.db_rows(f"SELECT count(*) FROM {table}")[0][0])  # noqa: S608 - fixed names


def google_api_key() -> str:
    """The real Gemini key: GOOGLE_API_KEY in the environment, else from `.env`."""
    value = os.environ.get(LLM_KEY_ENV, "")
    env_file = REPO_ROOT / ".env"
    if not value and env_file.is_file():
        for line in env_file.read_text("utf-8").splitlines():
            name, sep, raw = line.partition("=")
            if sep and name.strip() == LLM_KEY_ENV:
                value = raw.strip().strip("'\"")
    return value


def _base_env(workdir: Path) -> dict[str, str]:
    """Host PATH and Docker settings, plus fresh secrets; no other host variable.

    COMPOSE_PROFILES passes through, so `COMPOSE_PROFILES=observability` runs the
    suite with the observability stack up.
    """
    keep = (
        "PATH",
        "HOME",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "XDG_RUNTIME_DIR",
        "COMPOSE_PROFILES",
    )
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env.update({name: secrets.token_urlsafe(32) for name in SECRET_NAMES})
    env["COMPOSE_PROJECT_NAME"] = PROJECT
    env["TMPDIR"] = str(workdir)
    return env


def save_logs(stack: ComposeStack, target: Path = EVIDENCE_DIR) -> None:
    """Service logs (`compose.log`) and the stderr of every agent run (`agent.log`).

    The agent containers are removed after each run, so their logs exist only
    in what the runs captured.
    """
    target.mkdir(parents=True, exist_ok=True)
    (target / "compose.log").write_text(stack.logs(), "utf-8")
    (target / "agent.log").write_text("".join(run.stderr for run in stack.runs), "utf-8")


@contextmanager
def compose_stack() -> Iterator[ComposeStack]:
    """Up, transpile both specs, yield; `down -v` on the way out, even after a failure."""
    with tempfile.TemporaryDirectory(prefix="lab-agent-e2e-") as tmp:
        workdir = Path(tmp)
        env_file = workdir / "empty.env"
        env_file.write_text("", "utf-8")
        stack = ComposeStack(env=_base_env(workdir), env_file=env_file)
        try:
            stack.up()
            stack.transpile(MAIN_SPEC)
            stack.transpile(LOOKUP_SPEC)
            yield stack
        finally:
            stack.down()

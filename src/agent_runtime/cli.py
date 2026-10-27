"""`lab-agent run`: runs a generated workflow on one order image.

    lab-agent run --image <path> [--agent-dir generated/exam_scheduling_agent]
        [--spec specs/exam_scheduling_agent.json] [--non-interactive]
        [--output table|json] [--run-id <uuid>] [--verbose]

Before anything runs, the CLI checks the configuration (every missing
variable is listed, exit 8), that the generated code matches the spec
(manifest hash, exit 7), that the package is under GENERATED_ROOT and that
the image is acceptable (exit 2). The workflow then runs with an in-memory
session; review requests are answered at the terminal, choosing among the
offered candidates only. In table mode a short header (agent and image)
opens the run on stderr. The report goes to stdout. The JSON logs go to
stderr and, by default, only errors, so the terminal shows the review
questions, the report and the summary; --verbose (or LOG_LEVEL=INFO) shows
every run event.

Exit codes: 0 done, 2 invalid input, 3 blocked by the guard, 4 low OCR
quality, 5 no exam resolved, 6 dependency failure or scheduling refused,
7 stale generated code, 8 invalid configuration.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import sys
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import structlog
from google.adk import Agent, Event, Runner, Workflow
from google.adk.apps import App
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import BaseNode, NodeTimeoutError
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from rich.console import Console

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.errors import AgentRuntimeError, ConfigurationError, InvalidInputError
from agent_runtime.nodes.ocr_extract import OcrExtractParams, load_image
from agent_runtime.nodes.present import RESULT_STATE_KEY, RunResult
from agent_runtime.report import (
    EXIT_CONFIG,
    EXIT_DEPENDENCY,
    EXIT_INVALID_INPUT,
    EXIT_STALE_CODE,
    RunTrace,
    build_report,
    exit_code_for,
    render_text,
)
from agent_runtime.schemas import RunInput
from agent_runtime.settings import AgentSettings, missing
from agent_runtime.terminal import ask_review, welcome
from lab_common import logging as lab_logging
from lab_common.mcp_server import LibraryLogFilter

SERVICE = "agent"
USER_ID = "lab-agent"
MANIFEST = "_manifest.json"
LIBRARY_LOGGERS = ("google_adk", "google.adk", "google_genai", "google.genai", "mcp", "httpx")
"""Libraries whose log records may quote payloads; LibraryLogFilter cuts their messages."""
LLM_KEY_ENV = "GOOGLE_API_KEY"
QUIET_LOG_LEVEL = "ERROR"
"""Warnings of a run (retries, injection, grounding) are in the report already."""
VERBOSE_LOG_LEVEL = "INFO"
REPORT_WIDTH = 120
"""Width of the table report when stdout is not a terminal (a file, a pipe, a capture)."""

type ReviewPrompt = Callable[[Mapping[str, Any]], dict[str, Any]]
type Announce = Callable[["LoadedAgent", str], None]

log = lab_logging.get_logger(__name__)


class CliError(Exception):
    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.message = message


@dataclass(frozen=True)
class LoadedAgent:
    name: str
    description: str
    app: App
    deps: RuntimeDeps | None
    spec_sha256: str
    output_format: str = "table"
    """Format of the spec's `present` node, used when --output is not given."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab-agent", description="Run a generated agent.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="process one medical order image")
    run.add_argument("--image", required=True, help="PNG or JPEG under INPUT_ROOT")
    run.add_argument("--agent-dir", default="generated/exam_scheduling_agent", type=Path)
    run.add_argument("--spec", default="specs/exam_scheduling_agent.json", type=Path)
    run.add_argument("--non-interactive", action="store_true", help="reject review items")
    run.add_argument("--output", choices=("table", "json"), default=None)
    run.add_argument("--run-id", default=None, help="UUID used to correlate logs")
    run.add_argument("--verbose", action="store_true", help="JSON logs of the run on stderr")
    return parser


def _run_id(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise CliError(EXIT_INVALID_INPUT, "--run-id must be a UUID") from None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import(agent_dir: Path) -> ModuleType:
    parent = str(agent_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        module = importlib.import_module(agent_dir.name)
    except ConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001 - any import failure means unusable generated code
        raise CliError(
            EXIT_STALE_CODE, f"generated code failed to import ({type(exc).__name__})"
        ) from None
    origin = Path(module.__file__ or "").resolve()
    if not origin.is_relative_to(agent_dir):
        raise CliError(EXIT_STALE_CODE, f"module {agent_dir.name!r} is loaded from elsewhere")
    return module


def _present_format(node: object) -> str | None:
    if isinstance(node, Mapping):
        if node.get("kind") == "present":
            params = node.get("params")
            return str(params.get("format", "table")) if isinstance(params, Mapping) else "table"
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            found = _present_format(child)
            if found is not None:
                return found
    return None


def load_agent(agent_dir: Path, spec: Path, settings: AgentSettings) -> LoadedAgent:
    """Generated app after the manifest and location checks (exit 7 or 8 on failure)."""
    resolved = agent_dir.resolve()
    if not resolved.is_relative_to(settings.generated_root):
        raise CliError(EXIT_CONFIG, "--agent-dir must be under GENERATED_ROOT")
    if not spec.is_file():
        raise CliError(EXIT_CONFIG, "--spec file not found")
    try:
        manifest = json.loads((resolved / MANIFEST).read_text("utf-8"))
    except (OSError, ValueError):
        raise CliError(EXIT_STALE_CODE, "generated code has no readable manifest") from None
    spec_sha256 = _sha256(spec)
    try:
        output_format = _present_format(json.loads(spec.read_text("utf-8"))) or "table"
    except (OSError, ValueError):
        raise CliError(EXIT_CONFIG, "--spec file is not valid JSON") from None
    if manifest.get("spec_sha256") != spec_sha256:
        raise CliError(
            EXIT_STALE_CODE, "generated code is out of date with the spec; run lab-transpile"
        )
    module = _import(resolved)
    app = getattr(module, "app", None)
    if not isinstance(app, App):
        raise CliError(EXIT_STALE_CODE, "generated package has no App")
    agent_module = sys.modules.get(f"{module.__name__}.agent")
    deps = getattr(agent_module, "deps", None)
    return LoadedAgent(
        name=app.name,
        description=getattr(app.root_agent, "description", "") or "",
        app=app,
        deps=deps if isinstance(deps, RuntimeDeps) else None,
        spec_sha256=spec_sha256,
        output_format=output_format,
    )


def _nodes(node: BaseNode) -> Iterator[BaseNode]:
    yield node
    if isinstance(node, Workflow) and node.graph is not None:
        seen: set[str] = set()
        for edge in node.graph.edges:
            for child in (edge.from_node, edge.to_node):
                if isinstance(child, BaseNode) and child.name not in seen:
                    seen.add(child.name)
                    yield from _nodes(child)


def required_env(agent: LoadedAgent) -> list[str]:
    names = agent.deps.required_env() if agent.deps else []
    root = agent.app.root_agent
    uses_gemini = isinstance(root, BaseNode) and any(
        isinstance(node, Agent) and isinstance(node.model, str) for node in _nodes(root)
    )
    return sorted({*names, *([LLM_KEY_ENV] if uses_gemini else [])})


def _event_key(event: Event) -> str:
    """What a node run produced, independent of the event id and timestamp."""
    if not event.node_info.path:
        return ""
    # A replay may carry the dump of a model the original event held as is, and
    # it loses the route, so the route is not part of the key.
    value = event.output
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    output = json.dumps(value, sort_keys=True, default=str)
    return f"{event.invocation_id}|{event.node_info.path}|{event.error_code}|{output}"


def _interrupt(events: list[Event]) -> tuple[Event, types.FunctionCall] | None:
    for event in events:
        if event.long_running_tool_ids and event.content and event.content.parts:
            call = event.content.parts[0].function_call
            if call is not None:
                return event, call
    return None


def _root_cause(exc: BaseException) -> BaseException:
    seen: set[int] = set()
    while id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            exc = exc.exceptions[0]
            continue
        inner = getattr(exc, "error", None)
        if isinstance(inner, BaseException):
            exc = inner
            continue
        break
    return exc


def _failure(exc: BaseException) -> tuple[int, str]:
    cause = _root_cause(exc)
    if isinstance(cause, InvalidInputError):
        return EXIT_INVALID_INPUT, cause.code
    if isinstance(cause, ConfigurationError):
        return EXIT_CONFIG, cause.code
    if isinstance(cause, AgentRuntimeError):
        return EXIT_DEPENDENCY, cause.code
    if isinstance(cause, NodeTimeoutError):
        return EXIT_DEPENDENCY, "NODE_TIMEOUT"
    if isinstance(cause, genai_errors.APIError):
        status = (
            cause.status
            if isinstance(cause.status, str) and cause.status.isidentifier()
            else cause.code
        )
        return EXIT_DEPENDENCY, f"LLM_{status}"
    return EXIT_DEPENDENCY, "INTERNAL_ERROR"


async def _drive(
    runner: Runner,
    session_id: str,
    message: types.Content,
    trace: RunTrace,
    review: ReviewPrompt | None,
) -> None:
    """Runs to the end, answering review interrupts with `review`.

    A resumed invocation yields again the events of the nodes that already
    finished, as new events with new ids (ADR-0009). Events seen in an earlier
    round, by id or by content, are neither traced nor logged again; repeats
    within one round (retry attempts) are kept.
    """
    invocation_id: str | None = None
    earlier: set[str] = set()
    while True:
        events: list[Event] = []
        this_round: set[str] = set()
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, invocation_id=invocation_id, new_message=message
        ):
            keys = {event.id, _event_key(event)} - {""}
            if keys & earlier:
                continue
            this_round |= keys
            trace.observe(event)
            events.append(event)
            name = event.node_info.name
            if event.error_code:
                log.warning("node_retry", node=name, error_code=event.error_code)
            elif event.output is not None:
                log.info("node_finished", node=name, route=event.actions.route)
        earlier |= this_round
        pending = _interrupt(events)
        if pending is None:
            return
        event, call = pending
        if review is None:
            raise CliError(EXIT_DEPENDENCY, "the run asked for input in non-interactive mode")
        args = call.args or {}
        answer = review(args.get("payload") or {})
        invocation_id = event.invocation_id
        message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response=answer
                    )
                )
            ],
        )


@dataclass
class RunOutcome:
    exit_code: int
    report: dict[str, Any]
    result: RunResult | None


async def execute(  # noqa: PLR0913 - the CLI options
    *,
    image: str,
    agent_dir: Path,
    spec: Path,
    interactive: bool,
    run_id: str,
    settings: AgentSettings,
    review: ReviewPrompt = ask_review,
    announce: Announce | None = None,
) -> RunOutcome:
    """Checks, runs and reports; raises CliError for failures before the run."""
    try:
        agent = load_agent(agent_dir, spec, settings)
    except ConfigurationError as error:
        raise CliError(EXIT_CONFIG, f"invalid configuration: {error.code}") from None
    absent = missing(required_env(agent))
    if absent:
        raise CliError(EXIT_CONFIG, "missing environment variables: " + ", ".join(absent))
    try:
        path = load_image(image, settings.input_root, OcrExtractParams().max_image_bytes)
    except InvalidInputError as error:
        raise CliError(EXIT_INVALID_INPUT, f"invalid image: {error.code}") from None
    del path
    resolved = (settings.input_root / image).resolve()
    relative = resolved.relative_to(settings.input_root)
    if announce is not None:
        announce(agent, image)

    trace = RunTrace()
    service = InMemorySessionService()
    runner = Runner(app=agent.app, session_service=service)
    session = await service.create_session(
        app_name=agent.app.name,
        user_id=USER_ID,
        state={"run.interactive": interactive, "run.id": run_id},
    )
    message = types.Content(
        role="user", parts=[types.Part(text=RunInput(image_path=str(relative)).model_dump_json())]
    )
    log.info("run_started", agent=agent.name, interactive=interactive)
    result: RunResult | None = None
    error_code: str | None = None
    try:
        await _drive(runner, session.id, message, trace, review if interactive else None)
        final = await service.get_session(
            app_name=agent.app.name, user_id=USER_ID, session_id=session.id
        )
        raw = final.state.get(RESULT_STATE_KEY) if final else None
        if raw is None:
            exit_code, error_code = EXIT_DEPENDENCY, "NO_RESULT"
        else:
            result = RunResult.model_validate(raw)
            exit_code = exit_code_for(result)
    except CliError as error:
        exit_code, error_code = error.exit_code, "INPUT_REQUIRED"
    except Exception as exc:  # noqa: BLE001 - every failure becomes an exit code
        exit_code, error_code = _failure(exc)
        log.error("run_failed", error_code=error_code, error_type=type(_root_cause(exc)).__name__)
    finally:
        trace.finish()
        if agent.deps is not None:
            await agent.deps.aclose()
    report = build_report(
        run_id=run_id,
        agent=agent.name,
        spec_sha256=agent.spec_sha256,
        trace=trace,
        exit_code=exit_code,
        result=result,
        error_code=error_code,
    )
    log.info(
        "run_finished",
        outcome=report["outcome"],
        exit_code=exit_code,
        duration_ms=report["timings_ms"]["total"],
    )
    return RunOutcome(exit_code, report, result)


_ERROR_MESSAGES = {
    "NO_RESULT": "a execução terminou sem resultado",
    "INPUT_REQUIRED": "a execução pediu revisão em modo não interativo",
}


def log_level(settings: AgentSettings, *, verbose: bool) -> str:
    """--verbose shows at least INFO; otherwise LOG_LEVEL, or ERROR when unset."""
    if verbose:
        return "DEBUG" if settings.log_level == "DEBUG" else VERBOSE_LOG_LEVEL
    return settings.log_level or QUIET_LOG_LEVEL


def _configure_logging(level: str) -> None:
    lab_logging.configure(SERVICE, level, filters=[LibraryLogFilter(LIBRARY_LOGGERS)])
    # One INFO line per HTTP request (MCP keeps SSE streams busy) says nothing about the run.
    logging.getLogger("httpx").setLevel(max(logging.WARNING, logging.getLogger().level))
    # ADK warns about experimental features on every import; they are not run events.
    warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")


def announcer(output: str | None) -> Announce:
    """Welcome header on stderr, only when the report will be a table."""

    def announce(agent: LoadedAgent, image: str) -> None:
        if (output or agent.output_format) == "table":
            welcome(Console(stderr=True), agent.name, agent.description, image)

    return announce


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = AgentSettings.from_env()
        _configure_logging(log_level(settings, verbose=args.verbose))
        run_id = _run_id(args.run_id)
        structlog.contextvars.bind_contextvars(run_id=run_id)
        outcome = asyncio.run(
            execute(
                image=args.image,
                agent_dir=args.agent_dir,
                spec=args.spec,
                interactive=not args.non_interactive,
                run_id=run_id,
                settings=settings,
                announce=announcer(args.output),
            )
        )
    except ConfigurationError as error:
        print(f"lab-agent: invalid configuration: {error.code}", file=sys.stderr)
        return EXIT_CONFIG
    except CliError as error:
        print(f"lab-agent: {error.message}", file=sys.stderr)
        return error.exit_code
    finally:
        structlog.contextvars.clear_contextvars()
    output = args.output or (outcome.result.format if outcome.result else "table")
    if output == "json":
        print(json.dumps(outcome.report, ensure_ascii=False, indent=2))
    else:
        stdout = Console()
        sys.stdout.write(
            render_text(
                outcome.report,
                outcome.result,
                verbose=args.verbose,
                width=stdout.width if stdout.is_terminal else REPORT_WIDTH,
                color=stdout.is_terminal and not stdout.no_color,
            )
        )
    error_code = outcome.report.get("error_code")
    if error_code and output == "json":
        detail = _ERROR_MESSAGES.get(error_code, f"falha de dependência ({error_code})")
        print(f"lab-agent: {detail}", file=sys.stderr)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

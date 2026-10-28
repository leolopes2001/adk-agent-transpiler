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
opens the run on stderr, with a transient progress indicator when both
streams are terminals. Colour is off with NO_COLOR or off a terminal. The
report goes to stdout. The JSON logs go to
stderr and, by default, only errors, so the terminal shows the review
questions, the report and the summary; --verbose (or LOG_LEVEL=INFO) shows
every run event.

Exit codes: 0 done, 2 invalid input, 3 blocked by the guard, 4 low OCR
quality, 5 no exam resolved, 6 dependency failure or scheduling refused,
7 stale generated code, 8 invalid configuration, 130 cancelled with Ctrl+C.
Everything the user reads on the terminal is in Portuguese
(`agent_runtime.messages`); codes, logs and the JSON report stay in English.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import signal
import sys
import threading
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, NoReturn

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
from agent_runtime.messages import (
    CANCELLED,
    REBUILD,
    argparse_error,
    config_error,
    image_error,
    line,
    run_error,
)
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
from agent_runtime.terminal import (
    Progress,
    ask_review,
    color_enabled,
    is_terminal,
    stderr_console,
    welcome,
)
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
EXIT_CANCELLED = 130
"""Ctrl+C, as shells report a process ended by SIGINT (128 + 2)."""
REPORT_WIDTH = 120
"""Width of the table report when stdout is not a terminal (a file, a pipe, a capture)."""

type ReviewPrompt = Callable[[Mapping[str, Any]], dict[str, Any]]
type Announce = Callable[["LoadedAgent", str], Progress | None]
"""Opens the run on the terminal; returns the progress indicator to drive, if any."""

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
    stages: tuple[tuple[str, str], ...] = ()
    """Long steps of the root workflow, in spec order, for the progress indicator."""


class _HelpFormatter(argparse.HelpFormatter):
    def add_usage(
        self, usage: str | None, actions: Any, groups: Any, prefix: str | None = None
    ) -> None:
        super().add_usage(usage, actions, groups, "uso: " if prefix is None else prefix)


class ArgumentParser(argparse.ArgumentParser):
    """argparse with its usage, help and errors in Portuguese."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", _HelpFormatter)
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        # argparse names its two default groups itself, in English.
        self._positionals.title = "argumentos"
        self._optionals.title = "opções"
        self.add_argument("-h", "--help", action="help", help="mostra esta ajuda e sai")

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_INVALID_INPUT, f"{self.prog}: erro: {argparse_error(message)}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(prog="lab-agent", description="Executa um agente gerado.")
    commands = parser.add_subparsers(
        dest="command", required=True, title="comandos", metavar="comando"
    )
    run = commands.add_parser(
        "run",
        help="processa a imagem de um pedido médico",
        description="Lê a imagem de um pedido médico, identifica os exames e pede o agendamento.",
    )
    run.add_argument(
        "--image", required=True, metavar="IMAGEM", help="PNG ou JPEG dentro de INPUT_ROOT"
    )
    run.add_argument(
        "--agent-dir",
        default="generated/exam_scheduling_agent",
        type=Path,
        metavar="PASTA",
        help="pacote gerado pelo lab-transpile (padrão: %(default)s)",
    )
    run.add_argument(
        "--spec",
        default="specs/exam_scheduling_agent.json",
        type=Path,
        metavar="ARQUIVO",
        help="spec do pacote (padrão: %(default)s)",
    )
    run.add_argument(
        "--non-interactive",
        action="store_true",
        help="não pergunta nada: itens que pedem revisão não são agendados",
    )
    run.add_argument(
        "--output",
        choices=("table", "json"),
        default=None,
        help="table (tabela para leitura) ou json (relatório completo); padrão: o da spec",
    )
    run.add_argument(
        "--run-id", default=None, metavar="UUID", help="identifica a execução nos logs"
    )
    run.add_argument(
        "--verbose",
        action="store_true",
        help="mostra a tabela técnica e os logs JSON da execução no stderr",
    )
    return parser


def _run_id(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise CliError(
            EXIT_INVALID_INPUT,
            "--run-id precisa ser um UUID. Omita a opção para gerar um automaticamente.",
        ) from None


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
            EXIT_STALE_CODE,
            f"o código gerado não pôde ser importado ({type(exc).__name__}). {REBUILD}",
        ) from None
    origin = Path(module.__file__ or "").resolve()
    if not origin.is_relative_to(agent_dir):
        raise CliError(
            EXIT_STALE_CODE,
            f"o módulo {agent_dir.name!r} foi carregado de outra pasta. "
            "Confira o --agent-dir e o PYTHONPATH.",
        )
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


STAGE_LABELS = (
    ("ocr_extract", "Lendo o pedido (OCR)"),
    ("llm_extract", "Interpretando o pedido com a IA"),
    ("rag_resolve", "Buscando os exames no catálogo"),
    ("lexical_extract", "Buscando os exames no catálogo"),
    ("schedule", "Registrando a solicitação"),
)


def _stage(nodes: list[Any]) -> tuple[str, str] | None:
    """(name of the node whose output ends the step, label) for the longest step in `nodes`.

    A model step ends with the output of its Agent, which is named after the agent.
    """
    by_kind = {n.get("kind"): n for n in nodes if isinstance(n, Mapping)}
    for kind, label in STAGE_LABELS:
        node = by_kind.get(kind)
        if node is not None:
            name = node.get("agent") if kind == "llm_extract" else node.get("id")
            return str(name), label
    return None


def progress_stages(spec: object) -> tuple[tuple[str, str], ...]:
    """Long steps of the root workflow in spec order; a subworkflow counts as
    its longest step (the model before the catalog)."""
    if not isinstance(spec, Mapping):
        return ()
    workflow = spec.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, Mapping) else None
    subworkflows = spec.get("subworkflows")
    stages: list[tuple[str, str]] = []
    for node in nodes if isinstance(nodes, list) else []:
        inner = [node]
        if isinstance(node, Mapping) and node.get("kind") == "subworkflow":
            sub = subworkflows.get(node.get("ref")) if isinstance(subworkflows, Mapping) else None
            sub_nodes = sub.get("nodes") if isinstance(sub, Mapping) else None
            inner = sub_nodes if isinstance(sub_nodes, list) else []
        stage = _stage(inner)
        if stage is not None:
            stages.append(stage)
    return tuple(stages)


def load_agent(agent_dir: Path, spec: Path, settings: AgentSettings) -> LoadedAgent:
    """Generated app after the manifest and location checks (exit 7 or 8 on failure)."""
    resolved = agent_dir.resolve()
    if not resolved.is_relative_to(settings.generated_root):
        raise CliError(
            EXIT_CONFIG,
            "--agent-dir precisa estar dentro de GENERATED_ROOT. "
            "Use o pacote gerado pelo lab-transpile.",
        )
    if not spec.is_file():
        raise CliError(EXIT_CONFIG, "arquivo da spec (--spec) não encontrado. Confira o caminho.")
    try:
        manifest = json.loads((resolved / MANIFEST).read_text("utf-8"))
    except (OSError, ValueError):
        raise CliError(
            EXIT_STALE_CODE, f"o código gerado não tem um manifesto legível. {REBUILD}"
        ) from None
    spec_sha256 = _sha256(spec)
    try:
        spec_data = json.loads(spec.read_text("utf-8"))
        output_format = _present_format(spec_data) or "table"
    except (OSError, ValueError):
        raise CliError(
            EXIT_CONFIG, "a spec (--spec) não é um JSON válido. Corrija o arquivo e rode de novo."
        ) from None
    if manifest.get("spec_sha256") != spec_sha256:
        raise CliError(
            EXIT_STALE_CODE,
            "o código gerado está desatualizado em relação à spec. "
            "Rode lab-transpile para gerá-lo de novo.",
        )
    module = _import(resolved)
    app = getattr(module, "app", None)
    if not isinstance(app, App):
        raise CliError(EXIT_STALE_CODE, f"o pacote gerado não tem um App. {REBUILD}")
    agent_module = sys.modules.get(f"{module.__name__}.agent")
    deps = getattr(agent_module, "deps", None)
    return LoadedAgent(
        name=app.name,
        description=getattr(app.root_agent, "description", "") or "",
        app=app,
        deps=deps if isinstance(deps, RuntimeDeps) else None,
        spec_sha256=spec_sha256,
        output_format=output_format,
        stages=progress_stages(spec_data),
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


_FIXED_FAILURES: tuple[tuple[type[BaseException], int, str], ...] = (
    (EOFError, EXIT_INVALID_INPUT, "INPUT_CLOSED"),
    (NodeTimeoutError, EXIT_DEPENDENCY, "NODE_TIMEOUT"),
)
"""Failures without a code of their own: stdin closed during a review, a node timeout."""


def _failure(exc: BaseException) -> tuple[int, str]:
    cause = _root_cause(exc)
    if isinstance(cause, InvalidInputError):
        return EXIT_INVALID_INPUT, cause.code
    if isinstance(cause, ConfigurationError):
        return EXIT_CONFIG, cause.code
    if isinstance(cause, AgentRuntimeError):
        return EXIT_DEPENDENCY, cause.code
    for kind, exit_code, code in _FIXED_FAILURES:
        if isinstance(cause, kind):
            return exit_code, code
    if isinstance(cause, genai_errors.APIError):
        status = (
            cause.status
            if isinstance(cause.status, str) and cause.status.isidentifier()
            else cause.code
        )
        return EXIT_DEPENDENCY, f"LLM_{status}"
    return EXIT_DEPENDENCY, "INTERNAL_ERROR"


async def _drive(  # noqa: PLR0913, PLR0917 - the run, plus what the operator sees
    runner: Runner,
    session_id: str,
    message: types.Content,
    trace: RunTrace,
    review: ReviewPrompt | None,
    progress: Progress | None = None,
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
                if progress is not None:
                    progress.finished(name)
        earlier |= this_round
        pending = _interrupt(events)
        if pending is None:
            return
        event, call = pending
        if review is None:
            raise CliError(EXIT_DEPENDENCY, line(run_error("INPUT_REQUIRED")))
        args = call.args or {}
        if progress is not None:
            progress.stop()
        answer = review(args.get("payload") or {})
        if progress is not None:
            progress.start()
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


def _checked_image(image: str, settings: AgentSettings) -> Path:
    """The image path relative to INPUT_ROOT, once the image is acceptable (exit 2)."""
    max_bytes = OcrExtractParams().max_image_bytes
    try:
        load_image(image, settings.input_root, max_bytes)
    except InvalidInputError as error:
        raise CliError(EXIT_INVALID_INPUT, line(image_error(error.code, max_bytes))) from None
    return (settings.input_root / image).resolve().relative_to(settings.input_root)


async def execute(  # noqa: PLR0913 - the CLI options
    *,
    image: str,
    agent_dir: Path,
    spec: Path,
    interactive: bool,
    run_id: str,
    settings: AgentSettings,
    review: ReviewPrompt | None = None,
    announce: Announce | None = None,
) -> RunOutcome:
    """Checks, runs and reports; raises CliError for failures before the run."""
    try:
        agent = load_agent(agent_dir, spec, settings)
    except ConfigurationError as error:
        raise CliError(EXIT_CONFIG, line(config_error(error.code, error.message))) from None
    absent = missing(required_env(agent))
    if absent:
        raise CliError(EXIT_CONFIG, line(config_error("MISSING_ENV", ", ".join(absent))))
    relative = _checked_image(image, settings)
    progress = announce(agent, image) if announce is not None else None

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
        if progress is not None:
            progress.start()
        prompt = (review or ask_review) if interactive else None
        await _drive(runner, session.id, message, trace, prompt, progress)
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
        if progress is not None:
            progress.stop()
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


def announcer(output: str | None, *, verbose: bool = False) -> Announce:
    """Welcome header and progress on stderr, only when the report will be a table.

    The progress indicator also needs both streams on a terminal, and no
    --verbose (its log lines would be drawn over).
    """

    def announce(agent: LoadedAgent, image: str) -> Progress | None:
        if (output or agent.output_format) != "table":
            return None
        console = stderr_console()
        welcome(console, agent.name, agent.description, image)
        if verbose or not (is_terminal(sys.stdout) and is_terminal(sys.stderr)):
            return None
        return Progress(console, agent.stages)

    return announce


def _raise_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


@contextmanager
def _interrupt_at_once() -> Iterator[None]:
    """Ctrl+C stops the run right away, also while a review prompt waits for input.

    `asyncio.run` installs its own SIGINT handler only when the default one is
    in place, and that handler's first Ctrl+C just cancels the main task: a
    prompt blocked in `input()` would keep waiting.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, _raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = AgentSettings.from_env()
        _configure_logging(log_level(settings, verbose=args.verbose))
        run_id = _run_id(args.run_id)
        structlog.contextvars.bind_contextvars(run_id=run_id)
        with _interrupt_at_once():
            outcome = asyncio.run(
                execute(
                    image=args.image,
                    agent_dir=args.agent_dir,
                    spec=args.spec,
                    interactive=not args.non_interactive,
                    run_id=run_id,
                    settings=settings,
                    announce=announcer(args.output, verbose=args.verbose),
                )
            )
    except ConfigurationError as error:
        print(f"lab-agent: {line(config_error(error.code, error.message))}", file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        print(f"\nlab-agent: {CANCELLED}", file=sys.stderr)
        return EXIT_CANCELLED
    except CliError as error:
        print(f"lab-agent: {error.message}", file=sys.stderr)
        return error.exit_code
    finally:
        structlog.contextvars.clear_contextvars()
    output = args.output or (outcome.result.format if outcome.result else "table")
    if output == "json":
        print(json.dumps(outcome.report, ensure_ascii=False, indent=2))
    else:
        on_terminal = is_terminal(sys.stdout)
        sys.stdout.write(
            render_text(
                outcome.report,
                outcome.result,
                verbose=args.verbose,
                width=Console().width if on_terminal else REPORT_WIDTH,
                color=color_enabled(sys.stdout),
            )
        )
    error_code = outcome.report.get("error_code")
    if error_code and output == "json":
        path = outcome.report.get("path") or []
        message = line(run_error(error_code, path[-1] if path else None))
        print(f"lab-agent: {message}", file=sys.stderr)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import os
import signal
import sys
import threading
import uuid
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType, MappingProxyType, ModuleType
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
from agent_runtime.nodes.present import REPORT_WIDTH, RESULT_STATE_KEY, RunResult
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
from lab_common import telemetry as lab_telemetry
from lab_common.mcp_server import LibraryLogFilter

SERVICE = "agent"
USER_ID = "lab-agent"
MANIFEST = "_manifest.json"
# Their log records may quote payloads, so LibraryLogFilter cuts their messages.
LIBRARY_LOGGERS = ("google_adk", "google.adk", "google_genai", "google.genai", "mcp", "httpx")
LLM_KEY_ENV = "GOOGLE_API_KEY"
# The warnings of a run (retries, injection, grounding) are in the report already.
QUIET_LOG_LEVEL = "ERROR"
VERBOSE_LOG_LEVEL = "INFO"
EXIT_CANCELLED = 130
RUN_SPAN = "lab-agent run"
# ADK and google-genai put prompts, document text and model answers on spans by default.
CONTENT_CAPTURE_OFF = MappingProxyType(
    {
        "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS": "false",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "false",
    }
)
STAGE_LABELS = (
    ("ocr_extract", "Lendo o pedido (OCR)"),
    ("llm_extract", "Interpretando o pedido com a IA"),
    ("rag_resolve", "Buscando os exames no catálogo"),
    ("lexical_extract", "Buscando os exames no catálogo"),
    ("schedule", "Registrando a solicitação"),
)

_PROG = "lab-agent"
_FAILURES_WITHOUT_CODE: tuple[tuple[type[BaseException], int, str], ...] = (
    (EOFError, EXIT_INVALID_INPUT, "INPUT_CLOSED"),
    (NodeTimeoutError, EXIT_DEPENDENCY, "NODE_TIMEOUT"),
)

type ReviewPrompt = Callable[[Mapping[str, Any]], dict[str, Any]]
type Announce = Callable[["LoadedAgent", str], Progress | None]

log = lab_logging.get_logger(__name__)
tracer = lab_telemetry.tracer(__name__)


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
    stages: tuple[tuple[str, str], ...] = ()


@dataclass
class RunOutcome:
    exit_code: int
    report: dict[str, Any]
    result: RunResult | None


class _HelpFormatter(argparse.HelpFormatter):
    def add_usage(
        self, usage: str | None, actions: Any, groups: Any, prefix: str | None = None
    ) -> None:
        super().add_usage(usage, actions, groups, "uso: " if prefix is None else prefix)


class ArgumentParser(argparse.ArgumentParser):
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
    parser = ArgumentParser(prog=_PROG, description="Executa um agente gerado.")
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
        default="agent_specs/exam_scheduling_agent.json",
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


def progress_stages(spec: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(spec, Mapping):
        return ()

    workflow = spec.get("workflow")
    nodes = workflow.get("nodes") if isinstance(workflow, Mapping) else None
    if not isinstance(nodes, list):
        return ()

    subworkflows = spec.get("subworkflows")
    stages = (_longest_stage(_step_nodes(node, subworkflows)) for node in nodes)
    return tuple(stage for stage in stages if stage is not None)


def load_agent(agent_dir: Path, spec: Path, settings: AgentSettings) -> LoadedAgent:
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

    spec_sha256 = _file_sha256(spec)
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

    module = _import_generated_package(resolved)
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


def required_env(agent: LoadedAgent) -> list[str]:
    names = agent.deps.required_env() if agent.deps else []
    root = agent.app.root_agent
    uses_gemini = isinstance(root, BaseNode) and any(
        isinstance(node, Agent) and isinstance(node.model, str) for node in _walk_nodes(root)
    )
    return sorted({*names, *([LLM_KEY_ENV] if uses_gemini else [])})


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
    os.environ.update(CONTENT_CAPTURE_OFF)
    with tracer.start_as_current_span(
        RUN_SPAN, attributes={"lab.run_id": run_id}, record_exception=False
    ) as span:
        try:
            outcome = await _execute(
                image=image,
                agent_dir=agent_dir,
                spec=spec,
                interactive=interactive,
                run_id=run_id,
                settings=settings,
                review=review,
                announce=announce,
            )
        except CliError as error:
            lab_telemetry.annotate(exit_code=error.exit_code)
            lab_telemetry.fail("NOT_STARTED", span)
            raise

        report = outcome.report
        lab_telemetry.annotate(
            outcome=report["outcome"],
            exit_code=outcome.exit_code,
            error_code=report.get("error_code"),
        )
        if outcome.exit_code != 0:
            span.set_status(lab_telemetry.Status(lab_telemetry.StatusCode.ERROR))

        return outcome


def log_level(settings: AgentSettings, *, verbose: bool) -> str:
    if verbose:
        return "DEBUG" if settings.log_level == "DEBUG" else VERBOSE_LOG_LEVEL

    return settings.log_level or QUIET_LOG_LEVEL


def announcer(output: str | None, *, verbose: bool = False) -> Announce:
    def announce(agent: LoadedAgent, image: str) -> Progress | None:
        if (output or agent.output_format) != "table":
            return None

        console = stderr_console()
        welcome(console, agent.name, agent.description, image)
        # The spinner needs both streams on a terminal, and --verbose log lines would cross it.
        if verbose or not (is_terminal(sys.stdout) and is_terminal(sys.stderr)):
            return None

        return Progress(console, agent.stages)

    return announce


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    telemetry: lab_telemetry.Telemetry | None = None
    try:
        settings = AgentSettings.from_env()
        run_id = _parse_run_id(args.run_id)
        telemetry = lab_telemetry.setup(SERVICE, instance_id=run_id)
        _configure_logging(
            log_level(settings, verbose=args.verbose),
            export=telemetry.log_export if telemetry is not None else None,
        )
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
        _print_error(line(config_error(error.code, error.message)))
        return EXIT_CONFIG
    except KeyboardInterrupt:
        print(file=sys.stderr)
        _print_error(CANCELLED)
        return EXIT_CANCELLED
    except CliError as error:
        _print_error(error.message)
        return error.exit_code
    finally:
        structlog.contextvars.clear_contextvars()
        if telemetry is not None:
            telemetry.shutdown()

    _print_outcome(outcome, output=args.output, verbose=args.verbose)
    return outcome.exit_code


def _parse_run_id(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())

    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise CliError(
            EXIT_INVALID_INPUT,
            "--run-id precisa ser um UUID. Omita a opção para gerar um automaticamente.",
        ) from None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_generated_package(agent_dir: Path) -> ModuleType:
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


def _longest_stage(nodes: list[Any]) -> tuple[str, str] | None:
    by_kind = {n.get("kind"): n for n in nodes if isinstance(n, Mapping)}
    for kind, label in STAGE_LABELS:
        node = by_kind.get(kind)
        if node is not None:
            # A model step ends with the output of its Agent, which is named after the agent.
            name = node.get("agent") if kind == "llm_extract" else node.get("id")
            return str(name), label

    return None


def _step_nodes(node: object, subworkflows: object) -> list[Any]:
    if not (isinstance(node, Mapping) and node.get("kind") == "subworkflow"):
        return [node]

    sub = subworkflows.get(node.get("ref")) if isinstance(subworkflows, Mapping) else None
    sub_nodes = sub.get("nodes") if isinstance(sub, Mapping) else None
    return sub_nodes if isinstance(sub_nodes, list) else []


def _walk_nodes(node: BaseNode) -> Iterator[BaseNode]:
    yield node
    if isinstance(node, Workflow) and node.graph is not None:
        seen: set[str] = set()
        for edge in node.graph.edges:
            for child in (edge.from_node, edge.to_node):
                if isinstance(child, BaseNode) and child.name not in seen:
                    seen.add(child.name)
                    yield from _walk_nodes(child)


def _event_key(event: Event) -> str:
    if not event.node_info.path:
        return ""

    # A replay may carry the dump of a model the original event held as is, and loses the route.
    value = event.output
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    output = json.dumps(value, sort_keys=True, default=str)
    return f"{event.invocation_id}|{event.node_info.path}|{event.error_code}|{output}"


def _pending_review(events: list[Event]) -> tuple[Event, types.FunctionCall] | None:
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


def _exit_and_error_code(exc: BaseException) -> tuple[int, str]:
    cause = _root_cause(exc)
    if isinstance(cause, InvalidInputError):
        return EXIT_INVALID_INPUT, cause.code

    if isinstance(cause, ConfigurationError):
        return EXIT_CONFIG, cause.code

    if isinstance(cause, AgentRuntimeError):
        return EXIT_DEPENDENCY, cause.code

    for kind, exit_code, code in _FAILURES_WITHOUT_CODE:
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


# A resumed invocation yields the finished nodes' events again, with new ids (ADR-0009).
async def _run_answering_reviews(  # noqa: PLR0913, PLR0917 - the run, plus what the operator sees
    runner: Runner,
    session_id: str,
    message: types.Content,
    trace: RunTrace,
    review: ReviewPrompt | None,
    progress: Progress | None = None,
) -> None:
    invocation_id: str | None = None
    earlier: set[str] = set()
    while True:
        events, keys = await _new_events(
            runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                invocation_id=invocation_id,
                new_message=message,
            ),
            earlier,
            trace,
            progress,
        )
        earlier |= keys
        pending = _pending_review(events)
        if pending is None:
            return

        event, call = pending
        if review is None:
            raise CliError(EXIT_DEPENDENCY, line(run_error("INPUT_REQUIRED")))

        answer = _ask_with_progress_paused(review, call, progress)
        invocation_id = event.invocation_id
        message = _review_answer(call, answer)


async def _new_events(
    events: AsyncIterator[Event],
    earlier: set[str],
    trace: RunTrace,
    progress: Progress | None,
) -> tuple[list[Event], set[str]]:
    kept: list[Event] = []
    keys_seen: set[str] = set()
    async for event in events:
        keys = {event.id, _event_key(event)} - {""}
        if keys & earlier:
            continue

        keys_seen |= keys
        trace.observe(event)
        kept.append(event)
        _log_node_event(event, progress)

    return kept, keys_seen


def _log_node_event(event: Event, progress: Progress | None) -> None:
    name = event.node_info.name
    if event.error_code:
        log.warning("node_retry", node=name, error_code=event.error_code)
        return

    if event.output is not None:
        log.info("node_finished", node=name, route=event.actions.route)
        if progress is not None:
            progress.finished(name)


def _ask_with_progress_paused(
    review: ReviewPrompt, call: types.FunctionCall, progress: Progress | None
) -> dict[str, Any]:
    args = call.args or {}
    if progress is not None:
        progress.stop()
    answer = review(args.get("payload") or {})
    if progress is not None:
        progress.start()

    return answer


def _review_answer(call: types.FunctionCall, answer: dict[str, Any]) -> types.Content:
    response = types.FunctionResponse(id=call.id, name=call.name, response=answer)
    return types.Content(role="user", parts=[types.Part(function_response=response)])


def _checked_image(image: str, settings: AgentSettings) -> Path:
    max_bytes = OcrExtractParams().max_image_bytes
    try:
        load_image(image, settings.input_root, max_bytes)
    except InvalidInputError as error:
        raise CliError(EXIT_INVALID_INPUT, line(image_error(error.code, max_bytes))) from None

    return (settings.input_root / image).resolve().relative_to(settings.input_root)


async def _execute(  # noqa: PLR0913 - the CLI options
    *,
    image: str,
    agent_dir: Path,
    spec: Path,
    interactive: bool,
    run_id: str,
    settings: AgentSettings,
    review: ReviewPrompt | None,
    announce: Announce | None,
) -> RunOutcome:
    agent, image_path = _preflight(agent_dir, spec, image, settings)
    progress = announce(agent, image) if announce is not None else None

    trace = RunTrace()
    runner, service, session_id = await _open_session(agent, interactive=interactive, run_id=run_id)
    message = _run_message(image_path)
    log.info("run_started", agent=agent.name, interactive=interactive)
    result: RunResult | None = None
    error_code: str | None = None
    try:
        if progress is not None:
            progress.start()
        prompt = (review or ask_review) if interactive else None
        await _run_answering_reviews(runner, session_id, message, trace, prompt, progress)
        exit_code, error_code, result = await _final_result(service, agent.app.name, session_id)
    except CliError as error:
        exit_code, error_code = error.exit_code, "INPUT_REQUIRED"
    except Exception as exc:  # noqa: BLE001 - every failure becomes an exit code
        exit_code, error_code = _exit_and_error_code(exc)
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


def _preflight(
    agent_dir: Path, spec: Path, image: str, settings: AgentSettings
) -> tuple[LoadedAgent, Path]:
    try:
        agent = load_agent(agent_dir, spec, settings)
    except ConfigurationError as error:
        raise CliError(EXIT_CONFIG, line(config_error(error.code, error.message))) from None

    absent = missing(required_env(agent))
    if absent:
        raise CliError(EXIT_CONFIG, line(config_error("MISSING_ENV", ", ".join(absent))))

    return agent, _checked_image(image, settings)


async def _open_session(
    agent: LoadedAgent, *, interactive: bool, run_id: str
) -> tuple[Runner, InMemorySessionService, str]:
    service = InMemorySessionService()
    runner = Runner(app=agent.app, session_service=service)
    session = await service.create_session(
        app_name=agent.app.name,
        user_id=USER_ID,
        state={"run.interactive": interactive, "run.id": run_id},
    )
    return runner, service, session.id


def _run_message(image_path: Path) -> types.Content:
    text = RunInput(image_path=str(image_path)).model_dump_json()
    return types.Content(role="user", parts=[types.Part(text=text)])


async def _final_result(
    service: InMemorySessionService, app_name: str, session_id: str
) -> tuple[int, str | None, RunResult | None]:
    final = await service.get_session(app_name=app_name, user_id=USER_ID, session_id=session_id)
    raw = final.state.get(RESULT_STATE_KEY) if final else None
    if raw is None:
        return EXIT_DEPENDENCY, "NO_RESULT", None

    result = RunResult.model_validate(raw)
    return exit_code_for(result), None, result


def _configure_logging(level: str, export: lab_logging.LogExport | None = None) -> None:
    lab_logging.configure(
        SERVICE, level, filters=[LibraryLogFilter(LIBRARY_LOGGERS)], export=export
    )
    # One INFO line per HTTP request (MCP keeps SSE streams busy) says nothing about the run.
    logging.getLogger("httpx").setLevel(max(logging.WARNING, logging.getLogger().level))
    # ADK warns about experimental features on every import; they are not run events.
    warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")


def _raise_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


# asyncio.run's own SIGINT handler only cancels the main task: a prompt in input() would wait.
@contextmanager
def _interrupt_at_once() -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous = signal.signal(signal.SIGINT, _raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _print_outcome(outcome: RunOutcome, *, output: str | None, verbose: bool) -> None:
    chosen = output or (outcome.result.format if outcome.result else "table")
    if chosen != "json":
        on_terminal = is_terminal(sys.stdout)
        sys.stdout.write(
            render_text(
                outcome.report,
                outcome.result,
                verbose=verbose,
                width=Console().width if on_terminal else REPORT_WIDTH,
                color=color_enabled(sys.stdout),
            )
        )
        return

    print(json.dumps(outcome.report, ensure_ascii=False, indent=2))
    error_code = outcome.report.get("error_code")
    if error_code:
        path = outcome.report.get("path") or []
        _print_error(line(run_error(error_code, path[-1] if path else None)))


def _print_error(message: str) -> None:
    print(f"{_PROG}: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

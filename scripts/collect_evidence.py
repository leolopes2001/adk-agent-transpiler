"""Collect the evidence of a working delivery into evidence/.

Usage:
    uv run python scripts/collect_evidence.py [--skip-coverage] [--fake-llm]

Everything is gathered from the host: the containers have read-only file
systems and never write to the repository. The script

- runs the transpiler (a spec with errors, generation of both specs, the diff
  between them) and exports the OpenAPI document and the RAG calibration;
- measures test coverage (every test except e2e; needs Docker and Tesseract);
- brings up the Compose stack as project `lab-agent-e2e` (the e2e override
  only adds a random API port and the fake model, which is switched back to
  Gemini when a GOOGLE_API_KEY is found in the environment or in .env), runs
  the scenarios, saves the JSON reports, the CLI output, the API's view of
  the appointment, the database rows, the logs of every container and the
  generated code, then removes the stack and its volumes.

The folders it writes (runs/, cli/, logs/, api/, transpiler/) are replaced on
every run. Logs are already redacted by the services; the security test
`tests/security/test_e2e_pii.py` scans the whole folder for personal data.
A Gemini quota error (429/402) is recorded as pending; the script does not retry.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from e2e.helpers import (  # noqa: E402 - the e2e helpers live under tests/
    FAKE_LLM_FIXTURES,
    LLM_KEY_ENV,
    LOOKUP_AGENT_DIR,
    LOOKUP_SPEC,
    MAIN_SPEC,
    SAMPLES,
    TEST_IMAGES,
    AgentRun,
    ComposeStack,
    compose_stack,
    google_api_key,
)

EVIDENCE = REPO_ROOT / "evidence"
OWN_DIRS = ("runs", "cli", "logs", "api", "transpiler")
SPECS = {
    "exam_scheduling_agent": REPO_ROOT / "specs" / "exam_scheduling_agent.json",
    "exam_lookup_only": REPO_ROOT / "specs" / "examples" / "exam_lookup_only.json",
}
LLM_QUOTA_ERRORS = {"LLM_RESOURCE_EXHAUSTED", "LLM_429", "LLM_402"}
NON_INTERACTIVE = ("--non-interactive",)
DB_QUERY = (
    "SELECT a.id, a.patient_ref, a.source_document_sha256, a.requested_by, "
    "string_agg(e.code, ',' ORDER BY e.code) FROM appointments a "
    "JOIN appointment_exams e ON e.appointment_id = a.id GROUP BY a.id ORDER BY a.id"
)


def _run(*command: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed commands built here
        command, cwd=cwd, capture_output=True, text=True, check=False
    )


def _write(relative: str, text: str) -> Path:
    path = EVIDENCE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", "utf-8")
    return path


def _transcript(command: str, result: subprocess.CompletedProcess[str]) -> str:
    return f"$ {command}\n{result.stdout}{result.stderr}(exit {result.returncode})\n"


# --- transpiler, OpenAPI, calibration, coverage ---------------------------------


def _invalid_spec(target: Path) -> Path:
    """The main spec with three mistakes: a tool outside the allowlist, an undeclared
    HTTP service and an edge that reaches `schedule` without the confidence gate."""
    spec = json.loads(SPECS["exam_scheduling_agent"].read_text("utf-8"))
    nodes = {node["id"]: node for node in spec["workflow"]["nodes"]}
    nodes["ocr"]["tool"] = "read_file"
    nodes["schedule"]["service"] = "payments"
    spec["workflow"]["edges"].append({"from": "guard", "to": "schedule", "route": "clean"})
    shutil.copytree(REPO_ROOT / "specs" / "instructions", target / "instructions")
    path = target / "exam_scheduling_agent_invalid.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), "utf-8")
    return path


def transpiler_evidence() -> None:
    transpile = shutil.which("lab-transpile") or "lab-transpile"
    # lab-transpile only writes inside the repository: a temporary folder there.
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".evidence-") as tmp:
        work = Path(tmp)
        invalid = _invalid_spec(work)
        result = _run(transpile, "validate", str(invalid))
        _write(
            "transpiler/validate-errors.txt",
            _transcript("lab-transpile validate exam_scheduling_agent_invalid.json", result),
        )
        shown = json.dumps(json.loads(invalid.read_text("utf-8")), ensure_ascii=False, indent=2)
        _write("transpiler/exam_scheduling_agent_invalid.json", shown)

        out = work / "generated"
        transcripts = []
        for spec in SPECS.values():
            relative = spec.relative_to(REPO_ROOT)
            result = _run(transpile, "generate", str(relative), "--out", str(out))
            result.stdout = result.stdout.replace(f"{work.name}/", "")
            transcripts.append(
                _transcript(f"lab-transpile generate {relative} --out generated/", result)
            )
        _write("transpiler/generate.txt", "\n".join(transcripts))

        spec_lines = {
            name: (json.dumps(json.loads(path.read_text("utf-8")), indent=2) + "\n").splitlines(
                keepends=True
            )
            for name, path in SPECS.items()
        }
        code_lines = {
            name: (out / name / "agent.py").read_text("utf-8").splitlines(keepends=True)
            for name in SPECS
        }
        main, lookup = SPECS
        _write(
            "transpiler/diff-specs.diff",
            "".join(
                difflib.unified_diff(
                    spec_lines[main],
                    spec_lines[lookup],
                    f"specs/{main}.json",
                    f"specs/examples/{lookup}.json",
                )
            ),
        )
        _write(
            "transpiler/diff-generated.diff",
            "".join(
                difflib.unified_diff(
                    code_lines[main],
                    code_lines[lookup],
                    f"generated/{main}/agent.py",
                    f"generated/{lookup}/agent.py",
                )
            ),
        )


def openapi_and_calibration() -> None:
    for command in (
        (sys.executable, "-m", "scheduler_api.openapi", "evidence/openapi.json"),
        (
            sys.executable,
            "scripts/calibrate_rag.py",
            "--eval",
            "tests/data/rag_eval.jsonl",
            "--out",
            "evidence/rag_calibration.md",
        ),
    ):
        result = _run(*command)
        if result.returncode != 0:
            raise SystemExit(f"{' '.join(command[1:])} failed:\n{result.stderr}")


def coverage_evidence() -> None:
    command = (
        sys.executable, "-m", "pytest", "-m", "not e2e and not llm", "-q",
        "--cov", "--cov-report=term",
    )  # fmt: skip
    result = _run(*command)
    lines = result.stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if "coverage:" in line), len(lines))
    summary = lines[-1] if lines else ""
    body = "\n".join(lines[start:])
    _write(
        "coverage.txt",
        f'$ uv run pytest -m "not e2e and not llm" --cov --cov-report=term\n'
        f"{body}\n(exit {result.returncode})",
    )
    if result.returncode != 0:
        raise SystemExit(f"tests failed during the coverage run:\n{summary}")


# --- Compose stack ----------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    title: str
    image: str
    args: tuple[str, ...] = NON_INTERACTIVE
    fake_fixture: str = "pedido_medico_ficticio.json"
    uses_llm: bool = True


REPORT_SCENARIOS = [
    Scenario("feliz", "Fluxo feliz", f"{SAMPLES}/pedido_medico_ficticio.png"),
    Scenario("replay", "Reexecução (replay idempotente)", f"{SAMPLES}/pedido_medico_ficticio.png"),
    Scenario("injection", "Injeção de prompt", f"{SAMPLES}/pedido_injecao.png", uses_llm=False),
    Scenario(
        "ambiguo",
        "Pedido ambíguo, não interativo",
        f"{SAMPLES}/pedido_ambiguo.png",
        fake_fixture="pedido_ambiguo.json",
    ),
]
TEXT_ONLY_SCENARIOS = [
    Scenario("ilegivel", "Imagem ilegível", f"{TEST_IMAGES}/blank.png", uses_llm=False),
    Scenario(
        "spec-alternativa",
        "Spec alternativa (exam_lookup_only)",
        f"{SAMPLES}/pedido_medico_ficticio.png",
        args=("--agent-dir", LOOKUP_AGENT_DIR, "--spec", LOOKUP_SPEC, *NON_INTERACTIVE),
        uses_llm=False,
    ),
]


@dataclass
class Collector:
    stack: ComposeStack
    llm: bool
    quota_error: str | None = None
    index: list[dict[str, Any]] = field(default_factory=list)
    agent_logs: list[str] = field(default_factory=list)
    stale_exit: int | None = None
    stale_file: str = ""

    def run(self, scenario: Scenario, *output: str) -> AgentRun | None:
        backend = "gemini" if self.llm and scenario.uses_llm else "fake"
        if scenario.uses_llm and self.llm and self.quota_error:
            self.index.append(
                {"scenario": scenario.title, "output": output, "pending": self.quota_error}
            )
            return None
        env = {"FAKE_LLM_FIXTURE": f"{FAKE_LLM_FIXTURES}/{scenario.fake_fixture}"}
        pass_env: tuple[str, ...] = ()
        if self.llm:
            env |= {"APP_ENV": "dev", "AGENT_LLM_BACKEND": "gemini"}
            pass_env = (LLM_KEY_ENV,)
        run = self.stack.run_agent(
            scenario.image, *scenario.args, *output, env=env, pass_env=pass_env
        )
        error = (run.report or {}).get("error_code")
        if error in LLM_QUOTA_ERRORS:
            self.quota_error = error
        self.agent_logs.append(run.stderr)
        self.index.append(
            {
                "scenario": scenario.title,
                "name": scenario.name,
                "output": output,
                "backend": backend if scenario.uses_llm else "sem LLM",
                "run": run,
            }
        )
        return run


def _cli_text(scenario: Scenario, run: AgentRun) -> str:
    """The command inside the container, stdout and the CLI's own stderr messages
    (its JSON logs go to logs/compose.log)."""
    messages = [line for line in run.stderr.splitlines() if line.startswith("lab-agent:")]
    command = " ".join(("lab-agent", *run.args))
    body = run.stdout + "".join(f"{line}\n" for line in messages)
    return f"# {scenario.title}\n$ {command}\n{body}(exit {run.exit_code})\n"


def _stale_spec(stack: ComposeStack) -> tuple[int, str]:
    script = (
        "import json, sys\n"
        "spec = json.load(open(sys.argv[1]))\n"
        "for node in spec['workflow']['nodes']:\n"
        "    if node['kind'] == 'confidence_gate':\n"
        "        node['params']['accept_threshold'] = 0.95\n"
        "json.dump(spec, open('/tmp/spec.json', 'w'))\n"
    )
    image = f"{SAMPLES}/pedido_medico_ficticio.png"
    shell = (
        f'python -c "$0" {MAIN_SPEC} && exec lab-agent run --image {image} '
        "--spec /tmp/spec.json --non-interactive"
    )
    result = stack.compose(
        "run", "--rm", "-T", "--entrypoint", "sh", "agent", "-c", shell, script, check=False
    )
    messages = [line for line in result.stderr.splitlines() if line.startswith("lab-agent:")]
    return result.returncode, (
        "# Spec alterada sem regenerar (accept_threshold 0.85 -> 0.95 numa cópia em /tmp)\n"
        f"$ lab-agent run --image {image} --spec /tmp/spec.json --non-interactive\n"
        + "".join(f"{line}\n" for line in messages)
        + f"(exit {result.returncode})\n"
    )


def _index_markdown(collector: Collector) -> str:
    lines = [
        "# Execuções coletadas",
        "",
        "Gerado por `scripts/collect_evidence.py`; não editar à mão.",
        "",
        "Primeiro as execuções com `--output json` (relatórios em `runs/`); depois o banco "
        "é esvaziado (`TRUNCATE appointments CASCADE`) e os mesmos cenários rodam com a "
        "saída em tabela (`cli/`), para que as duas séries comecem com um agendamento novo.",
        "",
        "| Cenário | Saída | LLM | Exit | Outcome | Agendamento | Arquivo |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in collector.index:
        output = " ".join(entry["output"]) or "padrão da spec"
        if "pending" in entry:
            lines.append(
                f"| {entry['scenario']} | `{output}` | gemini | — | pendente: cota do Gemini "
                f"({entry['pending']}) | — | — |"
            )
            continue
        run: AgentRun = entry["run"]
        if run.report is not None:
            outcome = run.report["outcome"]
            appointment = (run.report.get("appointment") or {}).get("status", "—")
        else:
            outcome, appointment = _from_text(run.stdout)
        lines.append(
            f"| {entry['scenario']} | `{output}` | {entry['backend']} | {run.exit_code} | "
            f"{outcome} | {appointment} | {entry.get('file', '—')} |"
        )
    lines.append(
        f"| Spec alterada sem regenerar | — | — | {collector.stale_exit} | — | — | "
        f"[`{collector.stale_file}`](../{collector.stale_file}) |"
    )
    return "\n".join(lines) + "\n"


def _from_text(stdout: str) -> tuple[str, str]:
    """Outcome and appointment line of a table report."""
    outcomes = re.findall(r": (\S+) \(exit \d+\)", stdout)
    appointment = re.search(r"^Agendamento: ([^—\n]+)", stdout, re.MULTILINE)
    return (
        outcomes[-1] if outcomes else "—",
        appointment.group(1).strip() if appointment else "—",
    )


def stack_evidence(use_llm: bool) -> str | None:
    """Runs the scenarios; returns the Gemini quota error code, if any."""
    with compose_stack() as stack:
        collector = Collector(stack, llm=use_llm)

        for scenario in REPORT_SCENARIOS:
            run = collector.run(scenario, "--output", "json")
            if run is not None and run.report is not None:
                path = _write(
                    f"runs/{run.report['run_id']}.json",
                    json.dumps(run.report, ensure_ascii=False, indent=2),
                )
                collector.index[-1]["file"] = f"[`runs/{path.name}`](./{path.name})"

        stack.db_rows("TRUNCATE appointments CASCADE")
        appointment_id = None
        for number, scenario in enumerate([*REPORT_SCENARIOS, *TEXT_ONLY_SCENARIOS], start=1):
            # The lookup-only spec keeps its own output format.
            output = () if scenario.name == "spec-alternativa" else ("--output", "table")
            run = collector.run(scenario, *output)
            if run is None:
                continue
            name = f"cli/{number:02d}-{scenario.name}.txt"
            _write(name, _cli_text(scenario, run))
            collector.index[-1]["file"] = f"[`{name}`](../{name})"
            match = re.search(r"\bid (apt_[0-9A-Z]{26})", run.stdout)
            if scenario.name == "feliz" and match:
                appointment_id = match.group(1)
        stale = len(REPORT_SCENARIOS) + len(TEXT_ONLY_SCENARIOS) + 1
        collector.stale_file = f"cli/{stale:02d}-spec-alterada.txt"
        collector.stale_exit, text = _stale_spec(stack)
        _write(collector.stale_file, text)

        if appointment_id:
            status, body = stack.api_get(f"/v1/appointments/{appointment_id}")
            _write(
                "api/appointment.json",
                json.dumps({"status": status, "body": body}, ensure_ascii=False, indent=2),
            )
        rows = [" | ".join(row) for row in stack.db_rows(DB_QUERY)]
        header = "id | patient_ref | source_document_sha256 | requested_by | exams"
        _write("api/database.txt", "\n".join([f'$ psql -At -c "{DB_QUERY}"', header, *rows]))

        for package in SPECS:
            code = stack.compose(
                "run", "--rm", "-T", "--entrypoint", "cat", "agent",
                f"/app/generated/{package}/agent.py",
            ).stdout  # fmt: skip
            _write(f"transpiler/{package}.agent.py.txt", code)

        agent = "".join(
            f"agent-run-{number:02d} | {line}\n"
            for number, logs in enumerate(collector.agent_logs, start=1)
            for line in logs.splitlines()
            if line.startswith("{")
        )
        _write("logs/compose.log", stack.logs() + agent)
        _write("runs/index.md", _index_markdown(collector))
        return collector.quota_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-coverage", action="store_true", help="keep evidence/coverage.txt")
    parser.add_argument("--fake-llm", action="store_true", help="never call Gemini")
    args = parser.parse_args(argv)

    key = "" if args.fake_llm else google_api_key()
    if key:
        os.environ[LLM_KEY_ENV] = key
    else:
        print("no GOOGLE_API_KEY: the scenarios run with the fake model", file=sys.stderr)

    for name in OWN_DIRS:
        shutil.rmtree(EVIDENCE / name, ignore_errors=True)
    transpiler_evidence()
    openapi_and_calibration()
    if not args.skip_coverage:
        coverage_evidence()
    quota = stack_evidence(use_llm=bool(key))
    if quota:
        print(f"Gemini quota error ({quota}): LLM scenarios left pending", file=sys.stderr)
    print(f"evidence written to {EVIDENCE.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

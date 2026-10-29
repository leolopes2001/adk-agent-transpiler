import dataclasses
import hashlib
from typing import Any

import pytest

from transpiler.ir.builder import build_program
from transpiler.ir.model import ProgramIR, RetryIR
from transpiler.registry import NodeKind
from transpiler.spec.loader import InstructionFile, load_spec, parse_spec, read_instruction

pytestmark = pytest.mark.unit

type SpecDict = dict[str, Any]

INSTRUCTION = InstructionFile("instructions/extractor.md", "Extraia exames.\n", "ab" * 32)


def _build(
    spec: SpecDict, instruction: InstructionFile = INSTRUCTION, sha: str = "0" * 64
) -> ProgramIR:
    return build_program(
        parse_spec(spec),
        spec_sha256=sha,
        instructions={name: instruction for name in spec["llm_agents"]},
    )


@pytest.fixture
def program(spec_dict: SpecDict) -> ProgramIR:
    return _build(spec_dict)


def test_program_header(program: ProgramIR) -> None:
    assert (program.agent_name, program.version) == ("full_agent", "0.1.0")
    assert program.root.name == "full_agent"
    assert program.spec_sha256 == "0" * 64


def test_nodes_and_edges_keep_spec_order(program: ProgramIR) -> None:
    assert [n.id for n in program.root.nodes] == [
        "ocr",
        "guard",
        "extraction",
        "resolve",
        "gate",
        "review",
        "schedule",
        "present",
        "abort",
    ]
    assert program.root.edges[0].source == "START"
    assert (program.root.edges[7].target, program.root.edges[7].route) == ("schedule", "accepted")


def test_kinds_are_enums(program: ProgramIR) -> None:
    assert program.root.node("gate").kind is NodeKind.CONFIDENCE_GATE


def test_explicit_retry_and_timeout(program: ProgramIR) -> None:
    ocr = program.root.node("ocr")

    assert ocr.retry == RetryIR(max_attempts=2, initial_delay_s=0.5, max_delay_s=2.0)
    assert ocr.timeout_s == 45.0


def test_policy_defaults_fill_io_nodes(program: ProgramIR) -> None:
    resolve = program.root.node("resolve")

    assert resolve.retry == RetryIR(max_attempts=1, initial_delay_s=0.5, max_delay_s=2.0)
    assert resolve.timeout_s == 20.0


def test_non_io_nodes_have_no_retry(program: ProgramIR) -> None:
    guard = program.root.node("guard")

    assert (guard.retry, guard.timeout_s) == (None, None)


def test_param_defaults(program: ProgramIR) -> None:
    assert dict(program.root.node("guard").params) == {"max_chars": 8000, "block_at": "high"}
    assert dict(program.root.node("resolve").params) == {"top_k": 5}
    assert program.root.node("ocr").params["min_mean_confidence"] == 0.5
    assert dict(program.root.node("abort").params) == {}


def test_references_are_resolved(program: ProgramIR) -> None:
    ocr, schedule = program.root.node("ocr"), program.root.node("schedule")

    assert ocr.server is not None
    assert (ocr.server.key, ocr.server.url_env, ocr.tool) == ("ocr", "OCR_MCP_URL", "extract_text")
    assert ocr.server.allowed_tools == ("extract_text",)
    assert schedule.service is not None
    assert schedule.service.api_key_env == "SCHEDULER_API_KEY"


def test_subworkflow_and_agent(program: ProgramIR) -> None:
    extraction = program.root.node("extraction").subgraph

    assert extraction is not None
    assert extraction.name == "extraction"
    agent = extraction.node("llm_extract").agent
    assert agent is not None
    assert (agent.key, agent.output_schema, agent.instruction) == (
        "extractor_agent",
        "ExtractedExams",
        "Extraia exames.\n",
    )
    assert (agent.model.name, agent.model.temperature, agent.model.max_output_tokens) == (
        "gemini-flash-latest",
        0.0,
        1024,
    )


def test_only_used_servers_and_services(spec_dict: SpecDict) -> None:
    spec_dict["mcp_servers"]["spare"] = dict(spec_dict["mcp_servers"]["rag"])
    program = _build(spec_dict)

    assert [s.key for s in program.servers] == ["ocr", "rag"]
    assert [s.key for s in program.services] == ["scheduler"]


def test_resumability_follows_human_review(spec_dict: SpecDict, program: ProgramIR) -> None:
    assert program.needs_resumability is True

    spec_dict["workflow"]["nodes"][5] = {"id": "review", "kind": "confidence_gate", "params": {}}
    spec_dict["workflow"]["nodes"][5]["params"] = spec_dict["workflow"]["nodes"][4]["params"]
    assert _build(spec_dict).needs_resumability is False


def test_ir_is_immutable(program: ProgramIR) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        program.agent_name = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        program.root.node("guard").params["max_chars"] = 1  # type: ignore[index]


def test_digest_is_stable(spec_dict: SpecDict, program: ProgramIR) -> None:
    again = _build(spec_dict)

    assert program.digest() == again.digest()
    assert len(program.digest()) == 64


def test_digest_ignores_spec_file_hash(spec_dict: SpecDict, program: ProgramIR) -> None:
    assert _build(spec_dict, sha="f" * 64).digest() == program.digest()


def test_digest_tracks_semantics(spec_dict: SpecDict, program: ProgramIR) -> None:
    spec_dict["workflow"]["nodes"][4]["params"]["accept_threshold"] = 0.9

    assert _build(spec_dict).digest() != program.digest()


def test_digest_tracks_instruction(spec_dict: SpecDict, program: ProgramIR) -> None:
    text = "Outra instrução.\n"
    other = InstructionFile("x.md", text, hashlib.sha256(text.encode()).hexdigest())

    assert _build(spec_dict, instruction=other).digest() != program.digest()


def test_build_from_files(spec_dict: SpecDict, write_spec: Any) -> None:
    path = write_spec(spec_dict)
    loaded = load_spec(path)
    instruction = read_instruction(path.parent, "instructions/extractor.md")
    program = build_program(
        loaded.spec,
        spec_sha256=loaded.sha256,
        instructions={"extractor_agent": instruction},
    )

    assert program.spec_sha256 == loaded.sha256

"""Property: any valid spec the generator can build renders into code that parses,
compiles and passes the AST allowlist, with spec text preserved exactly."""

import ast
import keyword
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from transpiler import semantic
from transpiler.codegen.render import render
from transpiler.errors import Code, errors_of
from transpiler.ir.builder import build_program
from transpiler.lowering import lower
from transpiler.spec.loader import InstructionFile, parse_spec
from transpiler.spec.models import RESERVED_IDENTIFIERS
from transpiler.validation.ast_rules import check_source
from transpiler.validation.compile_check import compile_source

pytestmark = pytest.mark.unit

identifiers = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True).filter(
    lambda name: (
        not keyword.iskeyword(name)
        and not keyword.issoftkeyword(name)
        and name not in RESERVED_IDENTIFIERS
    )
)
texts = st.text(min_size=1, max_size=120)
thresholds = st.floats(min_value=0, max_value=1, allow_nan=False)


@st.composite
def specs(draw: st.DrawFn) -> tuple[dict[str, Any], str]:
    names = draw(st.lists(identifiers, min_size=16, max_size=16, unique=True))
    (agent, ocr, guard, extract, resolve, gate, present, abort, review, schedule) = names[:10]
    (llm, lexical, merge, reconcile, llm_agent, sub) = names[10:]
    use_llm = draw(st.booleans())
    use_review = draw(st.booleans())
    use_schedule = draw(st.booleans())
    accept = draw(thresholds)
    review_t = draw(st.floats(min_value=0, max_value=accept, allow_nan=False))

    nodes: list[dict[str, Any]] = [
        {
            "id": ocr,
            "kind": "ocr_extract",
            "server": "ocr",
            "tool": "extract_text",
            "params": {"min_mean_confidence": draw(thresholds)},
            "timeout_s": 60,
        },
        {
            "id": guard,
            "kind": "untrusted_text_guard",
            "params": {"block_at": draw(st.sampled_from(["high", "medium"]))},
        },
        {
            "id": resolve,
            "kind": "rag_resolve",
            "server": "rag",
            "tool": "batch_search",
            "params": {"top_k": draw(st.integers(2, 10))},
            "timeout_s": 60,
        },
        {
            "id": gate,
            "kind": "confidence_gate",
            "params": {
                "accept_threshold": accept,
                "review_threshold": review_t,
                "min_margin": draw(st.floats(0, 0.5, allow_nan=False)),
            },
        },
        {
            "id": present,
            "kind": "present",
            "params": {"format": draw(st.sampled_from(["table", "json"]))},
        },
        {"id": abort, "kind": "abort"},
    ]
    edges: list[dict[str, Any]] = [
        {"from": "START", "to": ocr},
        {"from": ocr, "to": guard, "route": "ok"},
        {"from": ocr, "to": abort, "route": "low_quality"},
        {"from": guard, "to": extract, "route": "clean"},
        {"from": guard, "to": abort, "route": "blocked"},
        {"from": extract, "to": resolve},
        {"from": resolve, "to": gate},
        {"from": gate, "to": abort, "route": "empty"},
    ]
    spec: dict[str, Any] = {
        "spec_version": "1.0",
        "agent": {"name": agent, "description": draw(texts), "version": "1.0.0"},
        "mcp_servers": {
            "ocr": {
                "transport": "sse",
                "url_env": "OCR_URL",
                "auth_token_env": "TOKEN_ENV",
                "allowed_tools": ["extract_text"],
                "connect_timeout_s": 5,
                "read_timeout_s": 40,
            },
            "rag": {
                "transport": "sse",
                "url_env": "RAG_URL",
                "auth_token_env": "TOKEN_ENV",
                "allowed_tools": ["batch_search"],
                "connect_timeout_s": 5,
                "read_timeout_s": 15,
            },
        },
        "workflow": {"nodes": nodes, "edges": edges},
        "policies": {"pii": {"fail_closed": True}},
    }

    if use_llm:
        assume(f"{llm}_input" not in names)
        nodes.append({"id": extract, "kind": "subworkflow", "ref": sub})
        spec["models"] = {
            "m": {
                "provider": "gemini",
                "name": "gemini-flash-latest",
                "temperature": 0.0,
                "max_output_tokens": 512,
            }
        }
        spec["llm_agents"] = {
            llm_agent: {"model": "m", "instruction_file": "i.md", "output_schema": "ExtractedExams"}
        }
        spec["subworkflows"] = {
            sub: {
                "nodes": [
                    {"id": llm, "kind": "llm_extract", "agent": llm_agent},
                    {
                        "id": lexical,
                        "kind": "lexical_extract",
                        "server": "rag",
                        "tool": "batch_search",
                        "timeout_s": 60,
                    },
                    {"id": merge, "kind": "join"},
                    {"id": reconcile, "kind": "reconcile"},
                ],
                "edges": [
                    {"from": "START", "to": llm},
                    {"from": "START", "to": lexical},
                    {"from": llm, "to": merge},
                    {"from": lexical, "to": merge},
                    {"from": merge, "to": reconcile},
                ],
            }
        }
    else:
        nodes.append(
            {
                "id": extract,
                "kind": "lexical_extract",
                "server": "rag",
                "tool": "batch_search",
                "timeout_s": 60,
            }
        )

    after_gate = present
    if use_schedule:
        spec["http_services"] = {
            "api": {"base_url_env": "API_URL", "api_key_env": "API_KEY_ENV", "timeout_s": 10}
        }
        nodes.append({"id": schedule, "kind": "schedule", "service": "api", "timeout_s": 60})
        edges.append({"from": schedule, "to": present})
        after_gate = schedule
    edges.append({"from": gate, "to": after_gate, "route": "accepted"})
    if use_review:
        nodes.append({"id": review, "kind": "human_review"})
        edges.append({"from": gate, "to": review, "route": "review"})
        edges.append({"from": review, "to": after_gate, "route": "confirmed"})
        edges.append({"from": review, "to": present, "route": "rejected"})
    else:
        edges.append({"from": gate, "to": present, "route": "review"})

    nodes[:] = draw(st.permutations(nodes))
    edges[:] = draw(st.permutations(edges))
    return spec, draw(st.text(max_size=300))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(specs())
def test_valid_specs_render_to_safe_code(generated: tuple[dict[str, Any], str]) -> None:
    raw, instruction_text = generated
    spec = parse_spec(raw)
    diagnostics = semantic.validate(spec, spec_dir=Path())
    # instruction_file is not on disk here; everything else must be valid.
    assert [d for d in errors_of(diagnostics) if d.code is not Code.E_INSTRUCTION_FILE] == []
    instructions = {
        key: InstructionFile("i.md", instruction_text, "0" * 64) for key in spec.llm_agents
    }

    program = lower(build_program(spec, spec_sha256="0" * 64, instructions=instructions))
    package = render(program, spec_name="spec.json")

    for name, source in (("agent.py", package.agent_py), ("__init__.py", package.init_py)):
        assert check_source(source, name) == []
        assert compile_source(source, name) == []
    tree = ast.parse(package.agent_py)
    assigned = {ast.unparse(n.targets[0]) for n in tree.body if isinstance(n, ast.Assign)}
    assert set(program.node_names()) <= assigned
    assert raw["agent"]["description"] in {
        ast.literal_eval(kw.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "root_agent"
        and isinstance(n.value, ast.Call)
        for kw in n.value.keywords
        if kw.arg == "description"
    }
    if spec.llm_agents:
        [constant] = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]).endswith("_INSTRUCTION")
        ]
        assert ast.literal_eval(constant.value) == (instruction_text or "")

from typing import Any
from unittest.mock import MagicMock

import pytest
from google.adk import Agent, Context, Event, Workflow
from google.adk.models import LlmRequest, LlmResponse
from google.adk.workflow import FunctionNode
from google.genai import types
from runtime_fakes import ScriptedLlm, run_node
from structlog.testing import capture_logs

from agent_runtime.guardrails import pii_tripwire
from agent_runtime.guardrails.tripwire import TRIPWIRE_STATE_KEY, findings
from agent_runtime.schemas import ExtractedExams

pytestmark = pytest.mark.unit


class FakeContext:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}


def _request(*texts: str, instruction: Any = None) -> LlmRequest:
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=t)]) for t in texts]
    )
    if instruction is not None:
        request.config = types.GenerateContentConfig(system_instruction=instruction)
    return request


def test_pii_tripwire_blocks_valid_cpf() -> None:
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="CPF 529.982.247-25")])]
    )

    assert isinstance(pii_tripwire(MagicMock(), request), LlmResponse)


@pytest.mark.parametrize(
    ("text", "entity"),
    [
        ("CPF 529.982.247-25", "CPF"),
        ("cpf 52998224725", "CPF"),
        ("CNS 100 0000 0000 0007", "CNS"),
        ("contato: maria.ficticia@example.com", "EMAIL"),
        ("Telefone: (31) 99876-5432", "PHONE"),
    ],
)
def test_blocks_each_high_precision_entity(text: str, entity: str) -> None:
    context = FakeContext()

    with capture_logs() as logs:
        response = pii_tripwire(context, _request("Documento:", text))  # type: ignore[arg-type]

    assert isinstance(response, LlmResponse)
    assert response.error_code == "PII_TRIPWIRE"
    assert context.state == {TRIPWIRE_STATE_KEY: True}
    [event] = [e for e in logs if e["event"] == "pii_tripwire_triggered"]
    assert event["counts"] == {entity: 1}
    assert text.rsplit(maxsplit=1)[-1] not in str(logs)


@pytest.mark.parametrize(
    "text",
    [
        "Paciente: [PESSOA_1]\nCPF: [CPF_1]\nTelefone: [TELEFONE_1]\nE-mail: [EMAIL_1]",
        "1. Hemograma completo\n2. Glicemia de jejum\n3. TSH\n4. T4 livre",
        "CPF 529.982.247-26",
        "Belo Horizonte, 20/10/2026",
        "documento 2f3b5fd6-2866-4383-9a4e-6d1f0b8c7a11",
    ],
)
def test_lets_masked_and_ordinary_text_through(text: str) -> None:
    context = FakeContext()

    assert pii_tripwire(context, _request(text)) is None  # type: ignore[arg-type]
    assert context.state == {}


@pytest.mark.parametrize(
    "instruction",
    [
        "Escreva para ana@example.com",
        types.Content(parts=[types.Part(text="ana@example.com")]),
        types.Part(text="ana@example.com"),
        ["Regras", types.Part(text="ana@example.com")],
    ],
)
def test_checks_the_system_instruction(instruction: Any) -> None:
    assert findings(_request("TSH", instruction=instruction)) == {"EMAIL": 1}


def test_clean_instruction_passes() -> None:
    assert findings(_request("TSH", instruction="Extraia exames.")) is None


def test_binary_parts_cannot_be_inspected_and_are_blocked() -> None:
    image = types.Part(inline_data=types.Blob(mime_type="image/png", data=b"\x89PNG"))
    request = LlmRequest(contents=[types.Content(role="user", parts=[image])])

    assert findings(request) == {"UNINSPECTABLE": 1}


def test_function_parts_are_inspected() -> None:
    call = types.Part(function_call=types.FunctionCall(name="f", args={"to": "ana@example.com"}))
    request = LlmRequest(contents=[types.Content(role="model", parts=[call])])

    assert findings(request) == {"EMAIL": 1}


async def test_inside_an_agent_the_model_is_never_called() -> None:
    async def to_text(ctx: Context, node_input: Any) -> Event:
        return Event(output="Documento: CPF 529.982.247-25")

    model = ScriptedLlm()
    agent = Agent(
        name="extractor",
        model=model,
        mode="single_turn",
        instruction="Extraia exames.",
        output_schema=ExtractedExams,
        before_model_callback=pii_tripwire,
    )
    text = FunctionNode(func=to_text, name="text")
    sub = Workflow(name="sub", edges=[("START", text), (text, agent)])

    run = await run_node(sub)

    assert model.requests == []
    assert run.state[TRIPWIRE_STATE_KEY] is True
    assert any(e.error_code == "PII_TRIPWIRE" for e in run.events)

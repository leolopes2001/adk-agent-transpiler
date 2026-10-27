import hashlib
import json

import httpx
import pytest
from google.adk.workflow import RetryConfig
from runtime_fakes import ApiAnswer, Confirm, answer, candidate, run_node

from agent_runtime.bootstrap import HttpServiceConfig, RuntimeDeps
from agent_runtime.errors import ConfigurationError, PermanentError, TransientError
from agent_runtime.gateways.scheduler import SchedulerClient
from agent_runtime.nodes.schedule import accepted_exams, idempotency_key, schedule
from agent_runtime.schemas import Candidate, ExamMention, GateDecision, GateItem, ScheduleOutcome

pytestmark = pytest.mark.unit

DOC = "ab" * 32
PATIENT = "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA"
RUN_ID = "2f3b5fd6-2866-4383-9a4e-6d1f0b8c7a11"
APPOINTMENT_ID = "apt_01JABCDEFGHJKMNPQRSTVWXYZ0"
STATE = {"run.id": RUN_ID, "run.patient_ref": PATIENT, "run.document_sha256": DOC}
SERVICE = HttpServiceConfig(
    base_url_env="SCHEDULER_API_URL", api_key_env="SCHEDULER_API_KEY", timeout_s=10.0
)


def _cand(code: str, name: str) -> Candidate:
    return Candidate.model_validate(candidate(code, name, 1.0))


def _item(status: str, selected: Candidate | None) -> GateItem:
    return GateItem.model_validate(
        {
            "mention": ExamMention(text="x", source="both", evidence="x"),
            "status": status,
            "selected": selected,
            "candidates": [selected] if selected else [],
            "reason": None,
        }
    )


TSH = _cand("LAB-10043", "TSH")
HEMOGRAMA = _cand("LAB-10001", "Hemograma completo")
DECISION = GateDecision(
    items=[_item("accepted", TSH), _item("rejected", None), _item("accepted", HEMOGRAMA)]
)


def _deps(handler: httpx.MockTransport) -> RuntimeDeps:
    deps = RuntimeDeps.lazy(http_services={"scheduler": SERVICE})
    client = SchedulerClient("http://api.local", "k" * 32, 5.0, transport=handler)
    deps.use(schedulers={"scheduler": client})
    return deps


def _transport(*responses: ApiAnswer, seen: list[httpx.Request]) -> httpx.MockTransport:
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer(queue.pop(0) if len(queue) > 1 else queue[0], request)

    return httpx.MockTransport(handle)


def _node(deps: RuntimeDeps, retry: RetryConfig | None = None) -> object:
    return schedule(name="schedule", deps=deps, service="scheduler", retry=retry)


def test_idempotency_key_is_deterministic_and_order_free() -> None:
    key = idempotency_key(DOC, PATIENT, ["LAB-10043", "LAB-10001"])

    assert key == idempotency_key(DOC, PATIENT, ["LAB-10001", "LAB-10043"])
    expected = hashlib.sha256(f"v1|{DOC}|{PATIENT}|LAB-10001,LAB-10043".encode()).hexdigest()
    assert key == expected


@pytest.mark.parametrize(
    ("document", "patient", "codes"),
    [
        ("cd" * 32, PATIENT, ["LAB-10001", "LAB-10043"]),
        (DOC, "pt_A7Q2M4XZB3C6D7E5F6G7H2J4KA", ["LAB-10001", "LAB-10043"]),
        (DOC, PATIENT, ["LAB-10001"]),
        (DOC, PATIENT, ["LAB-10001", "LAB-10043", "LAB-10044"]),
    ],
)
def test_idempotency_key_changes_with_each_component(
    document: str, patient: str, codes: list[str]
) -> None:
    assert idempotency_key(document, patient, codes) != idempotency_key(
        DOC, PATIENT, ["LAB-10001", "LAB-10043"]
    )


def test_accepted_exams_are_unique_and_sorted() -> None:
    decision = GateDecision(items=[*DECISION.items, _item("accepted", TSH)])

    assert [c.code for c in accepted_exams(decision)] == ["LAB-10001", "LAB-10043"]


async def test_created() -> None:
    seen: list[httpx.Request] = []
    deps = _deps(_transport(Confirm(201, APPOINTMENT_ID), seen=seen))

    run = await run_node(_node(deps), DECISION, STATE)

    outcome = ScheduleOutcome.model_validate(run.output)
    assert (outcome.status, outcome.appointment_id, outcome.reason) == (
        "scheduled",
        APPOINTMENT_ID,
        None,
    )
    assert outcome.decision == DECISION
    [request] = seen
    assert request.headers["Idempotency-Key"] == idempotency_key(
        DOC, PATIENT, ["LAB-10001", "LAB-10043"]
    )
    assert request.headers["X-Request-ID"] == RUN_ID
    assert json.loads(request.content) == {
        "patient_ref": PATIENT,
        "exams": [
            {"code": "LAB-10001", "name": "Hemograma completo"},
            {"code": "LAB-10043", "name": "TSH"},
        ],
        "source_document_sha256": DOC,
        "requested_by": "unit_app",
    }


async def test_replayed() -> None:
    seen: list[httpx.Request] = []
    replay = Confirm(200, APPOINTMENT_ID, {"Idempotent-Replayed": "true"})

    run = await run_node(_node(_deps(_transport(replay, seen=seen))), DECISION, STATE)

    outcome = ScheduleOutcome.model_validate(run.output)
    assert (outcome.status, outcome.appointment_id) == ("replayed", APPOINTMENT_ID)


@pytest.mark.parametrize(
    ("status", "title"), [(409, "Idempotency key reuse"), (422, "Unknown exam code")]
)
async def test_refusals_become_failed_outcomes_without_retry(status: int, title: str) -> None:
    seen: list[httpx.Request] = []
    refusal = httpx.Response(status, json={"title": title, "status": status})
    retry = RetryConfig(max_attempts=3, initial_delay=0.0, max_delay=0.0, jitter=0.0)

    run = await run_node(_node(_deps(_transport(refusal, seen=seen)), retry), DECISION, STATE)

    outcome = ScheduleOutcome.model_validate(run.output)
    assert (outcome.status, outcome.appointment_id, outcome.reason) == ("failed", None, title)
    assert len(seen) == 1


async def test_unexpected_response_becomes_failed() -> None:
    seen: list[httpx.Request] = []

    run = await run_node(_node(_deps(_transport(httpx.Response(204), seen=seen))), DECISION, STATE)

    assert ScheduleOutcome.model_validate(run.output).reason == "SCHEDULER_UNEXPECTED_RESPONSE"


async def test_server_errors_are_retried_with_the_same_key() -> None:
    seen: list[httpx.Request] = []
    transport = _transport(
        httpx.Response(503),
        httpx.Response(500),
        Confirm(201, APPOINTMENT_ID),
        seen=seen,
    )
    retry = RetryConfig(max_attempts=3, initial_delay=0.0, max_delay=0.0, jitter=0.0)

    run = await run_node(_node(_deps(transport), retry), DECISION, STATE)

    assert ScheduleOutcome.model_validate(run.output).status == "scheduled"
    assert len({request.headers["Idempotency-Key"] for request in seen}) == 1
    assert len(seen) == 3


async def test_server_errors_after_the_last_attempt_fail_the_run() -> None:
    seen: list[httpx.Request] = []
    retry = RetryConfig(max_attempts=2, initial_delay=0.0, max_delay=0.0, jitter=0.0)

    with pytest.raises(TransientError):
        await run_node(
            _node(_deps(_transport(httpx.Response(502), seen=seen)), retry), DECISION, STATE
        )

    assert len(seen) == 2


async def test_without_patient_ref_nothing_is_requested() -> None:
    seen: list[httpx.Request] = []
    state = {**STATE, "run.patient_ref": None}

    run = await run_node(_node(_deps(_transport(httpx.Response(201), seen=seen))), DECISION, state)

    outcome = ScheduleOutcome.model_validate(run.output)
    assert (outcome.status, outcome.reason) == ("not_scheduled", "patient_identifier_missing")
    assert seen == []


async def test_without_accepted_exams_nothing_is_requested() -> None:
    seen: list[httpx.Request] = []
    decision = GateDecision(items=[_item("rejected", None)])

    run = await run_node(_node(_deps(_transport(httpx.Response(201), seen=seen))), decision, STATE)

    assert ScheduleOutcome.model_validate(run.output).reason == "no_accepted_exams"
    assert seen == []


async def test_missing_document_hash_is_permanent() -> None:
    seen: list[httpx.Request] = []
    state = {"run.patient_ref": PATIENT}

    with pytest.raises(PermanentError, match="MISSING_DOCUMENT_HASH"):
        await run_node(_node(_deps(_transport(httpx.Response(201), seen=seen))), DECISION, state)


async def test_deps_build_the_client_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = RuntimeDeps.lazy(http_services={"scheduler": SERVICE})
    monkeypatch.delenv("SCHEDULER_API_URL", raising=False)
    monkeypatch.delenv("SCHEDULER_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="SCHEDULER_API_URL, SCHEDULER_API_KEY"):
        deps.scheduler("scheduler")
    with pytest.raises(ConfigurationError, match="UNKNOWN_SERVICE"):
        deps.scheduler("billing")

    monkeypatch.setenv("SCHEDULER_API_URL", "http://api.local")
    monkeypatch.setenv("SCHEDULER_API_KEY", "k" * 32)
    client = deps.scheduler("scheduler")
    assert isinstance(client, SchedulerClient)
    assert deps.scheduler("scheduler") is client
    await deps.aclose()
    assert deps.scheduler("scheduler") is not client

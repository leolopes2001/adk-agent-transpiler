"""`schedule`: requests the appointment for the accepted exams (ADR-0006).

The only side effect of the workflow. The `Idempotency-Key` is derived from
what is being requested, `sha256("v1|<document sha256>|<patient_ref>|<codes>")`
with the codes sorted, so a node retry, a lost response or a second run with
the same image and result all map to one appointment.

- no `patient_ref` (the order has no patient identifier) -> `not_scheduled`;
- 201 -> `scheduled`; 200 replay -> `replayed`;
- refusal (4xx) or another permanent error -> `failed`, with the reason;
- 5xx, timeout, connection -> TransientError, retried by the node.
"""

import hashlib
import uuid
from typing import Any

from google.adk import Context, Event
from google.adk.workflow import FunctionNode, RetryConfig

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.errors import PermanentError
from agent_runtime.gateways.scheduler import AppointmentRequest, SchedulerRejected
from agent_runtime.nodes._common import function_node, parse_input
from agent_runtime.schemas import Candidate, GateDecision, ScheduleOutcome
from lab_common import logging as lab_logging

KEY_VERSION = "v1"

log = lab_logging.get_logger(__name__)


def idempotency_key(document_sha256: str, patient_ref: str, codes: list[str]) -> str:
    material = f"{KEY_VERSION}|{document_sha256}|{patient_ref}|{','.join(sorted(codes))}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def accepted_exams(decision: GateDecision) -> list[Candidate]:
    """Accepted candidates, one per code, sorted by code."""
    by_code = {item.selected.code: item.selected for item in decision.items if item.selected}
    return [by_code[code] for code in sorted(by_code)]


def _request_id(ctx: Context) -> str:
    try:
        return str(uuid.UUID(str(ctx.state.get("run.id"))))
    except ValueError:
        return str(uuid.uuid4())


def _outcome(decision: GateDecision, status: str, **fields: Any) -> ScheduleOutcome:
    return ScheduleOutcome.model_validate(
        {"decision": decision, "status": status, "appointment_id": None, "reason": None, **fields}
    )


def schedule(
    *,
    name: str,
    deps: RuntimeDeps,
    service: str,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
) -> FunctionNode:
    """Requests the appointment with a deterministic idempotency key."""

    async def run(ctx: Context, node_input: Any) -> Event:
        decision = parse_input(GateDecision, node_input)
        exams = accepted_exams(decision)
        patient_ref = ctx.state.get("run.patient_ref")
        document_sha256 = ctx.state.get("run.document_sha256")
        if not exams:
            outcome = _outcome(decision, "not_scheduled", reason="no_accepted_exams")
        elif not isinstance(patient_ref, str) or not patient_ref:
            outcome = _outcome(decision, "not_scheduled", reason="patient_identifier_missing")
        elif not isinstance(document_sha256, str):
            raise PermanentError("MISSING_DOCUMENT_HASH", "run.document_sha256 is not set")
        else:
            outcome = await _request(ctx, decision, exams, patient_ref, document_sha256)
        log.info(
            "schedule_result",
            node=name,
            status=outcome.status,
            reason=outcome.reason,
            appointment_id=outcome.appointment_id,
            exams=[exam.code for exam in exams],
        )
        return Event(output=outcome)

    async def _request(
        ctx: Context,
        decision: GateDecision,
        exams: list[Candidate],
        patient_ref: str,
        document_sha256: str,
    ) -> ScheduleOutcome:
        request = AppointmentRequest(
            patient_ref=patient_ref,
            exams=tuple((exam.code, exam.name) for exam in exams),
            source_document_sha256=document_sha256,
            requested_by=ctx.session.app_name,
        )
        key = idempotency_key(document_sha256, patient_ref, [exam.code for exam in exams])
        try:
            result = await deps.scheduler(service).create_appointment(
                request, idempotency_key=key, request_id=_request_id(ctx)
            )
        except SchedulerRejected as error:
            return _outcome(decision, "failed", reason=error.title)
        except PermanentError as error:
            return _outcome(decision, "failed", reason=error.code)
        return _outcome(decision, result.status, appointment_id=result.appointment_id)

    return function_node(name, run, retry=retry, timeout=timeout)

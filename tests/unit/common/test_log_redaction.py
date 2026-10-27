import hashlib
import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
import structlog
from hypothesis import given, settings
from hypothesis import strategies as st

from lab_common import logging as lab_logging
from lab_common.logging import MAX_STRING_CHARS, RedactPii

pytestmark = pytest.mark.unit

# Values printed on the sample images (scripts/make_sample_images.py).
CPF = "202.393.613-69"
PHONE = "(31) 90000-0000"


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    root = logging.getLogger()
    level = root.level
    yield
    if lab_logging._handler is not None:
        root.removeHandler(lab_logging._handler)
        lab_logging._handler = None
    root.setLevel(level)
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


@pytest.fixture
def stream() -> io.StringIO:
    lab_logging.configure("ocr-mcp", stream=(buffer := io.StringIO()))
    return buffer


def records(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def redact(**event: object) -> dict[str, object]:
    return RedactPii()(None, "info", dict(event))


class TestEnabledByDefault:
    def test_redacts_cpf_in_structlog_event_and_fields(self, stream: io.StringIO) -> None:
        lab_logging.get_logger().info(f"paciente {CPF}", phone=PHONE, ref=20239361369)

        [record] = records(stream)
        assert record["event"] == "paciente [REDACTED:CPF]"
        assert record["phone"] == "[REDACTED:PHONE]"
        assert record["ref"] == "[REDACTED:CPF]"
        assert CPF not in stream.getvalue()

    def test_redacts_stdlib_messages_and_extra(self, stream: io.StringIO) -> None:
        logging.getLogger("mcp.server").warning(
            "client sent %s", CPF, extra={"contact": "maria.ficticia@example.com"}
        )

        [record] = records(stream)
        assert record["event"] == "client sent [REDACTED:CPF]"
        assert record["contact"] == "[REDACTED:EMAIL]"

    def test_redacts_exception_text(self, stream: io.StringIO) -> None:
        try:
            raise ValueError(f"bad value {CPF}")
        except ValueError:
            lab_logging.get_logger().exception("ocr_rejected")

        assert CPF not in stream.getvalue()
        assert "[REDACTED:CPF]" in records(stream)[0]["exception"]

    def test_redacts_context_variables(self, stream: io.StringIO) -> None:
        structlog.contextvars.bind_contextvars(request_id=PHONE)
        lab_logging.get_logger().info("ocr_request")

        assert records(stream)[0]["request_id"] == "[REDACTED:PHONE]"

    def test_drops_forbidden_keys(self, stream: io.StringIO) -> None:
        lab_logging.get_logger().info("ocr_done", raw_text="Paciente: Maria", image_base64="iVBO")

        [record] = records(stream)
        assert "raw_text" not in record
        assert "image_base64" not in record

    def test_runs_after_final_processors(self) -> None:
        def add_raw_text(logger: object, method_name: str, event_dict: Any) -> Any:
            event_dict["raw_text"] = "Paciente: Maria"
            event_dict["note"] = CPF
            return event_dict

        out = io.StringIO()
        lab_logging.configure("ocr-mcp", stream=out, final_processors=[add_raw_text])
        lab_logging.get_logger().info("ocr_done")

        [record] = records(out)
        assert "raw_text" not in record
        assert record["note"] == "[REDACTED:CPF]"

    def test_can_be_disabled_explicitly(self) -> None:
        out = io.StringIO()
        lab_logging.configure("ocr-mcp", stream=out, redact_pii=False)
        lab_logging.get_logger().info("debug", value=CPF)

        assert records(out)[0]["value"] == CPF


class TestProcessor:
    def test_drops_forbidden_keys_at_any_depth(self) -> None:
        result = redact(
            event="x", payload={"body": "{}", "items": [{"text_raw": "a", "image_bytes": "b"}]}
        )
        assert result["payload"] == {"items": [{}]}

    def test_redacts_inside_nested_values(self) -> None:
        result = redact(event="x", data={"contacts": [PHONE, ("ok", CPF)], CPF: 1})
        assert result["data"] == {
            "contacts": ["[REDACTED:PHONE]", ["ok", "[REDACTED:CPF]"]],
            "[REDACTED:CPF]": 1,
        }

    def test_replaces_bytes(self) -> None:
        assert redact(event="x", image=b"\x89PNG")["image"] == "[REDACTED:BYTES]"

    def test_converts_other_objects_before_checking(self) -> None:
        class Patient:
            def __str__(self) -> str:
                return f"Patient(cpf={CPF})"

        assert redact(event="x", patient=Patient())["patient"] == "Patient(cpf=[REDACTED:CPF])"

    def test_keeps_safe_scalars(self) -> None:
        result = redact(event="ocr_done", lines=24, mean_confidence=0.91, ok=True, err=None)
        assert result == {
            "event": "ocr_done",
            "lines": 24,
            "mean_confidence": 0.91,
            "ok": True,
            "err": None,
        }

    def test_does_not_redact_low_precision_matches(self) -> None:
        # Names and invalid CPFs are only masked in documents, by the full detector set.
        event = "Paciente: Maria Fictícia, CPF 202.393.613-60, nasc. 14/03/1985"
        assert redact(event=event)["event"] == event

    def test_truncates_long_strings_after_redacting(self) -> None:
        # "x" is not a hex digit: a CPF glued to "aaa…" would read as part of a hex id.
        text = "x" * (MAX_STRING_CHARS - 5) + CPF
        result = redact(event=text)["event"]

        assert isinstance(result, str)
        assert result.startswith("x" * (MAX_STRING_CHARS - 5) + "[REDA")
        assert result.endswith("[truncated]")
        assert len(result) <= MAX_STRING_CHARS + len("…[truncated]")


class TestIdentifiers:
    """Correlation ids and hashes must reach the log intact."""

    @settings(max_examples=500)
    @given(st.uuids())
    def test_uuid_is_kept(self, value: object) -> None:
        assert redact(request_id=str(value)) == {"request_id": str(value)}

    @settings(max_examples=300)
    @given(st.binary(max_size=64))
    def test_sha256_is_kept(self, data: bytes) -> None:
        digest = hashlib.sha256(data).hexdigest()
        assert redact(catalog_sha256=digest, event=f"loaded {digest}") == {
            "catalog_sha256": digest,
            "event": f"loaded {digest}",
        }

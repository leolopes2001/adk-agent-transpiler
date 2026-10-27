"""`patient_ref`: a pseudonymous, stable id for the patient of an order.

`pt_` + the first 26 characters of base32(HMAC-SHA256(secret, identifier)),
which is exactly the pattern the scheduler API accepts. The same patient gets
the same ref on every order (so appointments stay idempotent) while the ref
reveals nothing without the secret, which only this server holds.

The identifier is, in order of preference: the first CPF with valid check
digits, the first valid CNS, the first name after a patient label. CPFs with
wrong check digits are skipped, since a misread digit would yield a different
patient. The kind is part of the HMAC input, so a CPF and a CNS with the same
digits never collide.
"""

import base64
import hashlib
import hmac
from collections.abc import Sequence

from lab_common.pii import PiiEntity, PiiSpan, is_valid_cns, is_valid_cpf
from lab_common.pii.types import only_digits
from lab_common.text import tokens

REF_PREFIX = "pt_"
REF_LENGTH = 26


def normalize_identifier(kind: PiiEntity, value: str) -> str:
    if kind in (PiiEntity.CPF, PiiEntity.CNS):
        return only_digits(value)
    if kind is PiiEntity.PERSON:
        return " ".join(tokens(value))
    raise ValueError(f"{kind} is not a patient identifier")


def patient_ref(secret: bytes, kind: PiiEntity, value: str) -> str:
    message = f"{kind.value}:{normalize_identifier(kind, value)}".encode()
    digest = hmac.new(secret, message, hashlib.sha256).digest()
    return REF_PREFIX + base64.b32encode(digest).decode("ascii")[:REF_LENGTH]


def _is_identifier(text: str, span: PiiSpan, kind: PiiEntity) -> bool:
    if span.entity is not kind:
        return False
    value = text[span.start : span.end]
    if kind is PiiEntity.CPF:
        return is_valid_cpf(value)
    if kind is PiiEntity.CNS:
        return is_valid_cns(value)
    return span.detector == "person.patient" and bool(tokens(value))


def find_patient_ref(secret: bytes, text: str, spans: Sequence[PiiSpan]) -> str | None:
    """Ref from the best identifier among `spans` (offsets into the raw `text`)."""
    for kind in (PiiEntity.CPF, PiiEntity.CNS, PiiEntity.PERSON):
        for span in spans:
            if _is_identifier(text, span, kind):
                return patient_ref(secret, kind, text[span.start : span.end])
    return None

import base64
import hashlib
import hmac
from collections.abc import Sequence

from lab_common.pii import PiiEntity, PiiSpan, is_valid_cns, is_valid_cpf
from lab_common.pii.types import only_digits
from lab_common.text import tokens

REF_PREFIX = "pt_"
REF_LENGTH = 26
IDENTIFIER_PREFERENCE = (PiiEntity.CPF, PiiEntity.CNS, PiiEntity.PERSON)
PATIENT_NAME_DETECTOR = "person.patient"


def find_patient_ref(secret: bytes, text: str, spans: Sequence[PiiSpan]) -> str | None:
    for kind in IDENTIFIER_PREFERENCE:
        for span in spans:
            if _is_patient_identifier(text, span, kind):
                return patient_ref(secret, kind, text[span.start : span.end])

    return None


def patient_ref(secret: bytes, kind: PiiEntity, value: str) -> str:
    # The kind is hashed too, so a CPF and a CNS with the same digits never collide.
    message = f"{kind.value}:{normalize_identifier(kind, value)}".encode()
    digest = hmac.new(secret, message, hashlib.sha256).digest()
    return REF_PREFIX + base64.b32encode(digest).decode("ascii")[:REF_LENGTH]


def normalize_identifier(kind: PiiEntity, value: str) -> str:
    if kind in (PiiEntity.CPF, PiiEntity.CNS):
        return only_digits(value)

    if kind is PiiEntity.PERSON:
        return " ".join(tokens(value))

    raise ValueError(f"{kind} is not a patient identifier")


def _is_patient_identifier(text: str, span: PiiSpan, kind: PiiEntity) -> bool:
    if span.entity is not kind:
        return False

    value = text[span.start : span.end]
    # Invalid check digits are skipped: a misread digit would yield another patient.
    if kind is PiiEntity.CPF:
        return is_valid_cpf(value)

    if kind is PiiEntity.CNS:
        return is_valid_cns(value)

    return span.detector == PATIENT_NAME_DETECTOR and bool(tokens(value))

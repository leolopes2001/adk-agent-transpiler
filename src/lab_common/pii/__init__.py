"""Deterministic detection of Brazilian personal data in document text.

Each detector lives in its own module and finds one entity; `default_detectors`
is the set the OCR server masks with.
"""

from lab_common.pii.address import AddressDetector
from lab_common.pii.birth_date import BirthDateDetector
from lab_common.pii.cep import CepDetector
from lab_common.pii.cns import CnsDetector, is_valid_cns
from lab_common.pii.cpf import CpfDetector, is_valid_cpf
from lab_common.pii.crm import CrmDetector
from lab_common.pii.email import EmailDetector
from lab_common.pii.health_card import HealthCardDetector
from lab_common.pii.phone import PhoneDetector
from lab_common.pii.rg import RgDetector
from lab_common.pii.types import PiiDetector, PiiEntity, PiiSpan

__all__ = [
    "AddressDetector",
    "BirthDateDetector",
    "CepDetector",
    "CnsDetector",
    "CpfDetector",
    "CrmDetector",
    "EmailDetector",
    "HealthCardDetector",
    "PhoneDetector",
    "PiiDetector",
    "PiiEntity",
    "PiiSpan",
    "RgDetector",
    "default_detectors",
    "is_valid_cns",
    "is_valid_cpf",
]


def default_detectors() -> list[PiiDetector]:
    return [
        CpfDetector(),
        CnsDetector(),
        RgDetector(),
        PhoneDetector(),
        EmailDetector(),
        CepDetector(),
        AddressDetector(),
        BirthDateDetector(),
        CrmDetector(),
        HealthCardDetector(),
    ]

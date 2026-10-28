from lab_common.pii.address import AddressDetector
from lab_common.pii.birth_date import BirthDateDetector
from lab_common.pii.cep import CepDetector
from lab_common.pii.cns import CnsDetector, is_valid_cns
from lab_common.pii.cpf import CpfDetector, is_valid_cpf
from lab_common.pii.crm import CrmDetector
from lab_common.pii.email import EmailDetector
from lab_common.pii.health_card import HealthCardDetector
from lab_common.pii.person import PersonDetector
from lab_common.pii.phone import PhoneDetector
from lab_common.pii.redactor import PiiEngineError, PiiRedactor, RedactionResult
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
    "PersonDetector",
    "PhoneDetector",
    "PiiDetector",
    "PiiEngineError",
    "PiiEntity",
    "PiiRedactor",
    "PiiSpan",
    "RedactionResult",
    "RgDetector",
    "default_detectors",
    "high_precision_detectors",
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
        PersonDetector(),
        CrmDetector(),
        HealthCardDetector(),
    ]


# Values that are personal data wherever they appear (check digits, unambiguous format).
def high_precision_detectors() -> list[PiiDetector]:
    return [CpfDetector(require_valid=True), CnsDetector(), EmailDetector(), PhoneDetector()]

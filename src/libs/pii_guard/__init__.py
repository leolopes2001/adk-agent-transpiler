from pii_guard.address import AddressDetector
from pii_guard.birth_date import BirthDateDetector
from pii_guard.cep import CepDetector
from pii_guard.cns import CnsDetector, is_valid_cns
from pii_guard.cpf import CpfDetector, is_valid_cpf
from pii_guard.crm import CrmDetector
from pii_guard.email import EmailDetector
from pii_guard.health_card import HealthCardDetector
from pii_guard.labeled_field import LabeledFieldDetector
from pii_guard.person import PersonDetector
from pii_guard.phone import PhoneDetector
from pii_guard.redactor import PiiEngineError, PiiRedactor, RedactionResult
from pii_guard.rg import RgDetector
from pii_guard.types import PiiDetector, PiiEntity, PiiSpan

__all__ = [
    "AddressDetector",
    "BirthDateDetector",
    "CepDetector",
    "CnsDetector",
    "CpfDetector",
    "CrmDetector",
    "EmailDetector",
    "HealthCardDetector",
    "LabeledFieldDetector",
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
        LabeledFieldDetector(),
    ]


# Values that are personal data wherever they appear (check digits, unambiguous format).
def high_precision_detectors() -> list[PiiDetector]:
    return [CpfDetector(require_valid=True), CnsDetector(), EmailDetector(), PhoneDetector()]

"""Read-only exam catalog (codes and official names).

The API checks every requested exam against this catalog: a code the agent
invented, or a name that does not belong to the code, is rejected instead of
being recorded. Only codes and names are kept; the rest of the file (aliases,
preparation) is for the search service.
"""

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_NON_WORD = re.compile(r"[\W_]+")


def normalize_name(text: str) -> str:
    """Case, accents, punctuation and spacing do not count when comparing names."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_NON_WORD.sub(" ", without_marks.casefold()).split())


class CatalogError(ValueError):
    """The catalog file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class CatalogExam:
    code: str
    name: str


class Catalog:
    def __init__(self, exams: Iterable[CatalogExam], version: str) -> None:
        self.version = version
        self._exams: dict[str, CatalogExam] = {}
        self._normalized: dict[str, str] = {}
        for exam in exams:
            if exam.code in self._exams:
                raise CatalogError(f"duplicate exam code {exam.code!r} in catalog")
            self._exams[exam.code] = exam
            self._normalized[exam.code] = normalize_name(exam.name)

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CatalogError(f"cannot read catalog {path}: {exc}") from exc
        try:
            exams = [CatalogExam(code=e["code"], name=e["name"]) for e in raw["exams"]]
            version = raw["catalog_version"]
        except (KeyError, TypeError) as exc:
            raise CatalogError(f"catalog {path} is malformed: {exc!r}") from exc
        for exam in exams:
            if not (isinstance(exam.code, str) and exam.code and isinstance(exam.name, str)):
                raise CatalogError(f"catalog {path} has an invalid entry: {exam!r}")
        if not exams:
            raise CatalogError(f"catalog {path} has no exams")
        return cls(exams, version=str(version))

    def __len__(self) -> int:
        return len(self._exams)

    def get(self, code: str) -> CatalogExam | None:
        return self._exams.get(code)

    def name_matches(self, code: str, name: str) -> bool:
        return self._normalized.get(code) == normalize_name(name)

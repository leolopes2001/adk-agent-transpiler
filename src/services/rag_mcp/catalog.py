import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from lab_common.contracts.rag import CatalogInfo, CatalogVersion, Exam
from lab_common.text import normalize_for_search

MIN_EXAMS = 100
_MAX_REPORTED_ERRORS = 5


class CatalogError(Exception):
    pass


class _CatalogFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_version: CatalogVersion
    exams: list[Exam]


@dataclass(frozen=True)
class Catalog:
    version: str
    sha256: str
    exams: tuple[Exam, ...]
    _by_code: dict[str, Exam] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_code", {exam.code: exam for exam in self.exams})

    def __len__(self) -> int:
        return len(self.exams)

    def __iter__(self) -> Iterator[Exam]:
        return iter(self.exams)

    def __contains__(self, code: object) -> bool:
        return code in self._by_code

    def get(self, code: str) -> Exam | None:
        return self._by_code.get(code)

    def info(self) -> CatalogInfo:
        return CatalogInfo(
            catalog_version=self.version, catalog_sha256=self.sha256, exam_count=len(self)
        )


def load_catalog(path: Path, *, min_exams: int = MIN_EXAMS) -> Catalog:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CatalogError(f"cannot read catalog {path}: {exc.strerror or exc}") from None

    return parse_catalog(raw, min_exams=min_exams)


def parse_catalog(raw: bytes, *, min_exams: int = MIN_EXAMS) -> Catalog:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"catalog is not valid UTF-8 JSON: {exc}") from None

    try:
        parsed = _CatalogFile.model_validate(data)
    except ValidationError as exc:
        raise CatalogError(f"invalid catalog: {_describe_validation_errors(exc)}") from None

    _check_min_exams(parsed.exams, min_exams)
    _check_unique_codes(parsed.exams)
    _check_unique_terms(parsed.exams)
    return Catalog(
        version=parsed.catalog_version,
        # Hash of the file's bytes, so a catalog version can be tied to one exact file.
        sha256=hashlib.sha256(raw).hexdigest(),
        exams=tuple(parsed.exams),
    )


def _describe_validation_errors(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors(include_input=False, include_url=False)
    ]
    shown = "; ".join(problems[:_MAX_REPORTED_ERRORS])
    hidden = len(problems) - _MAX_REPORTED_ERRORS
    return f"{shown} (and {hidden} more)" if hidden > 0 else shown


def _check_min_exams(exams: list[Exam], min_exams: int) -> None:
    if len(exams) < min_exams:
        raise CatalogError(f"catalog has {len(exams)} exams; at least {min_exams} are required")


def _check_unique_codes(exams: list[Exam]) -> None:
    seen: set[str] = set()
    for exam in exams:
        if exam.code in seen:
            raise CatalogError(f"duplicate exam code {exam.code}")

        seen.add(exam.code)


def _check_unique_terms(exams: list[Exam]) -> None:
    """A term shared after normalization would make the same mention resolve to different exams."""
    owners: dict[str, tuple[str, str]] = {}
    for exam in exams:
        for term in [exam.name, *exam.aliases]:
            key = normalize_for_search(term)
            if not key:
                raise CatalogError(f"{exam.code}: {term!r} has no searchable characters")

            if key in owners:
                other_code, other_term = owners[key]
                raise CatalogError(
                    f"{exam.code}: {term!r} collides with {other_term!r} of {other_code} "
                    f"(both normalize to {key!r})"
                )

            owners[key] = (exam.code, term)

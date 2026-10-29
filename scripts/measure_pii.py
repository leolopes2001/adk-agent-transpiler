"""Measure the personal-data detectors on the synthetic corpus (README, "Segurança").

    uv run python scripts/measure_pii.py

Three configurations of the OCR server's redactor: the rules alone, the rules
plus the NER model on the original text, and the rules plus the NER model on
the original and on a title-case copy (what the servers run). For each one:
values masked whole per type in tests/data/pii_ner_cases.jsonl, exam terms of
the corpus masked by mistake, catalog names and aliases masked by mistake (as
written and in upper case, in five contexts), and the mean time per snippet.
Timings depend on the machine; the counts do not.
"""

import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pii_guard import PiiDetector, PiiRedactor, default_detectors, ner, shared_ner_detector

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "data" / "pii_ner_cases.jsonl"
CATALOG = ROOT / "data" / "exams_catalog.json"
CONTEXTS = ["{}", "1. {}", "- {}", "Solicito: {}", "Exames: {}, Hemograma completo"]


def main() -> None:
    cases = [json.loads(line) for line in CASES.read_text("utf-8").splitlines() if line]
    catalog = json.loads(CATALOG.read_text("utf-8"))
    mentions = sorted({m for e in catalog["exams"] for m in [e["name"], *e["aliases"]]})

    started = time.perf_counter()
    shared_ner_detector()
    load_s = time.perf_counter() - started

    totals = Counter(pii["entity"] for case in cases for pii in case["pii"])
    configs: list[tuple[str, Callable[[], list[PiiDetector]], bool]] = [
        ("Regras atuais", default_detectors, True),
        ("Regras + NER (texto original)", _with_ner, False),
        ("Regras + NER (original e cópia capitalizada)", _with_ner, True),
    ]
    entities = sorted(totals)
    print(
        f"Carga do modelo: {load_s:.1f} s; {len(cases)} trechos; {len(mentions)} nomes do catálogo"
    )
    print()
    print("| Configuração | " + " | ".join(f"{e} /{totals[e]}" for e in entities), end="")
    print(" | Termos do corpus mascarados | Catálogo mascarado | ms/trecho |")
    print("|---" * (len(entities) + 4) + "|")
    for label, detectors, titled in configs:
        row = _measure(PiiRedactor(_titled(detectors, titled)), cases, mentions)
        found = " | ".join(str(row["found"][e]) for e in entities)
        print(
            f"| {label} | {found} | {row['medical']} | {row['catalog']}/{row['catalog_total']} "
            f"| {row['ms']:.1f} |"
        )


def _with_ner() -> list[PiiDetector]:
    return [*default_detectors(), shared_ner_detector()]


def _titled(detectors: Callable[[], list[PiiDetector]], titled: bool) -> list[PiiDetector]:
    # Without the copy: title_case returns the text as is, so only the original is read.
    ner.title_case = _TITLE_CASE if titled else str
    return detectors()


_TITLE_CASE = ner.title_case


def _measure(
    redactor: PiiRedactor, cases: list[dict[str, Any]], mentions: list[str]
) -> dict[str, Any]:
    found: Counter[str] = Counter()
    medical = 0
    started = time.perf_counter()
    for case in cases:
        text = case["text"]
        spans = redactor.redact(text).spans
        for pii in case["pii"]:
            start = text.index(pii["value"])
            end = start + len(pii["value"])
            if all(
                any(s.start <= i < s.end for s in spans)
                for i in range(start, end)
                if text[i].isalnum()
            ):
                found[pii["entity"]] += 1
        for term in case["medical"]:
            start = text.index(term)
            medical += any(s.start < start + len(term) and start < s.end for s in spans)
    ms = (time.perf_counter() - started) * 1000 / len(cases)

    catalog = 0
    variants = [m for mention in mentions for m in (mention, mention.upper())]
    for mention in variants:
        for context in CONTEXTS:
            catalog += bool(redactor.redact(context.format(mention)).spans)

    return {
        "found": found,
        "medical": medical,
        "catalog": catalog,
        "catalog_total": len(variants) * len(CONTEXTS),
        "ms": ms,
    }


if __name__ == "__main__":
    main()

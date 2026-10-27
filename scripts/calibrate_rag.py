"""Calibrate the confidence gate thresholds on the RAG evaluation set.

    uv run python scripts/calibrate_rag.py --eval tests/data/rag_eval.jsonl \
        --out evidence/rag_calibration.md

Every query of the evaluation set is searched once (top 5) with the real
catalog and scorer. Then every combination of the grid

    accept_threshold in [0.70, 0.95], review_threshold in [0.45, 0.80]
    (review <= accept), min_margin in [0.00, 0.15]

is replayed through the gate rule of the runtime and measured: precision and
recall of the accepted mentions, review rate and negatives accepted.

Selection: a combination is admissible when every accepted mention has the
right code (precision 100%; an accepted ambiguous mention counts as wrong) and
no negative is accepted. Among admissible ones, the highest recall wins; ties
go to the combination closest to the thresholds currently in the spec, so the
spec only changes when the data asks for it.

Accept and review move in steps of 0.05 and the margin in steps of 0.01 by
default: the set is small, and a finer grid would place the threshold in the
narrow gap between one negative and one positive. The report also shows what
a 0.01 grid would pick, for comparison.

The report is deterministic (no timestamps), so rerunning it on the same
inputs gives the same file. Exit status 1 when no combination is admissible.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab_common.contracts.rag import SearchResult
from rag_mcp.catalog import load_catalog
from rag_mcp.scoring import ExamSearcher

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPT_RANGE = (0.70, 0.95)
REVIEW_RANGE = (0.45, 0.80)
MARGIN_RANGE = (0.00, 0.15)
TOP_K = 5
GATE_KINDS = ("confidence_gate",)


@dataclass(frozen=True, order=True)
class Thresholds:
    accept: float
    review: float
    margin: float

    def __str__(self) -> str:
        return f"{self.accept:.2f} / {self.review:.2f} / {self.margin:.2f}"


@dataclass(frozen=True)
class Case:
    query: str
    kind: str
    expected: str | None
    acceptable: tuple[str, ...]
    result: SearchResult

    @property
    def positive(self) -> bool:
        """A mention with one right answer (not negative, not ambiguous)."""
        return self.kind not in ("negative", "ambiguous")

    @property
    def top(self) -> tuple[str | None, float, float | None]:
        candidates = self.result.candidates
        if not candidates:
            return None, 0.0, None
        second = candidates[1].score if len(candidates) > 1 else None
        return candidates[0].code, candidates[0].score, second

    @property
    def top_is_right(self) -> bool:
        code, _, _ = self.top
        return code is not None and (code == self.expected or code in self.acceptable)


@dataclass(frozen=True)
class Metrics:
    thresholds: Thresholds
    accepted: int
    accepted_right: int
    positives: int
    reviewed: int
    total: int
    negatives_accepted: int
    ambiguous_accepted: int
    positives_rejected: int

    @property
    def precision(self) -> float:
        return self.accepted_right / self.accepted if self.accepted else 1.0

    @property
    def recall(self) -> float:
        return self.accepted_right / self.positives if self.positives else 0.0

    @property
    def review_rate(self) -> float:
        return self.reviewed / self.total if self.total else 0.0

    @property
    def admissible(self) -> bool:
        return self.accepted_right == self.accepted and self.negatives_accepted == 0


def decide(case: Case, t: Thresholds) -> str:
    """The runtime's confidence gate rule for one mention."""
    _, first, second = case.top
    if not case.result.candidates:
        return "rejected"
    # Scores have 4 decimals; round the margin so 0.85 - 0.80 counts as 0.05.
    margin_ok = second is None or round(first - second, 4) >= t.margin
    if first >= t.accept and margin_ok:
        return "accepted"
    if first >= t.review:
        return "review"
    return "rejected"


def measure(cases: Sequence[Case], t: Thresholds) -> Metrics:
    accepted = accepted_right = reviewed = negatives = ambiguous = rejected_positives = 0
    for case in cases:
        decision = decide(case, t)
        if decision == "accepted":
            accepted += 1
            code, _, _ = case.top
            if case.positive and code == case.expected:
                accepted_right += 1
            negatives += case.kind == "negative"
            ambiguous += case.kind == "ambiguous"
        elif decision == "review":
            reviewed += 1
        elif case.positive:
            rejected_positives += 1
    return Metrics(
        thresholds=t,
        accepted=accepted,
        accepted_right=accepted_right,
        positives=sum(case.positive for case in cases),
        reviewed=reviewed,
        total=len(cases),
        negatives_accepted=negatives,
        ambiguous_accepted=ambiguous,
        positives_rejected=rejected_positives,
    )


def steps(bounds: tuple[float, float], step: float) -> list[float]:
    low, high = bounds
    count = round((high - low) / step)
    return [round(low + i * step, 4) for i in range(count + 1)]


def grid(accept_step: float, review_step: float, margin_step: float) -> list[Thresholds]:
    return [
        Thresholds(accept, review, margin)
        for accept in steps(ACCEPT_RANGE, accept_step)
        for review in steps(REVIEW_RANGE, review_step)
        if review <= accept
        for margin in steps(MARGIN_RANGE, margin_step)
    ]


def distance(t: Thresholds, current: Thresholds) -> float:
    return round(
        abs(t.accept - current.accept)
        + abs(t.review - current.review)
        + abs(t.margin - current.margin),
        4,
    )


def choose(metrics: Iterable[Metrics], current: Thresholds) -> Metrics | None:
    admissible = [m for m in metrics if m.admissible]
    if not admissible:
        return None
    return min(
        admissible,
        key=lambda m: (
            -m.recall,
            distance(m.thresholds, current),
            -m.thresholds.accept,
            -m.thresholds.margin,
            -m.thresholds.review,
        ),
    )


def load_cases(eval_path: Path, searcher: ExamSearcher) -> list[Case]:
    cases = []
    for line in eval_path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row: dict[str, Any] = json.loads(line)
        cases.append(
            Case(
                query=row["query"],
                kind=row["kind"],
                expected=row["expected"],
                acceptable=tuple(row.get("acceptable", ())),
                result=searcher.search(row["query"], TOP_K),
            )
        )
    return cases


def spec_thresholds(spec_path: Path) -> Thresholds:
    spec = json.loads(spec_path.read_text("utf-8"))
    [params] = [n["params"] for n in spec["workflow"]["nodes"] if n["kind"] in GATE_KINDS]
    return Thresholds(params["accept_threshold"], params["review_threshold"], params["min_margin"])


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def metrics_row(label: str, m: Metrics) -> str:
    return (
        f"| {label} | {m.thresholds} | {pct(m.precision)} | {pct(m.recall)} "
        f"({m.accepted_right}/{m.positives}) | {pct(m.review_rate)} ({m.reviewed}/{m.total}) "
        f"| {m.negatives_accepted} | {m.ambiguous_accepted} | {m.positives_rejected} |"
    )


METRICS_HEADER = (
    "| | accept / review / margin | precisão dos aceitos | recall dos aceitos "
    "| taxa de revisão | negativos aceitos | ambíguos aceitos | positivos rejeitados |\n"
    "|---|---|---|---|---|---|---|---|"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render(  # noqa: PLR0913 - one report, all of its inputs
    cases: Sequence[Case],
    *,
    chosen: Metrics | None,
    current: Metrics,
    fine: Metrics | None,
    admissible: Sequence[Metrics],
    paths: dict[str, Path],
    catalog_version: str,
    steps_used: tuple[float, float, float],
) -> str:
    kinds = Counter(case.kind for case in cases)
    non_negative = [c for c in cases if c.kind != "negative"]
    p_at_1 = sum(c.top_is_right for c in non_negative) / len(non_negative)
    lines = [
        "# Calibração dos thresholds do confidence gate",
        "",
        "Gerado por `scripts/calibrate_rag.py`; não editar à mão.",
        "",
        "## Entradas",
        "",
        f"- Conjunto de avaliação: `{_rel(paths['eval'])}` ({len(cases)} consultas; sha256 "
        f"`{_sha256(paths['eval'])[:16]}…`).",
        f"- Catálogo: `{_rel(paths['catalog'])}`, versão {catalog_version} (sha256 "
        f"`{_sha256(paths['catalog'])[:16]}…`).",
        f"- Thresholds atuais: `{_rel(paths['spec'])}`.",
        "- Consultas por tipo: "
        + ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items()))
        + ".",
        "",
        "## Método",
        "",
        f"Cada consulta é buscada uma vez (top {TOP_K}) com o catálogo e o scorer reais. "
        "A regra do gate é replicada: aceita se `c1 ≥ accept` e (`c2` ausente ou "
        "`c1 - c2 ≥ margin`); revisão se `c1 ≥ review`; senão rejeita.",
        "",
        f"Grade: accept {ACCEPT_RANGE[0]:.2f} a {ACCEPT_RANGE[1]:.2f} (passo {steps_used[0]:.2f}), "
        f"review {REVIEW_RANGE[0]:.2f} a {REVIEW_RANGE[1]:.2f} (passo {steps_used[1]:.2f}, "
        f"review ≤ accept), margin {MARGIN_RANGE[0]:.2f} a {MARGIN_RANGE[1]:.2f} "
        f"(passo {steps_used[2]:.2f}).",
        "",
        "- **Precisão dos aceitos**: aceitos com o código certo / aceitos (ambíguo aceito "
        "conta como erro).",
        "- **Recall dos aceitos**: aceitos certos / consultas com uma única resposta certa "
        "(exceto negativas e ambíguas).",
        "- **Taxa de revisão**: consultas enviadas para revisão / todas as consultas.",
        "- **Positivos rejeitados**: consultas com resposta certa que o gate descartaria "
        "sem revisão.",
        "",
        "Critério: precisão dos aceitos = 100% e zero negativos aceitos; entre essas "
        "combinações, maior recall; empate → a mais próxima dos thresholds atuais.",
        "",
        "## Resultado",
        "",
        METRICS_HEADER,
        metrics_row("atual (spec)", current),
    ]
    if chosen is None:
        lines += ["", "**Nenhuma combinação cumpre o critério.**"]
    else:
        lines.append(metrics_row("**escolhido**", chosen))
        if fine is not None:
            lines.append(metrics_row("grade fina (0,01), só referência", fine))
        same = chosen.thresholds == current.thresholds
        lines += [
            "",
            f"Precision@1 nas consultas não negativas: {pct(p_at_1)} "
            f"({sum(c.top_is_right for c in non_negative)}/{len(non_negative)}).",
            "",
            (
                "Os thresholds escolhidos são os que já estão na spec; nada muda."
                if same
                else f"Os thresholds mudam de {current.thresholds} para {chosen.thresholds}."
            ),
            "",
            f"Combinações admissíveis: {len(admissible)}; com o mesmo recall máximo: "
            f"{sum(1 for m in admissible if m.recall == chosen.recall)}.",
        ]
    lines += ["", *_decisions_table(cases, chosen or current), *_gaps(cases, chosen or current)]
    lines += [
        "",
        "## Limitações",
        "",
        "- O conjunto é pequeno e foi escrito à mão; os thresholds valem para ele. "
        "Pedidos reais devem alimentar o conjunto antes de qualquer ajuste fino.",
        "- O passo de 0,05 em accept/review é proposital: com passo 0,01 o accept cairia no "
        "intervalo estreito entre o negativo de maior score sem empate e o positivo mais fraco.",
        "",
    ]
    return "\n".join(lines)


def _decisions_table(cases: Sequence[Case], m: Metrics) -> list[str]:
    lines = [
        "## Decisões por consulta (thresholds escolhidos)",
        "",
        "| tipo | consulta | esperado | top-1 | score | margem | decisão |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        code, first, second = case.top
        margin = f"{first - second:.4f}" if second is not None else "—"
        expected = case.expected or ("/".join(case.acceptable) if case.acceptable else "—")
        mark = "" if case.kind == "negative" or case.top_is_right else " ✗"
        query = case.query.replace("|", "\\|")
        lines.append(
            f"| {case.kind} | `{query}` | {expected} | {code or '—'}{mark} | {first:.4f} "
            f"| {margin} | {decide(case, m.thresholds)} |"
        )
    return lines


def _gaps(cases: Sequence[Case], m: Metrics) -> list[str]:
    t = m.thresholds
    negatives = sorted(
        (c for c in cases if c.kind == "negative" and c.result.candidates),
        key=lambda c: -c.top[1],
    )[:5]
    weakest = sorted(
        (c for c in cases if c.positive and decide(c, t) == "accepted"), key=lambda c: c.top[1]
    )[:5]
    lines = ["", "## Folgas", "", "Negativos com maior score (top-1):", ""]
    lines += [
        f"- `{c.query}` → {c.top[0]} {c.top[1]:.4f} "
        f"(margem {(c.top[1] - (c.top[2] or 0)):.4f}; {decide(c, t)})"
        for c in negatives
    ]
    lines += ["", "Aceitos com menor score:", ""]
    lines += [f"- `{c.query}` → {c.top[0]} {c.top[1]:.4f}" for c in weakest]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", type=Path, required=True, help="evaluation set (JSONL)")
    parser.add_argument("--out", type=Path, required=True, help="Markdown report to write")
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "data" / "exams_catalog.json")
    parser.add_argument(
        "--spec", type=Path, default=REPO_ROOT / "specs" / "exam_scheduling_agent.json"
    )
    parser.add_argument("--accept-step", type=float, default=0.05)
    parser.add_argument("--review-step", type=float, default=0.05)
    parser.add_argument("--margin-step", type=float, default=0.01)
    args = parser.parse_args(argv)

    catalog = load_catalog(args.catalog)
    cases = load_cases(args.eval, ExamSearcher(catalog))
    current_t = spec_thresholds(args.spec)
    all_metrics = [
        measure(cases, t) for t in grid(args.accept_step, args.review_step, args.margin_step)
    ]
    chosen = choose(all_metrics, current_t)
    fine = choose((measure(cases, t) for t in grid(0.01, 0.01, 0.01)), current_t)
    report = render(
        cases,
        chosen=chosen,
        current=measure(cases, current_t),
        fine=fine,
        admissible=[m for m in all_metrics if m.admissible],
        paths={"eval": args.eval, "catalog": args.catalog, "spec": args.spec},
        catalog_version=catalog.version,
        steps_used=(args.accept_step, args.review_step, args.margin_step),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    if chosen is None:
        print("no threshold combination meets the criterion", file=sys.stderr)
        return 1
    print(
        f"chosen {chosen.thresholds} (spec {current_t}): precision {pct(chosen.precision)}, "
        f"recall {pct(chosen.recall)}, review rate {pct(chosen.review_rate)}, "
        f"negatives accepted {chosen.negatives_accepted}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

import re
import unicodedata
from dataclasses import dataclass

from agent_runtime.schemas import InjectionReport

LOW, MEDIUM, HIGH = 1, 3, 5
# Legitimate OCR output rarely has invisible characters.
INVISIBLE_CHARS_RULE = "invisible_chars"


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    pattern: re.Pattern[str]
    weight: int


def _rule(rule_id: str, weight: int, pattern: str) -> Rule:
    return Rule(rule_id, re.compile(pattern, re.MULTILINE), weight)


RULES: tuple[Rule, ...] = (
    _rule(
        "ignore_instructions",
        HIGH,
        r"\b(ignor\w*|desconsider\w*|esquec\w*|disregard\w*|forget|override|bypass)\b"
        r"[^\n]{0,40}?\b(instruc\w*|regras?|orientac\w*|comandos?|diretrizes|"
        r"instructions?|rules?|prompts?|directions?|guidelines?)\b",
    ),
    _rule(
        "role_switch",
        HIGH,
        r"\b(voce agora e|agora voce e|a partir de agora,? voce|you are now|from now on,? you|"
        r"aja como|finja (ser|que)|pretend (to be|you)|act as (an?|the)|novo papel|new role)\b",
    ),
    _rule(
        "role_tag",
        HIGH,
        r"(^\s*(system|assistant|developer|sistema|assistente)\s*:|<\|?\s*(system|assistant|"
        r"im_start|im_end)\s*\|?>|\[/?inst\])",
    ),
    _rule("delimiter_escape", HIGH, r"<\s*/?\s*documento_nao_confiavel\s*>"),
    _rule(
        "schedule_more",
        MEDIUM,
        r"\b(agende|agendar|marque|marcar|inclua|incluir|adicione|adicionar|schedule|book|add)\b"
        r"[^\n]{0,30}?\b(tambem|todos|todas|todo o|also|all|every)\b",
    ),
    _rule(
        "tool_call",
        MEDIUM,
        r"(\b(chame|chamar|use|usar|execute|executar|call|invoke|run)\b[^\n]{0,20}?"
        r"\b(ferramentas?|funcao|funcoes|tools?|functions?|comandos?|commands?)\b|<\s*tool|"
        r"\btool_call\b|\bfunction_call\b)",
    ),
    _rule(
        "prompt_leak",
        MEDIUM,
        r"\b(revele|mostre|repita|imprima|reveal|show|repeat|print)\b[^\n]{0,30}?"
        r"\b(prompt|instrucoes|instructions|system)\b",
    ),
    _rule(
        "output_override",
        LOW,
        r"\b(responda|retorne|devolva|respond|reply|return|output)\b[^\n]{0,15}?"
        r"\b(apenas|somente|only|exatamente|exactly)\b",
    ),
    _rule("code_fence", LOW, r"```"),
    _rule("url", LOW, r"\b(https?://|www\.)\S+"),
    _rule("encoded_blob", LOW, r"[A-Za-z0-9+/]{40,}={0,2}"),
)


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def severity_for(score: int) -> str:
    if score >= HIGH:
        return "high"

    if score >= MEDIUM:
        return "medium"

    if score >= LOW:
        return "low"

    return "none"


def scan(text: str, removed_chars: int = 0) -> InjectionReport:
    folded = fold(text)
    matched = {rule.id: rule.weight for rule in RULES if rule.pattern.search(folded)}
    if removed_chars > 0:
        matched[INVISIBLE_CHARS_RULE] = LOW

    return InjectionReport.model_validate(
        {"severity": severity_for(sum(matched.values())), "rule_ids": sorted(matched)}
    )

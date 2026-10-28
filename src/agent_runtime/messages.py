"""What `lab-agent run` tells the user when something goes wrong, in Portuguese.

Every message says what happened and what the user can do. Codes stay in
English where they are machine-readable (the JSON report, the logs, the
exceptions); this module is the only place that turns them into text for
the terminal.
"""

import re

MB = 1024 * 1024
CANCELLED = "execução cancelada pelo usuário."

type Message = tuple[str, str]
"""(what happened, what the user can do)."""

REBUILD = "Gere o pacote de novo com lab-transpile e rode outra vez."

_IMAGE: dict[str, Message] = {
    "IMAGE_NOT_FOUND": ("imagem não encontrada", "Confira o caminho passado em --image."),
    "IMAGE_OUTSIDE_INPUT_ROOT": (
        "a imagem está fora da pasta permitida (INPUT_ROOT)",
        "Coloque o arquivo dentro dela, por exemplo em samples/, e rode de novo.",
    ),
    "IMAGE_NOT_A_FILE": (
        "o caminho passado em --image não é um arquivo",
        "Informe o arquivo da imagem, não uma pasta.",
    ),
    "IMAGE_EMPTY": ("o arquivo da imagem está vazio", "Envie outra imagem do pedido."),
    "IMAGE_TOO_LARGE": (
        "a imagem é grande demais",
        "Reduza o tamanho ou a resolução da imagem e rode de novo.",
    ),
    "UNSUPPORTED_FORMAT": ("formato de imagem não suportado", "Envie um arquivo PNG ou JPEG."),
    "INVALID_IMAGE": (
        "o serviço de OCR não conseguiu abrir a imagem",
        "Confira se o arquivo é um PNG ou JPEG legível e não está corrompido.",
    ),
}

_CONFIG: dict[str, Message] = {
    "MISSING_ENV": (
        "faltam variáveis de ambiente",
        "Defina-as no .env (veja o .env.example) e rode de novo.",
    ),
    "INVALID_ENV": (
        "uma variável de ambiente tem valor inválido",
        "Confira o valor no .env (veja o .env.example).",
    ),
    "FAKE_LLM_OUTSIDE_TEST": (
        "AGENT_LLM_BACKEND=fake só é aceito com APP_ENV=test",
        "Remova AGENT_LLM_BACKEND do ambiente para usar o Gemini.",
    ),
}

# Which dependency a failing step talks to, by node name of the shipped specs.
_SERVICES = {
    "ocr": ("o serviço de leitura do pedido (OCR)", "ocr-mcp"),
    "extraction": ("o catálogo de exames (RAG)", "rag-mcp"),
    "lexical_extract": ("o catálogo de exames (RAG)", "rag-mcp"),
    "resolve": ("o catálogo de exames (RAG)", "rag-mcp"),
}
_SERVICES_UP = "Confira se os serviços estão no ar (docker compose ps) e tente de novo."

_RUN: dict[str, Message] = {
    "INPUT_REQUIRED": (
        "o pedido precisa de revisão humana, mas a execução é não interativa",
        "Rode de novo sem --non-interactive para escolher os exames.",
    ),
    "INPUT_CLOSED": (
        "a entrada terminou antes das respostas da revisão",
        "Rode de novo num terminal interativo, ou use --non-interactive.",
    ),
    "PII_TRIPWIRE": (
        "a execução foi interrompida para proteger dados pessoais do pedido",
        "Nada foi agendado. Encaminhe o pedido para conferência manual.",
    ),
    "NODE_TIMEOUT": ("um serviço necessário demorou demais para responder", _SERVICES_UP),
    "LLM_RESOURCE_EXHAUSTED": (
        "a cota do serviço de IA (Gemini) acabou",
        "Espere a cota renovar ou use outra GOOGLE_API_KEY.",
    ),
    "LLM_UNAUTHENTICATED": (
        "o serviço de IA (Gemini) recusou a chave",
        "Confira a GOOGLE_API_KEY no .env.",
    ),
    "LLM_PERMISSION_DENIED": (
        "o serviço de IA (Gemini) recusou a chave",
        "Confira a GOOGLE_API_KEY no .env.",
    ),
    "SCHEDULER_UNAVAILABLE": (
        "a API de agendamento não respondeu",
        "Confira se o scheduler-api está no ar (docker compose ps) e tente de novo.",
    ),
    "SCHEDULER_TIMEOUT": (
        "a API de agendamento demorou demais para responder",
        "Confira se o scheduler-api está no ar (docker compose ps) e tente de novo.",
    ),
    "SCHEDULER_UNEXPECTED_RESPONSE": (
        "a API de agendamento devolveu uma resposta inesperada",
        "Confira a SCHEDULER_API_URL e a SCHEDULER_API_KEY no .env e tente de novo.",
    ),
}
_UNEXPECTED: Message = (
    "ocorreu um erro inesperado e nada foi agendado",
    "Tente de novo; se continuar, rode com --verbose e procure o suporte.",
)


def line(message: Message) -> str:
    """One stderr line: 'what happened. What to do.'"""
    what, action = message
    return f"{what}. {action}"


def image_error(code: str, max_bytes: int | None = None) -> Message:
    what, action = _IMAGE.get(code, ("imagem inválida", "Envie um PNG ou JPEG legível."))
    if code == "IMAGE_TOO_LARGE" and max_bytes:
        what = f"{what} (máximo de {max_bytes // MB} MB)"
    return what, action


def config_error(code: str, detail: str | None = None) -> Message:
    what, action = _CONFIG.get(
        code, (f"configuração inválida ({code})", "Confira o .env e a spec e rode de novo.")
    )
    if detail and code in {"MISSING_ENV", "INVALID_ENV"}:
        what = f"{what}: {detail}"
    return what, action


_LLM_DOWN: Message = (
    "o serviço de IA (Gemini) não respondeu",
    "Tente de novo em alguns minutos; se continuar, confira a GOOGLE_API_KEY e a cota.",
)


def run_error(code: str, node: str | None = None) -> Message:
    """A run that ended without a result; `node` is the step that was running."""
    if code in _RUN:
        return _RUN[code]
    if code in _IMAGE:
        return image_error(code)
    if code in _CONFIG:
        return config_error(code)
    return _LLM_DOWN if code.startswith("LLM_") else _service_error(code, node)


def _service_error(code: str, node: str | None) -> Message:
    service, container = _SERVICES.get(node or "", ("um serviço necessário", ""))
    if code == "MCP_UNAUTHORIZED":
        return (
            f"{service} recusou a credencial",
            "Confira o MCP_AUTH_TOKEN no .env (o mesmo dos servidores MCP).",
        )
    if code.startswith("MCP_"):
        action = (
            f"Confira se o {container} está no ar (docker compose ps) e tente de novo."
            if container
            else _SERVICES_UP
        )
        return f"{service} não respondeu", action
    if code.endswith("_INVALID_RESULT"):
        return f"{service} devolveu uma resposta inesperada", _UNEXPECTED[1]
    return _UNEXPECTED


# argparse builds its messages from these templates (Python 3.12); the
# variable parts (option names, values) are kept as they are.
_ARGPARSE = (
    (r"the following arguments are required: (.+)", r"faltam argumentos obrigatórios: \1"),
    (
        r"argument (.+?): invalid choice: (.+) \(choose from (.+)\)",
        r"argumento \1: opção inválida: \2 (escolha entre \3)",
    ),
    (r"argument (.+?): expected one argument", r"argumento \1: falta o valor"),
    (r"argument (.+?): ignored explicit argument (.+)", r"argumento \1: valor não aceito: \2"),
    (r"argument (.+?): invalid (.+) value: (.+)", r"argumento \1: valor inválido: \3"),
    (r"unrecognized arguments: (.+)", r"argumentos não reconhecidos: \1"),
    (r"ambiguous option: (.+?) could match (.+)", r"opção ambígua: \1 pode ser \2"),
    (r"argument (.+?): not allowed with argument (.+)", r"argumento \1: não combina com \2"),
)


def argparse_error(message: str) -> str:
    for pattern, replacement in _ARGPARSE:
        translated, count = re.subn(f"^{pattern}$", replacement, message)
        if count:
            return translated
    return message

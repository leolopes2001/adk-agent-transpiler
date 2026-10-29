from pii_guard.labeled_field import starts_with_label
from pii_guard.text import tokens

# Field labels that end a name: "Dr. João CRM 1" leaves the CRM to its own detector.
LABELS_ENDING_NAME = frozenset(
    {
        "cpf",
        "rg",
        "cns",
        "crm",
        "cep",
        "tel",
        "telefone",
        "celular",
        "data",
        "nascimento",
        "dn",
        "idade",
        "sexo",
        "convenio",
        "carteirinha",
        "matricula",
        "endereco",
        "email",
        "assinatura",
    }
)
# Words that introduce a person. Not "pai": it is also an exam alias (PAI).
ROLE_LABELS = frozenset(
    {
        "paciente",
        "nome",
        "beneficiario",
        "beneficiaria",
        "sr",
        "sra",
        "srta",
        "dr",
        "dra",
        "medico",
        "medica",
        "solicitante",
        "profissional",
        "responsavel",
        "acompanhante",
        "mae",
    }
)
SECTION_TITLES = frozenset(
    {
        "exames",
        "solicito",
        "solicitados",
        "solicitacao",
        "pedido",
        "indicacao",
        "obs",
        "observacao",
        "diagnostico",
        "hipotese",
        "cid",
    }
)
PERSONAL_LABELS = LABELS_ENDING_NAME | ROLE_LABELS


def has_personal_label(line: str) -> bool:
    """A personal-data label anywhere on the line, or one OCR misread at its start."""
    return starts_with_label(line) or not PERSONAL_LABELS.isdisjoint(tokens(line))

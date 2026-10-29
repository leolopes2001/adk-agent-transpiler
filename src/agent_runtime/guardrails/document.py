import re

# The extractor instruction names these tags as the boundary of untrusted data.
OPEN_TAG = "<documento_nao_confiavel>"
CLOSE_TAG = "</documento_nao_confiavel>"
TAG = re.compile(r"<\s*/?\s*documento_nao_confiavel\s*>", re.IGNORECASE)
_BLOCK = re.compile(f"{re.escape(OPEN_TAG)}(.*?){re.escape(CLOSE_TAG)}", re.DOTALL)


def split_document(text: str) -> tuple[list[str], str]:
    """The document blocks of a model request, and the text around them."""
    return [m[1] for m in _BLOCK.finditer(text)], _BLOCK.sub("", text)

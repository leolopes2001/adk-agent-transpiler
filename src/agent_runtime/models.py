"""Model selection for generated agents."""


def resolve_model(name: str) -> str:
    """Model handed to `Agent(model=...)`.

    For now the name goes straight to ADK, which resolves Gemini models from
    `GOOGLE_API_KEY`. Phase 06 adds the test-only fake backend here.
    """
    return name

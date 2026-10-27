"""Exports the OpenAPI document: `python -m scheduler_api.openapi <path>`.

The versioned copy is `evidence/openapi.json`; a unit test fails when it is stale.

The document only depends on the code, never on configuration, so it is built
from placeholder settings without starting the app.
"""

import json
import sys
from pathlib import Path

from pydantic import SecretStr

from scheduler_api.main import create_app
from scheduler_api.settings import Settings

_PLACEHOLDER = Settings(
    database_url=SecretStr("postgresql+psycopg://openapi@localhost/openapi"),
    scheduler_api_key=SecretStr("openapi-export-placeholder-key-000000"),
    catalog_path=Path("unused.json"),
)


def render() -> str:
    schema = create_app(_PLACEHOLDER).openapi()
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m scheduler_api.openapi <output.json>", file=sys.stderr)
        return 1
    output = Path(args[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

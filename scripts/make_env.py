"""Create or complete the `.env` file that Docker Compose reads.

- Without `.env`: it is created from `.env.example` (comments included), with
  a new random value for every secret; the example's placeholders are public.
- With `.env`: nothing already in it changes. Variables of `.env.example`
  that it lacks are appended, secrets with random values and the others with
  the example's value.

Only variable names are printed, never values. A created file gets mode 0600.
Standard library only, so it runs with any Python 3.9+ without the project
environment (`make env` falls back to a container when there is no python3).

Usage:
    python3 scripts/make_env.py [--example .env.example] [--env .env]
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from pathlib import Path

SECRETS = frozenset({"POSTGRES_PASSWORD", "SCHEDULER_API_KEY", "MCP_AUTH_TOKEN", "PII_HMAC_SECRET"})
"""Generated locally. GOOGLE_API_KEY is a secret too, but only its owner can provide it."""
LLM_KEY = "GOOGLE_API_KEY"

_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def new_secret() -> str:
    # 43 URL-safe characters: valid inside DATABASE_URL and above every 32-character minimum.
    return secrets.token_urlsafe(32)


def assignments(text: str) -> dict[str, str]:
    """Variables assigned in an env file, in order (a repeated name keeps the last value)."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = _ASSIGNMENT.match(line)
        if match:
            found[match.group(1)] = match.group(2).strip()
    return found


def _line(name: str, example_value: str) -> str:
    return f"{name}={new_secret() if name in SECRETS else example_value}"


def create(example: str, env_path: Path) -> list[str]:
    lines = []
    for line in example.splitlines():
        match = _ASSIGNMENT.match(line)
        lines.append(_line(match.group(1), match.group(2).strip()) if match else line)
    # O_EXCL: never replaces a file created in the meantime.
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return sorted(SECRETS.intersection(assignments(example)))


def complete(example: str, env_path: Path) -> list[str]:
    current = env_path.read_text(encoding="utf-8")
    present = assignments(current)
    missing = [(n, v) for n, v in assignments(example).items() if n not in present]
    if missing:
        separator = "" if not current or current.endswith("\n") else "\n"
        added = "\n".join(_line(name, value) for name, value in missing)
        with env_path.open("a", encoding="utf-8") as file:
            file.write(f"{separator}{added}\n")
    return [name for name, _ in missing]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--example", type=Path, default=Path(".env.example"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)

    try:
        example = args.example.read_text(encoding="utf-8")
    except OSError:
        print(f"error: cannot read {args.example}", file=sys.stderr)
        return 1

    if args.env.exists():
        added = complete(example, args.env)
        detail = f"added {', '.join(added)}" if added else "nothing to add"
        print(f"{args.env} exists; existing values kept, {detail}")
    else:
        generated = create(example, args.env)
        print(f"{args.env} created from {args.example}; random values for {', '.join(generated)}")

    final = assignments(args.env.read_text(encoding="utf-8"))
    placeholders = assignments(example)
    empty = sorted(name for name in SECRETS if not final.get(name))
    public = sorted(n for n in SECRETS if final.get(n) and final[n] == placeholders.get(n))
    if empty:
        print(f"warning: empty in {args.env}: {', '.join(empty)}", file=sys.stderr)
    if public:
        print(
            f"warning: still the public placeholder of {args.example}: {', '.join(public)}"
            f" (delete these lines and run again for random values)",
            file=sys.stderr,
        )
    if not final.get(LLM_KEY):
        print(f"note: set {LLM_KEY} in {args.env} to run the agent with Gemini", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

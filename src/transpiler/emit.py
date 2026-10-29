"""Staged writes moved into place with os.replace: a failed run never leaves a partial package."""

import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from transpiler.errors import Code, Diagnostic, Stage, TranspileError

MANIFEST_NAME = "_manifest.json"
_STAGING_PREFIX = ".tmp-"
_PREVIOUS_DIR = ".previous"


class StagedPackage:
    def __init__(self, out_root: Path, name: str, staging: Path) -> None:
        self.out_root = out_root
        self.name = name
        self.staging = staging
        self.path = staging / name
        self.path.mkdir()

    def write(self, filename: str, content: str) -> None:
        if Path(filename).name != filename:
            raise ValueError(f"package files are flat: {filename}")

        (self.path / filename).write_text(content, encoding="utf-8", newline="\n")

    def commit(self) -> Path:
        target = self.out_root / self.name
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise _output_error(f"destino existe e não é um diretório: {target}")
        if target.exists() and not _is_replaceable(target):
            raise _output_error(f"destino existe e não foi gerado pelo lab-transpile: {target}")

        previous = self.staging / _PREVIOUS_DIR
        if target.exists():
            os.replace(target, previous)

        try:
            os.replace(self.path, target)
        except OSError:
            if previous.exists():
                os.replace(previous, target)
            raise

        return target


def resolve_output_root(out: Path, allowed_root: Path) -> Path:
    resolved = out.resolve()
    if not resolved.is_relative_to(allowed_root.resolve()):
        raise _output_error(f"--out precisa estar dentro de {allowed_root}: {out}")

    if resolved.exists() and not resolved.is_dir():
        raise _output_error(f"--out não é um diretório: {out}")

    return resolved


def build_manifest(  # noqa: PLR0913 - one argument per manifest field
    *,
    transpiler_version: str,
    spec_path: str,
    spec_sha256: str,
    description: str,
    ir_digest: str,
    instructions: Mapping[str, str],
    adk_requirement: str,
    node_names: Sequence[str],
) -> str:
    manifest = {
        "transpiler_version": transpiler_version,
        "spec_path": spec_path,
        "spec_sha256": spec_sha256,
        "description": description,
        "ir_digest": ir_digest,
        "instructions": dict(sorted(instructions.items())),
        "adk_requirement": adk_requirement,
        "node_names": sorted(node_names),
    }
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


@contextmanager
def staged_package(out_root: Path, name: str) -> Iterator[StagedPackage]:
    out_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=out_root))
    try:
        yield StagedPackage(out_root, name, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _is_replaceable(target: Path) -> bool:
    """Only an earlier package (it has a manifest) or an empty directory is ever replaced."""
    return (target / MANIFEST_NAME).is_file() or not any(target.iterdir())


def _output_error(message: str) -> TranspileError:
    return TranspileError(
        Stage.IO,
        [Diagnostic(code=Code.E_OUTPUT, message=message)],
    )

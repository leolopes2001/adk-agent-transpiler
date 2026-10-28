from dataclasses import dataclass
from pathlib import Path

from transpiler import __version__, semantic
from transpiler.codegen.render import render
from transpiler.emit import MANIFEST_NAME, build_manifest, resolve_output_root, staged_package
from transpiler.errors import Diagnostic, Stage, fail_on_errors
from transpiler.ir.builder import build_program
from transpiler.lowering import AdkProgram, lower
from transpiler.spec.loader import InstructionFile, LoadedSpec, load_spec, read_instruction
from transpiler.validation.runner import AGENT_FILE, INIT_FILE, PACKAGE_FILES, validate_package


@dataclass(frozen=True, slots=True)
class CompiledSpec:
    loaded: LoadedSpec
    instructions: dict[str, InstructionFile]
    program: AdkProgram
    warnings: tuple[Diagnostic, ...]


@dataclass(frozen=True, slots=True)
class GeneratedPackage:
    compiled: CompiledSpec
    path: Path
    files: tuple[str, ...]


def compile_spec(path: Path) -> CompiledSpec:
    loaded = load_spec(path)
    diagnostics = semantic.validate(loaded.spec, path.parent)
    fail_on_errors(Stage.SEMANTIC, diagnostics)

    instructions = {
        key: read_instruction(path.parent, agent.instruction_file)
        for key, agent in loaded.spec.llm_agents.items()
    }
    program = build_program(loaded.spec, spec_sha256=loaded.sha256, instructions=instructions)
    return CompiledSpec(
        loaded=loaded,
        instructions=instructions,
        program=lower(program),
        warnings=tuple(diagnostics),
    )


def generate(spec_path: Path, out: Path, *, allowed_root: Path) -> GeneratedPackage:
    output_root = resolve_output_root(out, allowed_root)
    compiled = compile_spec(spec_path)
    program = compiled.program
    package = render(program, spec_name=spec_path.name)

    with staged_package(output_root, program.agent_name) as stage:
        stage.write(AGENT_FILE, package.agent_py)
        stage.write(INIT_FILE, package.init_py)
        report = validate_package(stage.path, program)
        stage.write(MANIFEST_NAME, _manifest(spec_path, compiled, report.adk_version))
        target = stage.commit()

    return GeneratedPackage(
        compiled=compiled,
        path=target,
        files=(*PACKAGE_FILES, MANIFEST_NAME),
    )


def _manifest(spec_path: Path, compiled: CompiledSpec, adk_version: str) -> str:
    program = compiled.program
    return build_manifest(
        transpiler_version=__version__,
        spec_path=_display_path(spec_path),
        spec_sha256=program.spec_sha256,
        description=program.description,
        ir_digest=program.ir_digest,
        instructions={key: file.sha256 for key, file in compiled.instructions.items()},
        adk_requirement=f"google-adk=={adk_version}",
        node_names=program.node_names(),
    )


def _display_path(path: Path) -> str:
    """Relative to the cwd, so the manifest does not depend on where the repo is checked out."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name

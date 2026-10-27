import importlib.util
import stat
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "make_env.py"
EXAMPLE = REPO_ROOT / ".env.example"
SECRET_NAMES = ["MCP_AUTH_TOKEN", "PII_HMAC_SECRET", "POSTGRES_PASSWORD", "SCHEDULER_API_KEY"]


@pytest.fixture(scope="module")
def make_env() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_env", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env_path(tmp_path: Path) -> Path:
    return tmp_path / ".env"


def run(make_env: ModuleType, env_path: Path) -> int:
    code: int = make_env.main(["--example", str(EXAMPLE), "--env", str(env_path)])
    return code


def test_example_declares_every_generated_secret(make_env: ModuleType) -> None:
    declared = make_env.assignments(EXAMPLE.read_text())

    assert declared.keys() >= make_env.SECRETS
    assert declared["GOOGLE_API_KEY"] == ""


def test_creates_env_with_random_secrets(make_env: ModuleType, env_path: Path) -> None:
    assert run(make_env, env_path) == 0

    example = make_env.assignments(EXAMPLE.read_text())
    created = make_env.assignments(env_path.read_text())
    assert created.keys() == example.keys()
    for name in SECRET_NAMES:
        assert created[name] != example[name]
        assert len(created[name]) >= 32
    assert {k: v for k, v in created.items() if k not in SECRET_NAMES} == {
        k: v for k, v in example.items() if k not in SECRET_NAMES
    }
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_created_file_keeps_the_example_comments(make_env: ModuleType, env_path: Path) -> None:
    run(make_env, env_path)

    comments = [line for line in EXAMPLE.read_text().splitlines() if line.startswith("#")]
    assert [line for line in env_path.read_text().splitlines() if line.startswith("#")] == comments


def test_secrets_differ_between_files(make_env: ModuleType, tmp_path: Path) -> None:
    first, second = tmp_path / "a.env", tmp_path / "b.env"
    run(make_env, first)
    run(make_env, second)

    a, b = make_env.assignments(first.read_text()), make_env.assignments(second.read_text())
    assert all(a[name] != b[name] for name in SECRET_NAMES)
    assert len({a[name] for name in SECRET_NAMES}) == len(SECRET_NAMES)


def test_existing_env_is_only_appended_to(make_env: ModuleType, env_path: Path) -> None:
    kept = "kept-as-is"
    original = f"# mine\nGOOGLE_API_KEY=user-key\nPOSTGRES_PASSWORD={kept}\nLOG_LEVEL=DEBUG\n"
    env_path.write_text(original)

    assert run(make_env, env_path) == 0

    text = env_path.read_text()
    assert text.startswith(original)
    values = make_env.assignments(text)
    assert values["GOOGLE_API_KEY"] == "user-key"
    assert values["POSTGRES_PASSWORD"] == kept
    assert values["LOG_LEVEL"] == "DEBUG"
    assert values.keys() == make_env.assignments(EXAMPLE.read_text()).keys()
    for name in ["MCP_AUTH_TOKEN", "PII_HMAC_SECRET", "SCHEDULER_API_KEY"]:
        assert len(values[name]) >= 32
    assert values["AGENT_LLM_BACKEND"] == "gemini"


def test_appends_on_a_new_line_when_the_file_lacks_one(
    make_env: ModuleType, env_path: Path
) -> None:
    env_path.write_text("GOOGLE_API_KEY=user-key")

    run(make_env, env_path)

    lines = env_path.read_text().splitlines()
    assert lines[0] == "GOOGLE_API_KEY=user-key"
    assert lines[1].split("=")[0] in make_env.assignments(EXAMPLE.read_text())


def test_complete_env_is_left_byte_for_byte(make_env: ModuleType, env_path: Path) -> None:
    run(make_env, env_path)
    before = env_path.read_bytes()

    run(make_env, env_path)

    assert env_path.read_bytes() == before


def test_existing_empty_secret_is_kept_and_reported(
    make_env: ModuleType, env_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_path.write_text("POSTGRES_PASSWORD=\n")

    run(make_env, env_path)

    assert make_env.assignments(env_path.read_text())["POSTGRES_PASSWORD"] == ""
    assert "empty in" in capsys.readouterr().err


def test_public_placeholders_are_kept_and_reported(
    make_env: ModuleType, env_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_path.write_text(EXAMPLE.read_text())

    run(make_env, env_path)

    assert env_path.read_text() == EXAMPLE.read_text()
    err = capsys.readouterr().err
    assert "public placeholder" in err
    assert all(name in err for name in SECRET_NAMES)


def test_values_are_never_printed(
    make_env: ModuleType, env_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_path.write_text("GOOGLE_API_KEY=user-key-value\n")

    run(make_env, env_path)

    output = capsys.readouterr()
    printed = output.out + output.err
    values = make_env.assignments(env_path.read_text())
    assert all(value not in printed for value in values.values() if len(value) > 8)


def test_missing_example_is_an_error(make_env: ModuleType, tmp_path: Path, env_path: Path) -> None:
    code = make_env.main(["--example", str(tmp_path / "absent"), "--env", str(env_path)])

    assert code == 1
    assert not env_path.exists()

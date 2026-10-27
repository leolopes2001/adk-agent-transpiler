import hashlib
import importlib.util
import io
import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "make_sample_images.py"
SAMPLES = [
    "samples/pedido_medico_ficticio.png",
    "samples/pedido_injecao.png",
    "samples/pedido_ambiguo.png",
]


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_sample_images", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def files(generator: ModuleType) -> dict[str, bytes]:
    built: dict[str, bytes] = generator.build()
    return built


def run_generator(root: Path, hash_seed: str) -> dict[str, str]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    subprocess.run(  # noqa: S603 - fixed interpreter and script path
        [sys.executable, str(SCRIPT), "--root", str(root)],
        check=True,
        capture_output=True,
        env=env,
    )
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def is_valid_cpf(cpf: str) -> bool:
    digits = [int(c) for c in cpf if c.isdigit()]
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    for size in (9, 10):
        total = sum(d * w for d, w in zip(digits[:size], range(size + 1, 1, -1), strict=True))
        if (total * 10 % 11) % 10 != digits[size]:
            return False
    return True


def test_two_runs_produce_identical_files(tmp_path: Path, files: dict[str, bytes]) -> None:
    first = run_generator(tmp_path / "a", hash_seed="1")
    second = run_generator(tmp_path / "b", hash_seed="2")

    assert set(first) == set(files)
    assert first == second


def test_every_generated_file_is_committed(files: dict[str, bytes]) -> None:
    # Presence only: bytes may legitimately differ across Pillow/zlib builds.
    assert [p for p in files if not (REPO_ROOT / p).is_file()] == []


def test_uses_bundled_font_with_license(generator: ModuleType) -> None:
    fonts = REPO_ROOT / "scripts" / "fonts"
    assert generator.FONT_REGULAR.parent == fonts
    assert generator.FONT_BOLD.parent == fonts
    assert generator.FONT_REGULAR.is_file()
    assert generator.FONT_BOLD.is_file()
    assert (fonts / "LICENSE").is_file()


@pytest.mark.parametrize("path", SAMPLES)
def test_samples_are_a4_at_200_dpi(files: dict[str, bytes], path: str) -> None:
    with Image.open(io.BytesIO(files[path])) as image:
        assert image.format == "PNG"
        assert image.size == (1654, 2339)


def test_generated_cpf_has_valid_check_digits(generator: ModuleType) -> None:
    for seed in [generator.SEED, *range(200)]:
        assert is_valid_cpf(generator.generate_cpf(random.Random(seed)))  # noqa: S311


def test_cpf_validator_rejects_wrong_check_digits_and_repeated_digits(
    generator: ModuleType,
) -> None:
    # Guards the validator used above, so a bug shared with the generator cannot
    # make both agree on invalid numbers.
    valid = generator.generate_cpf(random.Random(7))  # noqa: S311
    assert is_valid_cpf(valid)
    for position in (-2, -1):  # each check digit
        chars = list(valid)
        chars[position] = str((int(chars[position]) + 1) % 10)
        assert not is_valid_cpf("".join(chars)), chars
    assert not any(is_valid_cpf(str(d) * 11) for d in range(10))


def test_expected_codes_come_from_catalog(files: dict[str, bytes], generator: ModuleType) -> None:
    expected = json.loads(files["samples/expected/pedido_medico_ficticio.json"])
    catalog = json.loads((REPO_ROOT / "data" / "exams_catalog.json").read_text(encoding="utf-8"))
    name_by_code = {e["code"]: e["name"] for e in catalog["exams"]}

    assert expected["catalog_version"] == catalog["catalog_version"]
    assert [e["mention"] for e in expected["exams"]] == list(generator.MAIN_EXAMS)
    for exam in expected["exams"]:
        assert name_by_code[exam["code"]] == exam["name"]
    pcr = next(e for e in expected["exams"] if e["mention"] == "PCR")
    assert pcr["name"] == "Proteína C reativa"


def test_decompression_bomb_declares_huge_size_in_tiny_file(files: dict[str, bytes]) -> None:
    data = files["tests/data/images/decompression_bomb.png"]
    width, height = struct.unpack(">II", data[16:24])

    assert (width, height) == (100_000, 100_000)
    assert len(data) < 1024
    with pytest.raises(Image.DecompressionBombError):
        Image.open(io.BytesIO(data))


def test_corrupted_png_has_valid_signature_but_fails_to_decode(files: dict[str, bytes]) -> None:
    data = files["tests/data/images/corrupted.png"]

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(OSError, match="truncated"), Image.open(io.BytesIO(data)) as image:
        image.load()


def test_blank_image_is_uniformly_white(files: dict[str, bytes]) -> None:
    with Image.open(io.BytesIO(files["tests/data/images/blank.png"])) as image:
        assert image.getextrema() == (255, 255)


def test_jpeg_variant_is_valid(files: dict[str, bytes]) -> None:
    with Image.open(io.BytesIO(files["tests/data/images/valid.jpg"])) as image:
        image.load()
        assert image.format == "JPEG"

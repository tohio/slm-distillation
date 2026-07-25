from pathlib import Path

import pytest

from distill.utils.env import get_env_value, load_env_file, require_env_value


def test_load_env_file_reads_dotenv(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=test-token\n", encoding="utf-8")

    values = load_env_file(path)

    assert values["HF_TOKEN"] == "test-token"


def test_get_env_value_reads_dotenv_without_os_fallback(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=test-token\n", encoding="utf-8")

    assert get_env_value("HF_TOKEN", path) == "test-token"


def test_require_env_value_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="HF_TOKEN"):
        require_env_value("HF_TOKEN", path)

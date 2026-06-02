from pathlib import Path

import pytest

from distill.utils.env import get_env_value, load_env_file, require_env_value


def test_load_env_file_reads_dotenv(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

    values = load_env_file(path)

    assert values["OPENROUTER_API_KEY"] == "test-key"


def test_get_env_value_reads_dotenv_without_os_fallback(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

    assert get_env_value("OPENROUTER_API_KEY", path) == "test-key"


def test_require_env_value_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        require_env_value("OPENROUTER_API_KEY", path)

from pathlib import Path

import pytest

from distill.utils.env import (
    configure_wandb_environment,
    get_env_value,
    load_env_file,
    require_env_value,
)


def test_load_env_file_reads_dotenv(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=test-token\n", encoding="utf-8")

    values = load_env_file(path)

    assert values["HF_TOKEN"] == "test-token"


def test_env_sample_declares_hugging_face_and_wandb_settings() -> None:
    values = load_env_file(".env.sample")

    assert set(values) == {
        "HF_TOKEN",
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
    }
    assert values["WANDB_PROJECT"] == "slm-distillation"


def test_get_env_value_reads_dotenv_without_os_fallback(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=test-token\n", encoding="utf-8")

    assert get_env_value("HF_TOKEN", path) == "test-token"


def test_require_env_value_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="HF_TOKEN"):
        require_env_value("HF_TOKEN", path)


def test_configure_wandb_environment_enables_authenticated_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_MODE",
        "WANDB_NAME",
        "WANDB_GROUP",
        "WANDB_JOB_TYPE",
    ):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "WANDB_API_KEY=test-key\n"
        "WANDB_PROJECT=test-project\n"
        "WANDB_ENTITY=test-team\n",
        encoding="utf-8",
    )

    report_to = configure_wandb_environment(
        run_name="test-model",
        stage="response-distill",
        env_path=path,
    )

    assert report_to == ["wandb"]
    assert __import__("os").environ["WANDB_PROJECT"] == "test-project"
    assert __import__("os").environ["WANDB_NAME"] == (
        "test-model-response-distill"
    )


def test_configure_wandb_environment_is_disabled_without_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    assert configure_wandb_environment(
        run_name="test-model",
        stage="response-distill",
        env_path=tmp_path / "missing.env",
    ) == []

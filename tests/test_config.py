from pathlib import Path

import pytest

from distill.utils.config import get_default_teacher, load_teachers_config, load_yaml


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("name: test\n", encoding="utf-8")

    assert load_yaml(path) == {"name": "test"}


def test_load_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


def test_load_teachers_config_reads_default_teacher() -> None:
    config = load_teachers_config("configs/teachers.yaml")
    teacher = get_default_teacher(config)

    assert config.default_teacher == "deepseek_v4_flash"
    assert teacher.provider == "openrouter"
    assert teacher.model == "deepseek/deepseek-v4-flash"
    assert teacher.mode == "response"
    assert teacher.distillation_allowed is True


def test_load_teachers_config_rejects_missing_default(tmp_path: Path) -> None:
    path = tmp_path / "teachers.yaml"
    path.write_text(
        """
default_teacher: missing

teachers:
  valid:
    provider: openrouter
    model: example/model
    mode: response
    purpose: test
    distillation_allowed: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_teacher"):
        load_teachers_config(path)

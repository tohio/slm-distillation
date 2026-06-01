from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TeacherConfig:
    name: str
    provider: str
    model: str
    mode: str
    purpose: str
    distillation_allowed: bool | str


@dataclass(frozen=True)
class TeachersConfig:
    default_teacher: str
    teachers: dict[str, TeacherConfig]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return data


def load_teachers_config(path: str | Path) -> TeachersConfig:
    data = load_yaml(path)

    default_teacher = data.get("default_teacher")
    teachers_raw = data.get("teachers")

    if not isinstance(default_teacher, str) or not default_teacher:
        raise ValueError("teachers config requires non-empty 'default_teacher'")

    if not isinstance(teachers_raw, dict) or not teachers_raw:
        raise ValueError("teachers config requires non-empty 'teachers' mapping")

    teachers: dict[str, TeacherConfig] = {}

    for name, raw in teachers_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"teacher '{name}' must be a mapping")

        missing = [
            field
            for field in (
                "provider",
                "model",
                "mode",
                "purpose",
                "distillation_allowed",
            )
            if field not in raw
        ]

        if missing:
            raise ValueError(f"teacher '{name}' missing required fields: {missing}")

        teachers[name] = TeacherConfig(
            name=name,
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            mode=str(raw["mode"]),
            purpose=str(raw["purpose"]),
            distillation_allowed=raw["distillation_allowed"],
        )

    if default_teacher not in teachers:
        raise ValueError(
            f"default_teacher '{default_teacher}' not found in teachers mapping"
        )

    return TeachersConfig(default_teacher=default_teacher, teachers=teachers)


def get_default_teacher(config: TeachersConfig) -> TeacherConfig:
    return config.teachers[config.default_teacher]

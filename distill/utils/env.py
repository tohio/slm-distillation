from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


DEFAULT_ENV_PATH = ".env"


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    env_path = Path(path)

    if not env_path.exists():
        return {}

    values = dotenv_values(env_path)

    return {
        str(key): str(value)
        for key, value in values.items()
        if key is not None and value is not None
    }


def get_env_value(
    key: str,
    env_path: str | Path = DEFAULT_ENV_PATH,
    fallback_to_os: bool = False,
) -> str | None:
    values = load_env_file(env_path)

    if key in values and values[key]:
        return values[key]

    if fallback_to_os:
        return os.getenv(key)

    return None


def require_env_value(
    key: str,
    env_path: str | Path = DEFAULT_ENV_PATH,
    fallback_to_os: bool = False,
) -> str:
    value = get_env_value(
        key=key,
        env_path=env_path,
        fallback_to_os=fallback_to_os,
    )

    if not value:
        raise ValueError(f"{key} is required in {env_path}")

    return value

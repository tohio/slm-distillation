from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


DEFAULT_ENV_PATH = ".env"
WANDB_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_MODE",
)


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


def configure_wandb_environment(
    *,
    run_name: str,
    stage: str,
    env_path: str | Path = DEFAULT_ENV_PATH,
) -> list[str]:
    """Load optional W&B settings and return Trainer reporting targets."""
    values = load_env_file(env_path)
    for key in WANDB_ENV_KEYS:
        value = values.get(key)
        if value and not os.getenv(key):
            os.environ[key] = value

    mode = os.getenv("WANDB_MODE", "").strip().lower()
    api_key = os.getenv("WANDB_API_KEY", "").strip()
    enabled = bool(api_key) or mode == "offline"
    if not enabled or mode == "disabled":
        return []

    os.environ.setdefault("WANDB_PROJECT", "slm-distillation")
    os.environ.setdefault("WANDB_NAME", f"{run_name}-{stage}")
    os.environ.setdefault("WANDB_GROUP", run_name)
    os.environ.setdefault("WANDB_JOB_TYPE", stage)
    return ["wandb"]

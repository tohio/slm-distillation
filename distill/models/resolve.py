from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from distill.utils.env import get_env_value


@dataclass(frozen=True)
class ModelResolution:
    reference: str
    source: str
    resolved_id: str
    revision: str | None
    commit_sha: str | None


ModelInfoLoader = Callable[..., Any]


def _looks_like_local_path(reference: str) -> bool:
    return reference.startswith((".", "/", "~"))


def resolve_model_reference(
    reference: str,
    *,
    revision: str | None = None,
    model_info_loader: ModelInfoLoader | None = None,
) -> ModelResolution:
    local_path = Path(reference).expanduser()
    if local_path.exists():
        if not local_path.is_dir():
            raise ValueError(f"Model path must be a directory: {local_path}")
        return ModelResolution(
            reference=reference,
            source="local",
            resolved_id=str(local_path.resolve()),
            revision=None,
            commit_sha=None,
        )

    if _looks_like_local_path(reference):
        raise FileNotFoundError(f"Model path not found: {local_path}")

    if model_info_loader is None:
        from huggingface_hub import HfApi

        model_info_loader = HfApi().model_info

    token = get_env_value("HF_TOKEN", fallback_to_os=True)
    info = model_info_loader(
        repo_id=reference,
        revision=revision,
        token=token,
    )

    return ModelResolution(
        reference=reference,
        source="huggingface",
        resolved_id=str(getattr(info, "id", reference)),
        revision=revision,
        commit_sha=getattr(info, "sha", None),
    )

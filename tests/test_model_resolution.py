from pathlib import Path
from types import SimpleNamespace

import pytest

from distill.models.resolve import (
    build_model_load_kwargs,
    resolve_model_reference,
)


def test_build_model_load_kwargs_uses_transformers_5_dtype() -> None:
    kwargs = build_model_load_kwargs(
        revision="main",
        token="secret",
        dtype="bfloat16",
    )

    assert kwargs == {
        "revision": "main",
        "token": "secret",
        "dtype": "bfloat16",
    }
    assert "torch_dtype" not in kwargs


def test_resolve_model_reference_accepts_local_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    result = resolve_model_reference(str(model_path))

    assert result.source == "local"
    assert result.resolved_id == str(model_path.resolve())
    assert result.commit_sha is None


def test_resolve_model_reference_resolves_hugging_face_id() -> None:
    calls = []

    def model_info_loader(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="example/student", sha="abc123")

    result = resolve_model_reference(
        "example/student",
        revision="main",
        model_info_loader=model_info_loader,
    )

    assert result.source == "huggingface"
    assert result.resolved_id == "example/student"
    assert result.commit_sha == "abc123"
    assert calls[0]["repo_id"] == "example/student"
    assert calls[0]["revision"] == "main"


def test_resolve_model_reference_rejects_missing_local_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Model path not found"):
        resolve_model_reference(str(tmp_path / "missing"))

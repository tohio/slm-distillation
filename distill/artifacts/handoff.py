from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from distill.utils.env import get_env_value


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactConfig:
    run_name: str
    repo_id: str
    repo_type: str
    local_dir: str
    bundle_path: str
    required: list[str]
    include: list[str]


@dataclass(frozen=True)
class ArtifactManifest:
    run_name: str
    created_at: str
    files: list[ArtifactFile]


@dataclass(frozen=True)
class ArtifactResult:
    run_name: str
    file_count: int
    total_bytes: int
    manifest_path: str
    bundle_path: str | None = None
    repo_id: str | None = None


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact config requires non-empty string '{key}'")
    return value


def _hf_token() -> str | None:
    return get_env_value("HF_TOKEN") or get_env_value("HUGGINGFACE_HUB_TOKEN")


def _require_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"artifact config requires non-empty list '{key}'")

    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"artifact config '{key}' item {index} must be a string")
        result.append(item)

    return result


def load_artifact_config(path: str | Path) -> ArtifactConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    artifact = data.get("artifact")

    if not isinstance(artifact, dict):
        raise ValueError("artifact config requires an 'artifact' mapping")

    return ArtifactConfig(
        run_name=_require_str(artifact, "run_name"),
        repo_id=_require_str(artifact, "repo_id"),
        repo_type=_require_str(artifact, "repo_type"),
        local_dir=_require_str(artifact, "local_dir"),
        bundle_path=_require_str(artifact, "bundle_path"),
        required=_require_str_list(artifact, "required"),
        include=_require_str_list(artifact, "include"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifact_files(
    include: list[str],
    *,
    root: str | Path = ".",
) -> list[Path]:
    root_path = Path(root)
    collected: dict[str, Path] = {}

    for pattern in include:
        for path in root_path.glob(pattern):
            if path.is_file():
                relative = path.relative_to(root_path).as_posix()
                collected[relative] = path

    return [collected[key] for key in sorted(collected)]


def validate_required_artifacts(
    required: list[str],
    *,
    root: str | Path = ".",
) -> None:
    root_path = Path(root)
    missing = [item for item in required if not (root_path / item).is_file()]

    if missing:
        formatted = ", ".join(missing)
        raise FileNotFoundError(f"Missing required artifact file(s): {formatted}")


def build_manifest(
    *,
    run_name: str,
    files: list[Path],
    root: str | Path = ".",
) -> ArtifactManifest:
    root_path = Path(root)
    entries = [
        ArtifactFile(
            path=path.relative_to(root_path).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in files
    ]

    return ArtifactManifest(
        run_name=run_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        files=entries,
    )


def write_manifest(path: str | Path, manifest: ArtifactManifest) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_manifest(path: str | Path) -> ArtifactManifest:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = [
        ArtifactFile(
            path=item["path"],
            size_bytes=int(item["size_bytes"]),
            sha256=item["sha256"],
        )
        for item in raw.get("files", [])
    ]

    return ArtifactManifest(
        run_name=raw["run_name"],
        created_at=raw["created_at"],
        files=files,
    )


def verify_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path = ".",
) -> ArtifactResult:
    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file)
    root_path = Path(root)

    total_bytes = 0
    for item in manifest.files:
        path = root_path / item.path
        if not path.exists():
            raise FileNotFoundError(f"Manifest file missing: {item.path}")
        actual_size = path.stat().st_size
        if actual_size != item.size_bytes:
            raise ValueError(
                f"Artifact size mismatch for {item.path}: "
                f"expected {item.size_bytes}, got {actual_size}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != item.sha256:
            raise ValueError(
                f"Artifact checksum mismatch for {item.path}: "
                f"expected {item.sha256}, got {actual_sha}"
            )
        total_bytes += actual_size

    return ArtifactResult(
        run_name=manifest.run_name,
        file_count=len(manifest.files),
        total_bytes=total_bytes,
        manifest_path=str(manifest_file),
    )


def stage_artifacts(
    config: ArtifactConfig,
    *,
    root: str | Path = ".",
) -> ArtifactResult:
    root_path = Path(root)
    validate_required_artifacts(config.required, root=root_path)

    local_dir = root_path / config.local_dir
    if local_dir.exists():
        shutil.rmtree(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    files = collect_artifact_files(config.include, root=root_path)
    manifest = build_manifest(run_name=config.run_name, files=files, root=root_path)

    for source in files:
        relative = source.relative_to(root_path)
        target = local_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest_path = write_manifest(local_dir / "manifest.json", manifest)
    total_bytes = sum(item.size_bytes for item in manifest.files)

    return ArtifactResult(
        run_name=config.run_name,
        file_count=len(manifest.files),
        total_bytes=total_bytes,
        manifest_path=str(manifest_path),
    )


def pack_artifacts(
    config_path: str | Path,
    *,
    root: str | Path = ".",
) -> ArtifactResult:
    config = load_artifact_config(config_path)
    root_path = Path(root)
    result = stage_artifacts(config, root=root_path)

    bundle_path = root_path / config.bundle_path
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    local_dir = root_path / config.local_dir
    with tarfile.open(bundle_path, "w:gz") as archive:
        archive.add(local_dir, arcname=config.run_name)

    return ArtifactResult(
        run_name=result.run_name,
        file_count=result.file_count,
        total_bytes=result.total_bytes,
        manifest_path=result.manifest_path,
        bundle_path=str(bundle_path),
    )


def unpack_artifacts(
    bundle_path: str | Path,
    *,
    target_dir: str | Path = ".",
) -> Path:
    bundle = Path(bundle_path)
    target = Path(target_dir)

    if not bundle.exists():
        raise FileNotFoundError(f"Artifact bundle not found: {bundle}")

    target.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(target)

    return target


def push_artifacts(
    config_path: str | Path,
    *,
    root: str | Path = ".",
) -> ArtifactResult:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for push-artifacts") from exc

    config = load_artifact_config(config_path)
    root_path = Path(root)
    result = stage_artifacts(config, root=root_path)
    token = _hf_token()

    api = HfApi(token=token)
    api.create_repo(
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        folder_path=str(root_path / config.local_dir),
        path_in_repo=config.run_name,
    )

    return ArtifactResult(
        run_name=result.run_name,
        file_count=result.file_count,
        total_bytes=result.total_bytes,
        manifest_path=result.manifest_path,
        repo_id=config.repo_id,
    )


def pull_artifacts(
    config_path: str | Path,
    *,
    root: str | Path = ".",
) -> ArtifactResult:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for pull-artifacts") from exc

    config = load_artifact_config(config_path)
    root_path = Path(root)
    token = _hf_token()

    snapshot_path = Path(
        snapshot_download(
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            allow_patterns=[f"{config.run_name}/**"],
            token=token,
        )
    )

    source_run_dir = snapshot_path / config.run_name
    if not source_run_dir.exists():
        raise FileNotFoundError(
            f"Run artifact folder not found in downloaded snapshot: {source_run_dir}"
        )

    copied_files = 0
    total_bytes = 0
    for source in source_run_dir.rglob("*"):
        if source.is_file():
            relative = source.relative_to(source_run_dir)
            target = root_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_files += 1
            total_bytes += target.stat().st_size

    manifest_path = root_path / "manifest.json"
    if manifest_path.exists():
        verify_manifest(manifest_path, root=root_path)

    return ArtifactResult(
        run_name=config.run_name,
        file_count=copied_files,
        total_bytes=total_bytes,
        manifest_path=str(manifest_path),
        repo_id=config.repo_id,
    )

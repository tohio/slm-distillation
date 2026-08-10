import json
from pathlib import Path

import pytest

from distill.artifacts.handoff import (
    collect_artifact_files,
    load_artifact_config,
    pack_artifacts,
    parse_s3_uri,
    resolve_s3_uri,
    stage_artifacts,
    unpack_artifacts,
    verify_manifest,
)


def test_load_artifact_config_reads_default_file() -> None:
    config = load_artifact_config("configs/artifacts.yaml")

    assert config.run_name == "smollm2-135m-response-distilled"
    assert config.backend == "s3"
    assert config.s3_bucket_env == "S3_BUCKET"
    assert config.s3_prefix_env == "S3_PREFIX"
    assert config.delete_remote_extra is True
    assert (
        "runs/smollm2-135m-response-distilled/dpo/checkpoints/final/config.json"
        in config.required
    )
    assert (
        "runs/smollm2-135m-response-distilled/dpo/checkpoints/final/*"
        in config.include
    )


def test_load_logit_artifact_config_uses_logit_branch() -> None:
    config = load_artifact_config("configs/artifacts_logit.yaml")

    assert config.run_name == "smollm2-135m-logit-distilled"
    assert (
        "runs/smollm2-135m-logit-distilled/dpo/checkpoints/final/config.json"
        in config.required
    )


def test_parse_s3_uri() -> None:
    location = parse_s3_uri("s3://my-bucket/slm-distillation/run")

    assert location.bucket == "my-bucket"
    assert location.prefix == "slm-distillation/run/"


def test_resolve_s3_uri_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path("configs/artifacts.yaml").resolve()
    (tmp_path / ".env").write_text(
        "S3_BUCKET=test-bucket\nS3_PREFIX=distillation\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = load_artifact_config(config_path)
    uri = resolve_s3_uri(config)

    assert (
        uri
        == "s3://test-bucket/distillation/smollm2-135m-response-distilled/"
    )


def test_collect_artifact_files_uses_include_patterns(tmp_path: Path) -> None:
    (tmp_path / "runs/test/checkpoints/final").mkdir(parents=True)
    (tmp_path / "runs/test/checkpoints/final/config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    files = collect_artifact_files(["runs/test/checkpoints/final/*"], root=tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "runs/test/checkpoints/final/config.json"
    ]


def test_stage_artifacts_requires_required_files(tmp_path: Path) -> None:
    config_path = tmp_path / "artifacts.yaml"
    config_path.write_text(
        '''
artifact:
  run_name: test-run
  backend: s3
  s3_bucket_env: S3_BUCKET
  s3_prefix_env: S3_PREFIX
  local_dir: artifacts/test-run
  bundle_path: artifacts/test-run.tar.gz
  delete_remote_extra: true
  required:
    - runs/test/checkpoints/final/config.json
  include:
    - runs/test/checkpoints/final/*
''',
        encoding="utf-8",
    )

    config = load_artifact_config(config_path)

    with pytest.raises(FileNotFoundError, match="Missing required artifact"):
        stage_artifacts(config, root=tmp_path)


def test_stage_artifacts_writes_manifest_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "runs/test/checkpoints/final").mkdir(parents=True)
    (tmp_path / "runs/test/checkpoints/final/config.json").write_text(
        "{\"model_type\": \"test\"}\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "artifacts.yaml"
    config_path.write_text(
        '''
artifact:
  run_name: test-run
  backend: s3
  s3_bucket_env: S3_BUCKET
  s3_prefix_env: S3_PREFIX
  local_dir: artifacts/test-run
  bundle_path: artifacts/test-run.tar.gz
  delete_remote_extra: true
  required:
    - runs/test/checkpoints/final/config.json
  include:
    - runs/test/checkpoints/final/*
''',
        encoding="utf-8",
    )

    config = load_artifact_config(config_path)
    result = stage_artifacts(config, root=tmp_path)
    manifest_path = tmp_path / "artifacts/test-run/manifest.json"

    assert result.file_count == 1
    assert manifest_path.exists()

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["files"][0]["path"] == "runs/test/checkpoints/final/config.json"

    verify_result = verify_manifest(manifest_path, root=tmp_path)
    assert verify_result.file_count == 1


def test_pack_and_unpack_artifacts(tmp_path: Path) -> None:
    (tmp_path / "runs/test/checkpoints/final").mkdir(parents=True)
    (tmp_path / "runs/test/checkpoints/final/config.json").write_text(
        "{\"model_type\": \"test\"}\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "artifacts.yaml"
    config_path.write_text(
        '''
artifact:
  run_name: test-run
  backend: s3
  s3_bucket_env: S3_BUCKET
  s3_prefix_env: S3_PREFIX
  local_dir: artifacts/test-run
  bundle_path: artifacts/test-run.tar.gz
  delete_remote_extra: true
  required:
    - runs/test/checkpoints/final/config.json
  include:
    - runs/test/checkpoints/final/*
''',
        encoding="utf-8",
    )

    bundle_path = pack_artifacts(config_path, root=tmp_path).bundle_path
    target = tmp_path / "unpacked"

    assert bundle_path is not None
    unpack_artifacts(bundle_path, target_dir=target)

    assert (target / "test-run/runs/test/checkpoints/final/config.json").exists()

import json
from pathlib import Path

import pytest

from distill.artifacts.handoff import (
    collect_artifact_files,
    load_artifact_config,
    pack_artifacts,
    stage_artifacts,
    unpack_artifacts,
    verify_manifest,
)


def test_load_artifact_config_reads_default_file() -> None:
    config = load_artifact_config("configs/artifacts.yaml")

    assert config.run_name == "slm-125m-deepseek-distilled"
    assert config.repo_id == "tohio/slm-distillation-artifacts"
    assert config.repo_type == "dataset"
    assert "data/distill/response_distill.jsonl" in config.required
    assert "data/distill/*.jsonl" in config.include


def test_collect_artifact_files_uses_include_patterns(tmp_path: Path) -> None:
    (tmp_path / "data/distill").mkdir(parents=True)
    (tmp_path / "data/distill/response_distill.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (tmp_path / "data/raw_teacher").mkdir(parents=True)
    (tmp_path / "data/raw_teacher/raw.jsonl").write_text("{}\n", encoding="utf-8")

    files = collect_artifact_files(["data/distill/*.jsonl"], root=tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "data/distill/response_distill.jsonl"
    ]


def test_stage_artifacts_requires_required_files(tmp_path: Path) -> None:
    config_path = tmp_path / "artifacts.yaml"
    config_path.write_text(
        '''
artifact:
  run_name: test-run
  repo_id: tohio/test-artifacts
  repo_type: dataset
  local_dir: artifacts/test-run
  bundle_path: artifacts/test-run.tar.gz
  required:
    - data/distill/response_distill.jsonl
  include:
    - data/distill/*.jsonl
''',
        encoding="utf-8",
    )

    config = load_artifact_config(config_path)

    with pytest.raises(FileNotFoundError, match="Missing required artifact"):
        stage_artifacts(config, root=tmp_path)


def test_stage_artifacts_writes_manifest_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "data/distill").mkdir(parents=True)
    (tmp_path / "data/distill/response_distill.jsonl").write_text(
        "{\"instruction\": \"x\"}\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "artifacts.yaml"
    config_path.write_text(
        '''
artifact:
  run_name: test-run
  repo_id: tohio/test-artifacts
  repo_type: dataset
  local_dir: artifacts/test-run
  bundle_path: artifacts/test-run.tar.gz
  required:
    - data/distill/response_distill.jsonl
  include:
    - data/distill/*.jsonl
''',
        encoding="utf-8",
    )

    config = load_artifact_config(config_path)
    result = stage_artifacts(config, root=tmp_path)
    manifest_path = tmp_path / "artifacts/test-run/manifest.json"

    assert result.file_count == 1
    assert manifest_path.exists()

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["files"][0]["path"] == "data/distill/response_distill.jsonl"

    verify_result = verify_manifest(manifest_path, root=tmp_path)
    assert verify_result.file_count == 1


def test_pack_and_unpack_artifacts(tmp_path: Path) -> None:
    (tmp_path / "data/preference").mkdir(parents=True)
    (tmp_path / "data/preference/dpo_pairs.jsonl").write_text(
        "{\"prompt\": \"x\"}\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "artifacts.yaml"
    config_path.write_text(
        '''
artifact:
  run_name: test-run
  repo_id: tohio/test-artifacts
  repo_type: dataset
  local_dir: artifacts/test-run
  bundle_path: artifacts/test-run.tar.gz
  required:
    - data/preference/dpo_pairs.jsonl
  include:
    - data/preference/*.jsonl
''',
        encoding="utf-8",
    )

    bundle_path = pack_artifacts(config_path, root=tmp_path).bundle_path
    target = tmp_path / "unpacked"

    assert bundle_path is not None
    unpack_artifacts(bundle_path, target_dir=target)

    assert (target / "test-run/data/preference/dpo_pairs.jsonl").exists()

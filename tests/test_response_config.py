from pathlib import Path

import pytest

from distill.training.train_response_distill import build_response_training_plan
from distill.utils.config import load_response_distill_config


def test_load_response_distill_config_reads_default_file() -> None:
    config = load_response_distill_config("configs/response_distill.yaml")

    assert (
        config.model.model_name_or_path
        == "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert config.model.tokenizer_name_or_path == config.model.model_name_or_path
    assert config.model.revision == "main"
    assert config.data.dataset_id == "tohio/slm-synthetic-distillation-sft"
    assert config.data.dataset_split == "train"
    assert config.data.prompt_field == "prompt"
    assert config.data.response_field == "response"


def test_build_response_training_plan() -> None:
    config = load_response_distill_config("configs/response_distill.yaml")
    plan = build_response_training_plan(config)

    assert plan.model_name_or_path == config.model.model_name_or_path
    assert plan.tokenizer_name_or_path == config.model.tokenizer_name_or_path
    assert plan.dataset_id == config.data.dataset_id
    assert plan.dataset_split == config.data.dataset_split
    assert plan.final_checkpoint_dir == config.output.final_checkpoint_dir


def test_load_response_distill_config_requires_dataset_fields(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "response.yaml"
    config_path.write_text(
        """
model:
  model_name_or_path: example/student
  tokenizer_name_or_path: example/student
  revision: main
data:
  dataset_id: example/data
  dataset_split: train
  id_field: id
  prompt_field: prompt
  metadata_field: metadata
output:
  model_name: example
  run_dir: runs/example
  checkpoint_dir: runs/example/checkpoints
  final_checkpoint_dir: runs/example/checkpoints/final
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="response_field"):
        load_response_distill_config(config_path)

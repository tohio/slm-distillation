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
    assert config.formatting.mode == "chat"
    assert config.training.max_length == 1024
    assert config.training.per_device_train_batch_size == 8
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.max_steps is None


def test_build_response_training_plan() -> None:
    config = load_response_distill_config("configs/response_distill.yaml")
    plan = build_response_training_plan(config)

    assert plan.model_name_or_path == config.model.model_name_or_path
    assert plan.tokenizer_name_or_path == config.model.tokenizer_name_or_path
    assert plan.dataset_id == config.data.dataset_id
    assert plan.dataset_split == config.data.dataset_split
    assert plan.formatting_mode == "chat"
    assert plan.max_length == 1024
    assert plan.learning_rate == pytest.approx(0.00002)
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
formatting:
  mode: chat
  system_prompt: null
training:
  max_length: 1024
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 0.00002
  num_train_epochs: 3
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 1.0
  lr_scheduler_type: cosine
  logging_steps: 10
  save_steps: 250
  save_total_limit: 2
  bf16: true
  gradient_checkpointing: true
  dataloader_num_workers: 4
  seed: 42
  max_steps: null
  max_train_samples: null
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


def test_load_response_distill_config_rejects_unknown_formatting_mode(
    tmp_path: Path,
) -> None:
    config_text = Path("configs/response_distill.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "response.yaml"
    config_path.write_text(
        config_text.replace("mode: chat", "mode: unsupported"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="formatting.mode"):
        load_response_distill_config(config_path)

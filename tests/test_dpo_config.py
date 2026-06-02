
from pathlib import Path

import pytest

from distill.training.train_dpo import build_dpo_training_plan
from distill.utils.config import load_dpo_config


def test_load_dpo_config_reads_default_file() -> None:
    config = load_dpo_config("configs/dpo.yaml")

    assert config.source.model_name == "slm-125m-deepseek-distilled"
    assert config.source.checkpoint_path == (
        "runs/slm-125m-deepseek-distilled/response_distill/checkpoints/final"
    )
    assert config.data.preference_dataset_path == "data/preference/dpo_pairs.jsonl"
    assert config.training.method == "dpo"
    assert config.training.beta == 0.1
    assert config.training.bf16 is True
    assert config.output.final_checkpoint_dir == (
        "runs/slm-125m-deepseek-distilled/dpo/checkpoints/final"
    )


def test_build_dpo_training_plan() -> None:
    config = load_dpo_config("configs/dpo.yaml")
    plan = build_dpo_training_plan(config)

    assert plan.source_checkpoint == config.source.checkpoint_path
    assert plan.tokenizer_path == config.source.tokenizer_path
    assert plan.preference_dataset_path == config.data.preference_dataset_path
    assert plan.output_dir == config.output.checkpoint_dir
    assert plan.final_checkpoint_dir == config.output.final_checkpoint_dir
    assert plan.beta == config.training.beta


def test_load_dpo_config_rejects_wrong_method(tmp_path: Path) -> None:
    path = tmp_path / "dpo.yaml"
    content = (
        "source:\n"
        "  model_name: slm-test\n"
        "  checkpoint_path: checkpoint\n"
        "  tokenizer_path: tokenizer\n"
        "\n"
        "data:\n"
        "  preference_dataset_path: preference.jsonl\n"
        "  rejected_path: rejected.jsonl\n"
        "\n"
        "training:\n"
        "  method: sft\n"
        "  beta: 0.1\n"
        "  max_length: 1024\n"
        "  max_prompt_length: 512\n"
        "  per_device_train_batch_size: 1\n"
        "  gradient_accumulation_steps: 8\n"
        "  learning_rate: 0.000005\n"
        "  num_train_epochs: 1\n"
        "  warmup_ratio: 0.03\n"
        "  bf16: true\n"
        "  seed: 42\n"
        "\n"
        "output:\n"
        "  model_name: slm-test\n"
        "  run_dir: runs/slm-test/dpo\n"
        "  checkpoint_dir: runs/slm-test/dpo/checkpoints\n"
        "  final_checkpoint_dir: runs/slm-test/dpo/checkpoints/final\n"
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="training.method='dpo'"):
        load_dpo_config(path)

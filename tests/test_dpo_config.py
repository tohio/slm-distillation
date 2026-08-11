
from pathlib import Path

import pytest

from distill.training.train_dpo import build_dpo_training_plan
from distill.utils.config import (
    load_dpo_config,
    load_logit_distill_config,
    load_response_distill_config,
)


def test_load_dpo_config_reads_default_file() -> None:
    config = load_dpo_config("configs/dpo.yaml")

    assert config.source.model_name == "smollm2-135m-response-distilled"
    assert config.source.checkpoint_path == (
        "runs/smollm2-135m-response-distilled/response_distill/checkpoints/final"
    )
    assert config.source.tokenizer_path == config.source.checkpoint_path
    assert config.source.revision is None
    assert config.data.dataset_id == "tohio/slm-synthetic-distillation-dpo"
    assert config.data.dataset_split == "train"
    assert config.data.prompt_field == "prompt"
    assert config.data.chosen_field == "chosen"
    assert config.data.rejected_field == "rejected"
    assert config.training.method == "dpo"
    assert config.training.beta == 0.1
    assert config.training.loss_type == ["sigmoid"]
    assert config.training.loss_weights == [1.0]
    assert config.training.max_length == 1024
    assert config.training.max_steps is None
    assert config.training.bf16 is True
    assert config.output.final_checkpoint_dir == (
        "runs/smollm2-135m-response-distilled/dpo/checkpoints/final"
    )


def test_build_dpo_training_plan() -> None:
    config = load_dpo_config("configs/dpo.yaml")
    plan = build_dpo_training_plan(config)

    assert plan.source_checkpoint == config.source.checkpoint_path
    assert plan.tokenizer_path == config.source.tokenizer_path
    assert plan.dataset_id == config.data.dataset_id
    assert plan.dataset_split == config.data.dataset_split
    assert plan.output_dir == config.output.checkpoint_dir
    assert plan.final_checkpoint_dir == config.output.final_checkpoint_dir
    assert plan.beta == config.training.beta
    assert plan.loss_type == ["sigmoid"]
    assert plan.loss_weights == [1.0]
    assert plan.max_length == 1024
    assert plan.max_steps is None


def test_logit_dpo_config_consumes_logit_final_checkpoint() -> None:
    config = load_dpo_config("configs/dpo_logit.yaml")

    assert config.source.checkpoint_path == (
        "runs/smollm2-135m-logit-distilled/"
        "logit_distill/checkpoints/final"
    )
    assert config.source.tokenizer_path == config.source.checkpoint_path
    assert config.output.final_checkpoint_dir == (
        "runs/smollm2-135m-logit-distilled/dpo/checkpoints/final"
    )
    assert config.training.loss_type == ["sigmoid"]
    assert config.training.loss_weights == [1.0]


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/dpo.yaml",
        "configs/dpo_smoke.yaml",
        "configs/dpo_logit.yaml",
        "configs/dpo_logit_smoke.yaml",
    ],
)
def test_all_dpo_branches_use_sigmoid_baseline(config_path: str) -> None:
    config = load_dpo_config(config_path)

    assert config.training.loss_type == ["sigmoid"]
    assert config.training.loss_weights == [1.0]


def test_load_dpo_config_accepts_single_loss_for_compatibility(
    tmp_path: Path,
) -> None:
    import yaml

    raw = yaml.safe_load(Path("configs/dpo.yaml").read_text(encoding="utf-8"))
    raw["training"]["loss_type"] = "sigmoid"
    raw["training"].pop("loss_weights", None)
    path = tmp_path / "dpo.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_dpo_config(path)

    assert config.training.loss_type == ["sigmoid"]
    assert config.training.loss_weights == [1.0]


def test_load_dpo_config_rejects_mismatched_loss_weights(
    tmp_path: Path,
) -> None:
    import yaml

    raw = yaml.safe_load(Path("configs/dpo.yaml").read_text(encoding="utf-8"))
    raw["training"]["loss_weights"] = [1.0, 1.0]
    path = tmp_path / "dpo.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must match"):
        load_dpo_config(path)


def test_smoke_branches_are_isolated_and_preserve_handoffs() -> None:
    response = load_response_distill_config(
        "configs/response_distill_smoke.yaml"
    )
    response_dpo = load_dpo_config("configs/dpo_smoke.yaml")
    logit = load_logit_distill_config("configs/logit_distill_smoke.yaml")
    logit_dpo = load_dpo_config("configs/dpo_logit_smoke.yaml")

    assert response.output.final_checkpoint_dir.startswith("runs/smoke/")
    assert (
        response_dpo.source.checkpoint_path
        == response.output.final_checkpoint_dir
    )
    assert response_dpo.output.final_checkpoint_dir.startswith("runs/smoke/")
    assert logit.output.final_checkpoint_dir.startswith("runs/smoke/")
    assert logit_dpo.source.checkpoint_path == logit.output.final_checkpoint_dir
    assert logit_dpo.output.final_checkpoint_dir.startswith("runs/smoke/")


def test_load_dpo_config_rejects_wrong_method(tmp_path: Path) -> None:
    path = tmp_path / "dpo.yaml"
    content = (
        "source:\n"
        "  model_name: slm-test\n"
        "  checkpoint_path: checkpoint\n"
        "  tokenizer_path: tokenizer\n"
        "\n"
        "data:\n"
        "  dataset_id: example/preference\n"
        "  dataset_split: train\n"
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

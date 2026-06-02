from pathlib import Path

from distill.export.export_model import build_export_plan, export_model
from distill.export.model_card import build_model_card
from distill.utils.config import load_export_config


def test_load_export_config_reads_default_file() -> None:
    config = load_export_config("configs/export.yaml")

    assert config.model.model_name == "slm-125m-deepseek-distilled"
    assert config.model.export_repo == "tohio/slm-125m-deepseek-distilled"
    assert config.model_card.teacher_model == "deepseek/deepseek-v4-flash"
    assert config.model_card.teacher_provider == "openrouter"
    assert config.model_card.distillation_type == "response"
    assert config.model_card.dpo_applied is True
    assert config.export.push_to_hub is False


def test_build_export_plan() -> None:
    config = load_export_config("configs/export.yaml")
    plan = build_export_plan(config)

    assert plan.model_name == config.model.model_name
    assert plan.checkpoint_path == config.model.checkpoint_path
    assert plan.tokenizer_path == config.model.tokenizer_path
    assert plan.export_repo == config.model.export_repo
    assert plan.model_card_path == config.model_card.output_path


def test_build_model_card_contains_required_metadata() -> None:
    config = load_export_config("configs/export.yaml")
    card = build_model_card(config)

    assert card.model_name == "slm-125m-deepseek-distilled"
    assert "# slm-125m-deepseek-distilled" in card.content
    assert "deepseek/deepseek-v4-flash" in card.content
    assert "openrouter" in card.content
    assert "DPO-aligned" in card.content


def test_export_model_writes_model_card(tmp_path: Path) -> None:
    config_path = tmp_path / "export.yaml"
    model_card_path = tmp_path / "model_card.md"

    config_path.write_text(
        f'''
model:
  model_name: slm-test-teacher-distilled
  checkpoint_path: checkpoint
  tokenizer_path: tokenizer
  export_repo: tohio/slm-test-teacher-distilled

model_card:
  output_path: {model_card_path}
  source_checkpoint: tohio/slm-test-instruct
  teacher_model: example/model
  teacher_provider: openrouter
  distillation_type: response
  dpo_applied: true
  preference_dataset: data/preference/dpo_pairs.jsonl
  eval_results_path: runs/slm-test/eval/results.json

export:
  push_to_hub: false
  include_tokenizer: true
  private: false
''',
        encoding="utf-8",
    )

    plan = export_model(str(config_path))

    assert plan.model_name == "slm-test-teacher-distilled"
    assert model_card_path.exists()
    assert "example/model" in model_card_path.read_text(encoding="utf-8")

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from distill.export.model_card import write_model_card
from distill.utils.config import ExportConfig, load_export_config


@dataclass(frozen=True)
class ExportPlan:
    model_name: str
    checkpoint_path: str
    tokenizer_path: str
    export_repo: str
    model_card_path: str
    push_to_hub: bool
    include_tokenizer: bool
    private: bool


def build_export_plan(config: ExportConfig) -> ExportPlan:
    return ExportPlan(
        model_name=config.model.model_name,
        checkpoint_path=config.model.checkpoint_path,
        tokenizer_path=config.model.tokenizer_path,
        export_repo=config.model.export_repo,
        model_card_path=config.model_card.output_path,
        push_to_hub=config.export.push_to_hub,
        include_tokenizer=config.export.include_tokenizer,
        private=config.export.private,
    )


def load_export_plan(config_path: str) -> ExportPlan:
    return build_export_plan(load_export_config(config_path))


def export_model(config_path: str) -> ExportPlan:
    config = load_export_config(config_path)
    write_model_card(config)

    plan = build_export_plan(config)

    if plan.push_to_hub:
        raise NotImplementedError(
            "Hub export is configured but not implemented yet. "
            "Model card generation is implemented."
        )

    return plan


def validate_export_paths(plan: ExportPlan) -> None:
    checkpoint_path = Path(plan.checkpoint_path)
    tokenizer_path = Path(plan.tokenizer_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    if plan.include_tokenizer and not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

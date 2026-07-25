from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from distill.export.model_card import write_model_card
from distill.utils.config import ExportConfig, load_export_config
from distill.utils.env import require_env_value


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


def export_model(
    config_path: str,
    *,
    push_to_hub: bool | None = None,
    hub_api: object | None = None,
) -> ExportPlan:
    config = load_export_config(config_path)
    plan = build_export_plan(config)
    if push_to_hub is not None:
        plan = ExportPlan(
            model_name=plan.model_name,
            checkpoint_path=plan.checkpoint_path,
            tokenizer_path=plan.tokenizer_path,
            export_repo=plan.export_repo,
            model_card_path=plan.model_card_path,
            push_to_hub=push_to_hub,
            include_tokenizer=plan.include_tokenizer,
            private=plan.private,
        )

    validate_export_paths(plan)
    model_card_path = write_model_card(config)

    if plan.push_to_hub:
        if hub_api is None:
            from huggingface_hub import HfApi

            hub_api = HfApi(
                token=require_env_value(
                    "HF_TOKEN",
                    fallback_to_os=True,
                )
            )

        hub_api.create_repo(
            repo_id=plan.export_repo,
            repo_type="model",
            private=plan.private,
            exist_ok=True,
        )
        hub_api.upload_folder(
            repo_id=plan.export_repo,
            repo_type="model",
            folder_path=plan.checkpoint_path,
            commit_message=f"Upload {plan.model_name} checkpoint",
        )
        if (
            plan.include_tokenizer
            and Path(plan.tokenizer_path).resolve()
            != Path(plan.checkpoint_path).resolve()
        ):
            hub_api.upload_folder(
                repo_id=plan.export_repo,
                repo_type="model",
                folder_path=plan.tokenizer_path,
                commit_message=f"Upload {plan.model_name} tokenizer",
            )
        hub_api.upload_file(
            repo_id=plan.export_repo,
            repo_type="model",
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            commit_message=f"Update {plan.model_name} model card",
        )

    return plan


def validate_export_paths(plan: ExportPlan) -> None:
    checkpoint_path = Path(plan.checkpoint_path)
    tokenizer_path = Path(plan.tokenizer_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    if plan.include_tokenizer and not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

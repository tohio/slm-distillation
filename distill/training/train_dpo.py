
from __future__ import annotations

from dataclasses import dataclass

from distill.utils.config import DpoConfig, load_dpo_config


@dataclass(frozen=True)
class DpoTrainingPlan:
    source_checkpoint: str
    tokenizer_path: str
    dataset_id: str
    dataset_split: str
    output_dir: str
    final_checkpoint_dir: str
    beta: float
    max_length: int
    max_prompt_length: int


def build_dpo_training_plan(config: DpoConfig) -> DpoTrainingPlan:
    return DpoTrainingPlan(
        source_checkpoint=config.source.checkpoint_path,
        tokenizer_path=config.source.tokenizer_path,
        dataset_id=config.data.dataset_id,
        dataset_split=config.data.dataset_split,
        output_dir=config.output.checkpoint_dir,
        final_checkpoint_dir=config.output.final_checkpoint_dir,
        beta=config.training.beta,
        max_length=config.training.max_length,
        max_prompt_length=config.training.max_prompt_length,
    )


def load_dpo_training_plan(config_path: str) -> DpoTrainingPlan:
    return build_dpo_training_plan(load_dpo_config(config_path))


def train_dpo(config_path: str) -> None:
    load_dpo_training_plan(config_path)
    raise NotImplementedError(
        "DPO training is configured to consume the published preference dataset. "
        "The trainer implementation is not available yet."
    )

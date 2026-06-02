from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from distill.utils.config import ExportConfig


@dataclass(frozen=True)
class ModelCard:
    model_name: str
    content: str


def build_model_card(config: ExportConfig) -> ModelCard:
    content = f'''# {config.model.model_name}

DPO-aligned distilled SLM model variant.

## Model

| Field | Value |
|---|---|
| Model name | `{config.model.model_name}` |
| Export repo | `{config.model.export_repo}` |
| Checkpoint path | `{config.model.checkpoint_path}` |
| Tokenizer path | `{config.model.tokenizer_path}` |

## Distillation

| Field | Value |
|---|---|
| Source checkpoint | `{config.model_card.source_checkpoint}` |
| Teacher model | `{config.model_card.teacher_model}` |
| Teacher provider | `{config.model_card.teacher_provider}` |
| Distillation type | `{config.model_card.distillation_type}` |
| DPO applied | `{config.model_card.dpo_applied}` |
| Preference dataset | `{config.model_card.preference_dataset}` |

## Evaluation

Evaluation results path:

    {config.model_card.eval_results_path}

## Notes

Final exported models from this repository are DPO-aligned distilled variants. The distillation method is recorded in this model card rather than in the model name.
'''
    return ModelCard(model_name=config.model.model_name, content=content)


def write_model_card(config: ExportConfig) -> Path:
    card = build_model_card(config)
    output_path = Path(config.model_card.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card.content.rstrip() + "\n", encoding="utf-8")
    return output_path

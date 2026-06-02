# Prompt Seeds

Prompt/task seed files for teacher generation.

## Files

| File | Purpose |
|---|---|
| `instruction_seed.jsonl` | General instruction-following prompts. |
| `code_seed.jsonl` | Code-generation and function-completion prompts. |
| `factual_restraint_seed.jsonl` | Prompts for uncertainty, refusal, and hallucination control. |
| `arithmetic_seed.jsonl` | Arithmetic and concise numeric-answer prompts. |

## Record Format

Each JSONL record should use this shape:

    {
      "id": "instruction-000001",
      "category": "instruction",
      "prompt": "Explain what a Python virtual environment is in two sentences.",
      "metadata": {
        "source": "seed"
      }
    }

## Fields

| Field | Required | Description |
|---|---:|---|
| `id` | yes | Stable prompt identifier. Must be unique across all merged prompt files. |
| `category` | yes | Task category such as `instruction`, `code`, `arithmetic`, or `factual_restraint`. |
| `prompt` | yes | Prompt sent to the teacher model. |
| `metadata` | no | Extra task attributes used by validation or formatting. |

## Merging

Generation configs may provide multiple prompt files through `data.prompts_paths`.

Prompt files are loaded in config order. Duplicate prompt IDs are rejected.

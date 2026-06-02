# Prompt Seeds

Prompt/task seed files for teacher generation.

Each JSONL record should contain:

```json
{
  "id": "instruction-000001",
  "category": "instruction",
  "prompt": "Explain what a Python virtual environment is in two sentences.",
  "metadata": {
    "source": "seed"
  }
}
```

## Fields

| Field | Required | Description |
|---|---:|---|
| `id` | yes | Stable prompt identifier. |
| `category` | yes | Task category such as `instruction`, `code`, `arithmetic`, or `factual_restraint`. |
| `prompt` | yes | Prompt sent to the teacher model. |
| `metadata` | no | Extra task attributes used by validation or formatting. |

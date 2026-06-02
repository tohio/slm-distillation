# Preference Data

DPO preference datasets.

## Files

| File | Purpose |
|---|---|
| `dpo_pairs.jsonl` | Accepted DPO preference pairs. |
| `*.jsonl` | Additional generated preference datasets. |

Generated preference data is ignored by git. Keep `.gitkeep` committed.

## Record Format

Each record contains:

    {
      "prompt": "Return only the capital city of France.",
      "chosen": "Paris",
      "rejected": "The capital city of France is Paris.",
      "metadata": {
        "prompt_id": "instruction-000001",
        "category": "instruction"
      }
    }

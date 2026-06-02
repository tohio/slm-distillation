# Datasets

Dataset builders and formatters for student training.

## Files

| File | Purpose |
|---|---|
| `formatters.py` | Converts validated teacher records into training records. |
| `build_distill_dataset.py` | Builds the final response-distillation JSONL dataset. |

## Output Format

The response-distillation dataset uses:

```json
{
  "instruction": "...",
  "input": null,
  "output": "...",
  "metadata": {}
}
```

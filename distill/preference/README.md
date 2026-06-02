# Preference

Preference dataset builders for DPO.

## Files

| File | Purpose |
|---|---|
| `records.py` | Preference record schema and JSONL IO. |
| `build_preference_dataset.py` | Builds DPO pairs from validated teacher outputs and rejected responses. |

## Inputs

| Input | Purpose |
|---|---|
| Validated teacher responses | Chosen responses. |
| Rejected response file | Optional rejected responses keyed by prompt ID. |
| Default rejected response | Fallback rejected answer when no rejected file is configured. |

## Output

The builder writes accepted DPO pairs to `data/preference/` and rejected pair candidates to `data/rejected/`.

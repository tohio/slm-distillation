# Validation

Validation and filtering for raw teacher outputs.

## Files

| File | Purpose |
|---|---|
| `filters.py` | Validation rules and accept/reject splitting. |
| `schema.py` | JSONL writer for validated/rejected records. |
| `quality_checks.py` | Reserved for deeper task-specific checks. |

## Current Checks

- reject empty responses
- reject unexpected refusals
- reject code fences for function-body tasks

Validation should stay deterministic and cheap.

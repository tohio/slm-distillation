# Data

Local data layout for distillation runs.

Most files in this directory are generated artifacts and should not be committed, except seed prompts, README files, and `.gitkeep` placeholders.

## Layout

| Directory | Purpose |
|---|---|
| `prompts/` | Prompt/task seed files used for teacher generation. |
| `raw_teacher/` | Raw teacher outputs from hosted or local inference. |
| `validated/` | Accepted teacher outputs after validation/filtering. |
| `rejected/` | Rejected teacher outputs with validation reasons. |
| `distill/` | Final student-training JSONL datasets. |

## Artifact Policy

Generated JSONL files are ignored by git. Persist important run artifacts to external storage before terminating an instance.

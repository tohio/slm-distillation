# Scripts

Command-line entrypoints for the distillation pipeline.

## Commands

| Script | Purpose |
|---|---|
| `generate_teacher_responses.py` | Generate raw teacher outputs from prompts. |
| `validate_teacher_responses.py` | Validate raw teacher outputs. |
| `build_dataset.py` | Build response-distillation training JSONL. |
| `train_response_distill.py` | Train a student with response distillation. |
| `train_logit_distill.py` | Train a student with logit distillation. |
| `run_eval.py` | Evaluate base and distilled checkpoints. |

Prefer Makefile targets for normal use.

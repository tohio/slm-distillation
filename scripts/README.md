# Scripts

Command-line entrypoints for the distillation pipeline.

## Commands

| Script | Purpose |
|---|---|
| `generate_teacher_responses.py` | Generate raw teacher outputs from prompts. |
| `validate_teacher_responses.py` | Validate raw teacher outputs. |
| `build_dataset.py` | Build response-distillation training JSONL. |
| `build_preference_dataset.py` | Build DPO preference-pair JSONL. |
| `train_response_distill.py` | Train a student with response distillation. |
| `train_logit_distill.py` | Run or inspect the local logit-distillation stage. |
| `train_dpo.py` | Run or inspect the DPO training stage. |
| `export_model.py` | Generate model card and export final model variant. |
| `run_eval.py` | Evaluate base and distilled checkpoints. |

# Scripts

Command-line entrypoints for the distillation pipeline.

## Commands

| Script | Purpose |
|---|---|
| `build_prompts.py` | Build scalable prompt JSONL for teacher generation. |
| `generate_teacher_responses.py` | Generate raw teacher outputs from prompts. |
| `validate_teacher_responses.py` | Validate raw teacher outputs. |
| `build_dataset.py` | Build response-distillation training JSONL. |
| `build_preference_dataset.py` | Build DPO preference-pair JSONL. |
| `pack_artifacts.py` | Pack generated JSONL artifacts into a local bundle. |
| `verify_artifacts.py` | Verify artifact manifest checksums. |
| `unpack_artifacts.py` | Unpack a local artifact bundle. |
| `push_artifacts.py` | Push generated JSONL artifacts to S3. |
| `pull_artifacts.py` | Pull generated JSONL artifacts from S3. |
| `train_response_distill.py` | Train a student with response distillation. |
| `train_logit_distill.py` | Run or inspect the local logit-distillation stage. |
| `train_dpo.py` | Run or inspect the DPO training stage. |
| `export_model.py` | Generate model card and export final model variant. |
| `run_eval.py` | Evaluate base and distilled checkpoints. |

`report_token_counts.py` reports estimated token counts for generated JSONL datasets.

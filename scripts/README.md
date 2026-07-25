# Scripts

Command-line entry points for training, evaluation, export, and artifacts.

| Script | Purpose |
|---|---|
| `train_response_distill.py` | Train a student from the published response dataset |
| `train_dpo.py` | Train or inspect the DPO stage |
| `train_logit_distill.py` | Train or inspect local logit distillation |
| `run_eval.py` | Evaluate base and distilled checkpoints |
| `export_model.py` | Generate the model card and export the final checkpoint |
| `pack_artifacts.py` | Package model run artifacts |
| `verify_artifacts.py` | Verify artifact checksums |
| `unpack_artifacts.py` | Unpack a model artifact bundle |
| `push_artifacts.py` | Push model artifacts to S3 |
| `pull_artifacts.py` | Pull model artifacts from S3 |

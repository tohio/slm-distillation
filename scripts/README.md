# Scripts

Command-line entry points for training, evaluation, export, and artifacts.

| Script | Purpose |
|---|---|
| `train_response_distill.py` | Inspect response config, validate inputs, or start response training |
| `train_dpo.py` | Train or inspect the DPO stage |
| `train_logit_distill.py` | Train or inspect local logit distillation |
| `run_eval.py` | Evaluate base and distilled checkpoints |
| `export_model.py` | Generate the model card and export the final checkpoint |
| `pack_artifacts.py` | Package model run artifacts |
| `verify_artifacts.py` | Verify artifact checksums |
| `unpack_artifacts.py` | Unpack a model artifact bundle |
| `push_artifacts.py` | Push model artifacts to S3 |
| `pull_artifacts.py` | Pull model artifacts from S3 |

Response and DPO input validation resolve their configured checkpoints and
inspect selected dataset rows without starting training. Both trainers support
bounded sample/step overrides and checkpoint resume.

Logit validation additionally resolves both models and compares their complete
tokenizer vocabularies and special-token maps. Evaluation validates both
response and preference datasets. Export uses `--push-to-hub` for an explicit
Hugging Face upload.

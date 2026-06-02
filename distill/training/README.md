# Training

Student distillation training code.

## Files

| File | Purpose |
|---|---|
| `train_response_distill.py` | Response-distillation trainer. |
| `train_logit_distill.py` | Logit-distillation trainer. |

## Modes

Response distillation trains on teacher-generated final answers.

Logit distillation trains the student to match a local teacher checkpoint's token-level output distribution. Teacher and student tokenizers should be compatible.

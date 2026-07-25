# Training

Student post-training stages.

| File | Input | Output | Status |
|---|---|---|---|
| `train_response_distill.py` | Published response dataset | Response-distilled checkpoint | Not implemented |
| `train_dpo.py` | Published preference dataset | DPO-aligned checkpoint | Config plan only |
| `train_logit_distill.py` | Local teacher and student | Logit-distilled checkpoint | Compatibility plan only |

Response and DPO datasets are loaded from Hugging Face. Logit distillation
requires compatible local teacher and student tokenizers.

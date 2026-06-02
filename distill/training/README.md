
# Training

Student post-training stages.

## Files

| File | Purpose |
|---|---|
| `train_response_distill.py` | Response-distillation trainer. |
| `train_logit_distill.py` | Logit-distillation trainer. |
| `train_dpo.py` | DPO training stage scaffold. |

## Stages

| Stage | Input | Output |
|---|---|---|
| Response distillation | Validated teacher responses | Intermediate distilled checkpoint |
| Logit distillation | Local teacher logits | Intermediate distilled checkpoint |
| DPO | Preference pairs | Final DPO-aligned distilled checkpoint |

Final exported models use the post-DPO checkpoint path defined in `configs/dpo.yaml`.

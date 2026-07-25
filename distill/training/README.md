# Training

Student post-training stages.

| File | Input | Output | Status |
|---|---|---|---|
| `train_response_distill.py` | Published response dataset | Response-distilled checkpoint | Implemented |
| `train_dpo.py` | Published preference dataset | DPO-aligned checkpoint | Implemented |
| `train_logit_distill.py` | Local teacher and student | Logit-distilled checkpoint | Implemented |

Response and DPO datasets are loaded from Hugging Face. Logit distillation
uses response records for hard-label loss while matching the teacher
distribution on supervised response tokens.

Response training applies causal language-model loss only to teacher-response
tokens. Prompt tokens and padding tokens use the `-100` ignore index.

DPO training uses TRL's DPO trainer with the response-distilled checkpoint as
the initial policy and reference policy. It accepts standard or conversational
preference datasets.

Logit training freezes the local teacher, trains the full student, combines
causal cross-entropy and temperature-scaled KL divergence, and saves the final
model and tokenizer. It requires matching tokenizers and exactly one supported
GPU.

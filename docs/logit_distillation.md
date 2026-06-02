# Logit Distillation

Local logit distillation trains a student model against teacher token distributions.

## Requirements

| Requirement | Description |
|---|---|
| Local teacher | Teacher model must run locally. |
| Full logits | Training requires teacher logits from the forward pass. |
| Tokenizer compatibility | Teacher and student must use the same tokenizer and vocabulary. |
| Single GPU | Teacher model must fit on one supported GPU. |

## Supported GPU Classes

    B300
    B200
    H200
    A100

## Tokenizer Gate

The logit-distillation stage checks tokenizer compatibility before training.

The gate verifies:

- vocabulary equality
- token ID equality
- special token equality

Response distillation does not require tokenizer compatibility because teacher text is retokenized for the student.

# Logit Distillation

Local logit distillation trains a student model against teacher token
distributions. The default pairing is SmolLM2-1.7B-Instruct as the frozen
teacher and SmolLM2-135M-Instruct as the trainable student.

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

## Objective

The trainer computes loss only on response tokens:

```text
loss = alpha * hard_cross_entropy
     + (1 - alpha) * temperature² * teacher_student_KL
```

`alpha: 1` is ordinary supervised fine-tuning; `alpha: 0` is pure logit
matching. The default is `0.5`.

## Execution

```bash
make validate-logit-inputs LOGIT_DATA_LIMIT=100
make train-logit-smoke
make train-logit
```

The smoke configuration writes under `runs/smoke/`; the full configuration
writes under `runs/`. The trainer fails early unless CUDA exposes exactly one
supported GPU.

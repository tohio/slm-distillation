# Logit Distillation

Objective, tokenizer compatibility gate, and hardware contract for local
teacher-to-student distillation.

## Default Pairing

| Role | Model |
|---|---|
| Frozen teacher | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Trainable student | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Training data | `tohio/slm-synthetic-distillation-sft` |

The teacher is loaded locally for forward passes. No hosted inference provider
or OpenRouter dependency is used by this branch.

## Objective

Loss is computed only on supervised response tokens:

~~~text
loss = alpha * hard_cross_entropy
     + (1 - alpha) * temperature² * teacher_student_KL
~~~

`alpha: 1` is ordinary supervised fine-tuning, `alpha: 0` is pure logit
matching, and the default is `0.5`. The default temperature is `2.0`.

## Tokenizer Gate

Teacher and student must have:

- identical vocabularies;
- identical token IDs;
- identical special-token mappings.

Logit comparison is token-aligned, so compatibility is a hard requirement.
Response distillation does not share this restriction because saved teacher
text is tokenized only by the student.

## Hardware Contract

Logit training requires exactly one visible CUDA GPU from the configured
classes:

- B300
- B200
- H200
- A100

Teacher and student are colocated on that device. The teacher is frozen and
loaded in bfloat16; gradient checkpointing applies to the student.

## Validation and Execution

~~~bash
make validate-logit-inputs LOGIT_DATA_LIMIT=100
export CUDA_VISIBLE_DEVICES=0
make train-logit-smoke
make train-logit
~~~

The validator resolves both model references, compares tokenizers, and
inspects the configured response dataset. It downloads model metadata and
tokenizer files, not teacher/student weight files, and it does not allocate a
GPU or run forward passes. The smoke command is the first check that downloads
weights and executes the teacher/student loss on CUDA. Smoke outputs go to
`runs/smoke/`; full outputs go to `runs/`.

## Swapping the Teacher or Student

Update all teacher/student model, tokenizer, and revision fields in the
matching logit config. Run the input validator before any smoke or full run.
If tokenizer compatibility fails, use response distillation or choose a
tokenizer-compatible pair.

## See Also

- [Architecture](architecture.md)
- [Training](training.md)
- [Configuration](configuration.md)
- [Evaluation and Export](evaluation-and-export.md)

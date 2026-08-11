# Response Distillation

Offline teacher-response training, student-only execution, and checkpoint
handoff for the response branch.

## Default Inputs

| Role | Default |
|---|---|
| Student | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Response dataset | `tohio/slm-synthetic-distillation-sft` |
| Preference dataset | `tohio/slm-synthetic-distillation-dpo` |

The response dataset already contains teacher-produced text. Training does not
load or call the teacher model, so changing the published dataset does not
introduce a hosted-provider dependency.

## Data Contract

Configured rows provide `id`, `prompt`, `response`, and `metadata`. Validation
checks required columns, non-empty prompt/response text, and unique IDs in the
inspected sample. Chat formatting uses the student tokenizer's chat template.

## Objective and Masking

Response distillation is supervised causal-language-model training. Prompt and
formatting tokens receive label `-100`; only response tokens contribute to the
loss. Padding tokens are masked by the collator. This teaches the saved teacher
response without training the student to reproduce the prompt.

## Execution

~~~bash
make validate-response-inputs RESPONSE_DATA_LIMIT=100
make train-response-smoke
make train-response
~~~

Smoke output is isolated under `runs/smoke/`. Production output is written to:

~~~text
runs/smollm2-135m-response-distilled/response_distill/checkpoints/final
~~~

The response DPO config consumes that model and tokenizer path. It combines
sigmoid preference loss with chosen-response SFT loss so preference learning
does not reduce the likelihood of both responses unchecked.

## Swapping Models or Data

Update the model/tokenizer reference, dataset ID and field mappings, output
identity, and every downstream checkpoint reference together. Update export
teacher and dataset provenance when the replacement dataset was produced by a
different teacher. Dataset-provider provenance remains with the published
dataset.

## Troubleshooting

- Run input validation before training to catch schema and model-resolution
  failures without allocating a GPU.
- Install `python3.12-dev` and `build-essential` when runtime compilation cannot
  find `Python.h` or a compiler.
- Use bounded `RESPONSE_MAX_STEPS` and `RESPONSE_MAX_TRAIN_SAMPLES` overrides
  for production-config probes; smoke paths deliberately use separate inputs
  and outputs.
- Resume only from a checkpoint belonging to the same branch and configuration.

## See Also

- [Training](training.md)
- [Configuration](configuration.md)
- [Evaluation and Export](evaluation-and-export.md)
- [Logit Distillation](logit_distillation.md)

# SLM Distillation

Teacher-student distillation workflows for the SLM project.

This repository focuses on using larger teacher models to improve smaller SLM models through response distillation, instruction distillation, and evaluation-driven refinement.

## Goals

* Generate teacher responses for selected prompts and tasks.
* Build distillation datasets for small language models.
* Fine-tune existing SLM student models using teacher-generated outputs.
* Compare teacher, base student, and distilled student behavior.
* Track quality, cost, and throughput of distillation runs.

## Scope

This repo is intended for distillation experiments and pipelines. It does not replace the core SLM training repo or the synthetic data generation repo.

The student models used by this repo come from the core `slm` repository. This repo does not define or train student model architectures from scratch. Instead, it consumes exported/base/chat/code checkpoints from `slm`, applies distillation fine-tuning, evaluates the results, and optionally exports distilled variants.

Related repos:

* `slm` — core from-scratch SLM training pipeline and source of student models.
* `slm-synthetic-data` — synthetic data generation pipeline.
* `slm-reasoning` — reasoning-model training and evaluation workflows.

## Planned Workflow

```text
slm student checkpoint
  + prompts/tasks
  -> teacher generation
  -> validation/filtering
  -> distillation dataset
  -> student distillation fine-tuning
  -> evaluation
  -> export distilled checkpoint
```

## Student Models

Student models are imported from the `slm` repo outputs.

Examples may include:

* base SLM checkpoints
* SFT/chat SLM checkpoints
* SFT/code SLM checkpoints
* future instruction-tuned SLM checkpoints

This repository should treat those checkpoints as inputs. Any architecture changes, tokenizer changes, base pretraining, or core SFT/DPO training should remain in the `slm` repository.

## Related Projects

* [slm](https://github.com/tohio/slm) — GPT-style LLM trained from scratch

## License

MIT

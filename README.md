# SLM Distillation

Teacher-student distillation workflows for the SLM project.

This repository focuses on using larger teacher models to improve smaller SLM models through response distillation, instruction distillation, and evaluation-driven refinement.

## Goals

- Generate teacher responses for selected prompts and tasks.
- Build distillation datasets for small language models.
- Fine-tune student models on teacher-generated outputs.
- Compare teacher, base student, and distilled student behavior.
- Track quality, cost, and throughput of distillation runs.

## Scope

This repo is intended for distillation experiments and pipelines. It does not replace the core SLM training repo or the synthetic data generation repo.

Related repos:

- `slm` — core from-scratch SLM training pipeline.
- `slm-synthetic-data` — synthetic data generation pipeline.
- `slm-reasoning` — reasoning-model training and evaluation workflows.

## Planned Workflow

```text
prompts/tasks
  -> teacher generation
  -> validation/filtering
  -> distillation dataset
  -> student fine-tuning
  -> evaluation
  -> export
```

---

## Related Projects

- [slm](https://github.com/tohio/slm) — GPT-style LLM trained from scratch

---

## License

MIT
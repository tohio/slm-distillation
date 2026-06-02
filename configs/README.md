
# Configs

Configuration files for teacher selection, distillation runs, validation rules, and evaluation.

## Files

| File | Purpose |
|---|---|
| `teachers.yaml` | Teacher model registry with provider, model slug, mode, and pricing metadata. |
| `response_distill.yaml` | OpenRouter response-distillation run config. |
| `response_distill_groq.yaml` | Groq response-distillation run config. |
| `logit_distill.yaml` | Local teacher/student logit-distillation config. |
| `dpo.yaml` | DPO stage config for the final distilled model. |
| `validation.yaml` | Shared validation and filtering rules. |
| `eval.yaml` | Evaluation config for base and distilled checkpoints. |

## Output Naming

Final exported model names use the source model and teacher family:

    slm-125m-deepseek-distilled
    slm-125m-groq-distilled
    slm-125m-qwen-distilled

Distillation method and DPO status are recorded in model card metadata, not in the model name.

## Output Fields

| Field | Purpose |
|---|---|
| `output.model_name` | Final exported model name. |
| `output.source_model_name` | Source SLM model family/name. |
| `output.teacher_family` | Teacher family used in the final model name. |
| `output.run_dir` | Run artifact directory. |
| `output.checkpoint_dir` | Intermediate distillation checkpoint directory. |
| `output.final_checkpoint_dir` | Final post-DPO checkpoint directory. |
| `output.export_repo` | Hugging Face export repository. |

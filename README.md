# SLM Distillation

Teacher-student distillation workflows for SLM checkpoints — prompts through to a validated, fine-tuned, evaluation-ready distilled model. Covers the full lifecycle: teacher selection, prompt/task preparation, teacher generation, validation, dataset construction, response distillation, optional logit distillation, evaluation, cost tracking, and export.

> **Status:** This project is under active development. Response distillation is the primary workflow. Logit distillation is supported for local teacher/student checkpoints with compatible tokenizers and vocabularies.

---

## Overview

SLM Distillation improves SLM checkpoints using permitted teacher models. The repository consumes student checkpoints produced by [`slm`](https://github.com/tohio/slm), generates or loads teacher supervision, validates the outputs, builds distillation datasets, fine-tunes the student, evaluates behavior changes, and exports distilled checkpoints.

The pipeline is modular and independently runnable at each stage. Teacher models, student checkpoints, providers, validation rules, training settings, and output paths are all configuration-driven.

**Default teacher:** `deepseek/deepseek-v4-flash`

```text
prompts/tasks
  -> teacher generation
  -> validation/filtering
  -> distillation dataset
  -> student distillation training
  -> evaluation
  -> export
```
---

## Tech Stack

| Stage               | Tool                                            |
| ------------------- | ----------------------------------------------- |
| Config              | YAML + Python dataclasses                       |
| Teacher inference   | OpenRouter, Groq, DeepSeek API, local backends  |
| Dataset format      | JSONL                                           |
| Validation          | Custom filters + schema checks                  |
| Student fine-tuning | HuggingFace `transformers` / `trl`              |
| Logit distillation  | PyTorch                                         |
| Evaluation          | `lm-evaluation-harness` + custom behavior evals |
| Cost tracking       | Provider pricing config + run metrics           |
| Experiment tracking | Weights & Biases                                |
| Export              | HuggingFace `transformers`                      |

---

## Repo Structure

```text
slm-distillation/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── .env.sample
│
├── configs/
│   ├── README.md
│   ├── teachers.yaml
│   ├── response_distill_openrouter.yaml
│   ├── logit_distill.yaml
│   ├── validation.yaml
│   └── eval.yaml
│
├── data/
│   ├── README.md
│   ├── prompts/
│   │   ├── README.md
│   │   ├── instruction_seed.jsonl
│   │   ├── code_seed.jsonl
│   │   ├── factual_restraint_seed.jsonl
│   │   └── arithmetic_seed.jsonl
│   ├── raw_teacher/
│   │   ├── README.md
│   │   └── .gitkeep
│   ├── validated/
│   │   ├── README.md
│   │   └── .gitkeep
│   ├── rejected/
│   │   ├── README.md
│   │   └── .gitkeep
│   └── distill/
│       ├── README.md
│       └── .gitkeep
│
├── distill/
│   ├── README.md
│   ├── __init__.py
│   ├── providers/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── openrouter.py
│   │   ├── groq.py
│   │   ├── deepseek.py
│   │   └── local.py
│   ├── generation/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── generate_responses.py
│   │   └── batch_runner.py
│   ├── validation/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── filters.py
│   │   ├── schema.py
│   │   └── quality_checks.py
│   ├── datasets/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── build_distill_dataset.py
│   │   └── formatters.py
│   ├── training/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── train_response_distill.py
│   │   └── train_logit_distill.py
│   ├── eval/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── run_eval.py
│   │   └── compare_outputs.py
│   └── utils/
│       ├── README.md
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       ├── pricing.py
│       └── logging.py
│
├── scripts/
│   ├── README.md
│   ├── generate_teacher_responses.py
│   ├── validate_teacher_responses.py
│   ├── build_dataset.py
│   ├── train_response_distill.py
│   ├── train_logit_distill.py
│   └── run_eval.py
│
├── tests/
│   ├── README.md
│   ├── test_config.py
│   ├── test_validation.py
│   ├── test_formatters.py
│   └── test_pricing.py
│
└── docs/
    ├── README.md
    ├── teacher_policy.md
    ├── response_distillation.md
    ├── logit_distillation.md
    └── cost_tracking.md
```

---

## Getting Started

**Prerequisites**

* Python 3.12+
* SLM checkpoint from the `slm` repo
* API key for hosted teacher inference, if using a hosted provider
* CUDA-capable GPU for student fine-tuning
* HuggingFace token, if loading gated models or pushing exports
* Weights & Biases account, optional

**Installation**

```bash
git clone https://github.com/tohio/slm-distillation.git
cd slm-distillation

cp .env.sample .env
vi .env

make install
```

Example `.env`:

```bash
OPENROUTER_API_KEY=
GROQ_API_KEY=
DEEPSEEK_API_KEY=
HF_TOKEN=
WANDB_API_KEY=
```

---

## Run the Pipeline

### Step 1: Prepare prompts

```bash
python scripts/prepare_prompts.py \
  --config configs/response_distill_openrouter.yaml
```

### Step 2: Generate teacher responses

```bash
python scripts/generate_teacher_responses.py \
  --config configs/response_distill_openrouter.yaml
```

### Step 3: Validate teacher responses

```bash
python scripts/validate_teacher_responses.py \
  --config configs/validation.yaml \
  --input data/raw_teacher/deepseek_v4_flash.jsonl \
  --output data/validated/deepseek_v4_flash.jsonl \
  --rejected data/rejected/deepseek_v4_flash.jsonl
```

### Step 4: Build the distillation dataset

```bash
python scripts/build_dataset.py \
  --input data/validated/deepseek_v4_flash.jsonl \
  --output data/distill/response_distill.jsonl
```

### Step 5: Train with response distillation

```bash
python scripts/train_response_distill.py \
  --config configs/response_distill_openrouter.yaml
```

### Step 6: Evaluate the distilled checkpoint

```bash
python scripts/run_eval.py \
  --config configs/eval.yaml
```

### Step 7: Export the distilled checkpoint

```bash
python scripts/export.py \
  --config configs/export.yaml
```

---

## Configuration

### Response Distillation

```yaml
teacher:
  name: deepseek_v4_flash

student:
  name: slm-student
  checkpoint_path: ../slm/results/runs/student/final
  tokenizer_path: ../slm/results/runs/student/tokenizer

distillation:
  mode: response
  target_tokens: 50000000
  max_output_tokens: 1024
  temperature: 0.2
  top_p: 0.9

data:
  prompts_path: data/prompts/instruction_seed.jsonl
  raw_teacher_path: data/raw_teacher/deepseek_v4_flash.jsonl
  validated_path: data/validated/deepseek_v4_flash.jsonl
  rejected_path: data/rejected/deepseek_v4_flash.jsonl
  distill_dataset_path: data/distill/response_distill.jsonl

validation:
  require_non_empty_output: true
  reject_refusals_when_not_expected: true
  reject_code_fences_for_function_body_tasks: true
  max_retries: 2

output:
  run_dir: runs/response_distill
  checkpoint_dir: runs/response_distill/checkpoints
```

### Logit Distillation

```yaml
teacher:
  mode: logit
  checkpoint_path: ../slm/results/runs/teacher/final
  tokenizer_path: ../slm/results/runs/teacher/tokenizer

student:
  checkpoint_path: ../slm/results/runs/student/final
  tokenizer_path: ../slm/results/runs/student/tokenizer

distillation:
  mode: logit
  temperature: 2.0
  alpha: 0.5
  top_k_logits: null

data:
  train_path: data/distill/response_distill.jsonl

output:
  run_dir: runs/logit_distill
  checkpoint_dir: runs/logit_distill/checkpoints
```

---

## Cost Tracking

Teacher selection should use cost per accepted sample, not only raw token price.

The pipeline records:

* provider
* teacher model
* input tokens
* output tokens
* total cost
* accepted samples
* rejected samples
* retry count
* validation pass rate
* cost per accepted sample
* tokens per second, when available

Example:

```json
{
  "teacher": "deepseek/deepseek-v4-flash",
  "provider": "openrouter",
  "input_tokens": 1200000,
  "output_tokens": 2800000,
  "accepted_samples": 3812,
  "rejected_samples": 188,
  "validation_pass_rate": 0.953,
  "total_cost_usd": 0.72,
  "cost_per_accepted_sample_usd": 0.000189
}
```

---

## Outputs

Runs are written under `runs/`.

```text
runs/
└── response_distill/
    ├── config.yaml
    ├── metrics.json
    ├── cost.json
    ├── eval.json
    └── checkpoints/
```

Distillation datasets are written under `data/distill/`.

```text
data/distill/
└── response_distill.jsonl
```

---

## Tests

Unit tests:

```bash
make test-unit
```

Validation tests:

```bash
make test-validation
```

Dataset formatting tests:

```bash
make test-formatters
```

Pricing and cost-accounting tests:

```bash
make test-pricing
```

Full test suite:

```bash
make test
```

| Target            | Validates                                                    |
| ----------------- | ------------------------------------------------------------ |
| `test-config`     | config loading, required fields, teacher/student references  |
| `test-validation` | schema checks, rejection rules, retry-safe validation        |
| `test-formatters` | prompt/response conversion into training records             |
| `test-pricing`    | token accounting, provider pricing, cost per accepted sample |
| `test-unit`       | all unit tests                                               |
| `test`            | full test suite                                              |

---

## Evaluation

Distilled checkpoints should be compared against their source checkpoints.

Evaluation should include:

* benchmark scores
* behavior sanity prompts
* code-generation checks
* arithmetic checks
* factual-restraint checks
* answer-format compliance
* output-length control
* cost per accepted sample

Example:

```bash
python scripts/run_eval.py \
  --config configs/eval.yaml \
  --base-checkpoint ../slm/results/runs/student/final \
  --distilled-checkpoint runs/response_distill/checkpoints/final
```

---

## Distillation Pipeline

The repository follows a staged lifecycle.

| Stage                   | Purpose                                                             | Output                           |
| ----------------------- | ------------------------------------------------------------------- | -------------------------------- |
| Prompt/task preparation | Select prompts and task records for teacher generation              | prompt JSONL                     |
| Teacher generation      | Generate teacher responses through hosted or local inference        | raw teacher JSONL                |
| Validation/filtering    | Reject malformed, low-quality, unsafe, or task-mismatched outputs   | validated JSONL + rejected JSONL |
| Dataset construction    | Convert accepted teacher outputs into student training records      | distillation dataset             |
| Response distillation   | Fine-tune the student on teacher-generated responses                | response-distilled checkpoint    |
| Logit distillation      | Train student against local teacher logits when compatible          | logit-distilled checkpoint       |
| Evaluation              | Compare base and distilled checkpoints                              | eval reports                     |
| Cost tracking           | Track provider cost, retries, acceptance rate, and throughput       | cost report                      |
| Export                  | Package distilled checkpoint for downstream inference or Hub upload | exported model artifact          |

---

## Distillation Modes

### Response Distillation

Response distillation trains a student checkpoint on teacher-generated completions.

```text
prompt/task
  -> teacher response
  -> validation/filtering
  -> distillation dataset
  -> student fine-tuning
```

This mode supports hosted teacher inference through providers such as OpenRouter, Groq, DeepSeek API, or local generation backends.

### Logit Distillation

Logit distillation trains a student checkpoint to match a teacher checkpoint's token-level output distribution.

```text
teacher checkpoint + student checkpoint
  -> teacher logits
  -> student logits
  -> KL / distillation loss
  -> distilled student checkpoint
```

This mode requires local access to teacher logits. Teacher and student checkpoints should use compatible tokenizers and vocabularies.

---

## Teacher Model Policy

Teacher models must be permitted for distillation use.

Before using a model for training-target generation, verify:

* model license
* model-owner terms
* hosted provider terms, if using hosted inference
* whether generated outputs may be used to train or improve another model

Closed commercial API models should not be used as distillation teachers unless their terms explicitly allow output-based training.

The serving provider does not determine distillation permission. Permission depends on both the underlying model terms and the provider terms.

---

## Teacher Candidates

Teacher models are configured, not hardcoded.

```yaml
default_teacher: deepseek_v4_flash

teachers:
  deepseek_v4_flash:
    provider: openrouter
    model: deepseek/deepseek-v4-flash
    mode: response
    purpose: bulk_response_distillation
    distillation_allowed: true

  qwen3_235b_a22b_2507:
    provider: openrouter
    model: qwen/qwen3-235b-a22b-2507
    mode: response
    purpose: quality_comparison
    distillation_allowed: verify

  qwen3_30b_a3b_2507:
    provider: openrouter
    model: qwen/qwen3-30b-a3b-instruct-2507
    mode: response
    purpose: low_cost_comparison
    distillation_allowed: verify

  gpt_oss_20b:
    provider: openrouter
    model: openai/gpt-oss-20b
    mode: response
    purpose: smoke_test
    distillation_allowed: policy_gated
```

---

## Student Checkpoints

Student checkpoints are supplied by configuration.

```yaml
student:
  name: slm-student
  checkpoint_path: ../slm/results/runs/student/final
  tokenizer_path: ../slm/results/runs/student/tokenizer
```

The pipeline should work with any compatible SLM checkpoint. Model size is a configuration detail, not a separate code path.

---

## Key Design Decisions

**Why response distillation?**
Response distillation works with hosted inference providers and does not require local access to teacher logits. It is the simplest and cheapest way to generate high-quality training targets from permitted teacher models.

**Why logit distillation?**
Logit distillation transfers richer teacher information than final-answer targets. It trains the student to match the teacher's token-level output distribution. This requires local teacher inference and compatible tokenization.

**Why hosted teacher inference?**
Hosted inference avoids renting GPU hardware only to generate teacher responses. The pipeline pays per token and records the cost per accepted sample.

**Why provider-agnostic teachers?**
Teacher quality, cost, and availability change over time. Keeping teacher selection in config allows the same pipeline to run against OpenRouter, Groq, DeepSeek API, or local models.

**Why validate teacher outputs?**
Teacher outputs can be malformed, too verbose, incorrect, or poorly formatted for the target task. Validation prevents low-quality completions from becoming training targets.

**Why cost per accepted sample?**
Raw token price is incomplete. A cheap model with many rejected outputs can be more expensive than a higher-quality model with a better acceptance rate.

**Why keep student checkpoints external?**
The `slm` repo owns model architecture, tokenizer training, pretraining, SFT, DPO, and export. This repo consumes those checkpoints and produces distilled variants.

---

## Related Projects

* [slm](https://github.com/tohio/slm) — core SLM training pipeline and source of student checkpoints
* [slm-synthetic-data](https://github.com/tohio/slm-synthetic-data) — synthetic data generation pipeline
* [slm-reasoning](https://github.com/tohio/slm-reasoning) — reasoning-specific training and evaluation workflows

---

## License

MIT

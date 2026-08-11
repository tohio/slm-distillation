# Training

End-to-end response and logit branch workflows, including isolated smoke runs
and production handoffs.

## Prerequisites

~~~bash
sudo apt-get update
sudo apt-get install -y python3.12-venv python3.12-dev build-essential

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cp .env.sample .env
make install
make test
~~~

Add your Hugging Face token to `.env` before accessing Hub resources. The
`make install` target installs all packages declared in `requirements.txt`
inside the active virtual environment.

`HF_TOKEN` is optional for public inputs but recommended for Hub rate limits;
it is required for private inputs and model uploads. Logit training requires
exactly one visible supported CUDA GPU.

`python3.12-dev` supplies `Python.h`, while `build-essential` supplies the
compiler toolchain used by runtime-compiled GPU kernels.

## Experiment Tracking

W&B tracking is optional. Set these values in `.env` to enable online runs:

~~~dotenv
WANDB_API_KEY=...
WANDB_PROJECT=slm-distillation
WANDB_ENTITY=...
~~~

Leave `WANDB_API_KEY` empty to keep W&B disabled. For a local offline run, set
`WANDB_MODE=offline` in `.env` or the process environment. Every trainer still
writes local Trainer state and metrics when W&B is disabled. W&B runs are
grouped by model identity and labeled by response, logit, or DPO stage.

## Training Output

Console logs use compact, stage-specific lines instead of the full Trainer
metric dictionary. DPO output emphasizes chosen/rejected rewards, margin,
accuracy, and entropy. Logit output separates hard-label and soft-KL loss.
Complete metrics remain available in Trainer state, local metric files, and
W&B when enabled.

## Response Branch

The response branch performs supervised fine-tuning on published teacher
responses, then aligns that checkpoint with DPO.

DPO combines the sigmoid preference loss with an SFT loss on chosen responses:

~~~yaml
loss_type: [sigmoid, sft]
loss_weights: [1.0, 1.0]
~~~

The chosen-response term prevents preference separation from being achieved
only by reducing rejected-response likelihood. During a bounded stability run,
watch chosen reward, rejected reward, reward margin, preference accuracy, and
entropy before committing to the full epoch.

### Smoke

~~~bash
make validate-response-inputs RESPONSE_DATA_LIMIT=100
make train-response-smoke
make validate-dpo-inputs DPO_CONFIG=configs/dpo_smoke.yaml DPO_DATA_LIMIT=100
make train-dpo-smoke
make validate-eval-response-inputs EVAL_CONFIG=configs/eval_smoke.yaml EVAL_LIMIT=20
make eval-response-smoke
~~~

The smoke chain is:

~~~text
configs/response_distill_smoke.yaml
  -> response smoke final
  -> configs/dpo_smoke.yaml
  -> DPO smoke final
  -> configs/eval_smoke.yaml
~~~

### Production

~~~bash
make validate-response-inputs RESPONSE_DATA_LIMIT=100
make train-response
make validate-dpo-inputs DPO_DATA_LIMIT=100
make train-dpo
make validate-eval-response-inputs EVAL_LIMIT=200
make eval-response
make export
~~~

Use `make export-push` only when the configured final repository should be
created or updated on Hugging Face.

## Logit Branch

The logit branch freezes SmolLM2-1.7B-Instruct as the local teacher and trains
SmolLM2-135M-Instruct as the student.

### Smoke

~~~bash
make validate-logit-inputs LOGIT_DATA_LIMIT=100
CUDA_VISIBLE_DEVICES=0 make train-logit-smoke
make validate-dpo-logit-inputs DPO_LOGIT_CONFIG=configs/dpo_logit_smoke.yaml DPO_DATA_LIMIT=100
make train-dpo-logit-smoke
make validate-eval-logit-inputs EVAL_LOGIT_CONFIG=configs/eval_logit_smoke.yaml EVAL_LIMIT=20
make eval-logit-smoke
~~~

The logit validator downloads model metadata and both tokenizers, resolves
their revisions, checks token-ID compatibility, and inspects response rows. It
does not download teacher weights, allocate a GPU, or run forward passes. The
logit smoke command is the first end-to-end model-weight and CUDA validation.

The smoke chain is:

~~~text
configs/logit_distill_smoke.yaml
  -> logit smoke final
  -> configs/dpo_logit_smoke.yaml
  -> logit-DPO smoke final
  -> configs/eval_logit_smoke.yaml
~~~

### Production

~~~bash
make validate-logit-inputs LOGIT_DATA_LIMIT=100
CUDA_VISIBLE_DEVICES=0 make train-logit
make validate-dpo-logit-inputs DPO_DATA_LIMIT=100
make train-dpo-logit
make validate-eval-logit-inputs EVAL_LIMIT=200
make eval-logit
make export-logit
~~~

Use `make export-logit-push` only when the configured logit-branch repository
should be created or updated on Hugging Face.

## Bounded Runs and Resume

Full configs can be bounded without using smoke output paths:

~~~bash
make train-response RESPONSE_MAX_STEPS=5 RESPONSE_MAX_TRAIN_SAMPLES=64
make train-logit LOGIT_MAX_STEPS=5 LOGIT_MAX_TRAIN_SAMPLES=64
make train-dpo DPO_MAX_STEPS=5 DPO_MAX_TRAIN_SAMPLES=64
~~~

Resume an interrupted stage with the corresponding checkpoint variable:

~~~bash
make train-response RESPONSE_RESUME_FROM_CHECKPOINT=runs/.../checkpoint-250
make train-logit LOGIT_RESUME_FROM_CHECKPOINT=runs/.../checkpoint-250
make train-dpo DPO_RESUME_FROM_CHECKPOINT=runs/.../checkpoint-250
~~~

Response and DPO can use an Accelerate launcher:

~~~bash
make train-response RESPONSE_LAUNCH="accelerate launch --multi_gpu --num_processes 2"
make train-dpo DPO_LAUNCH="accelerate launch --multi_gpu --num_processes 2"
~~~

The logit branch deliberately rejects multi-GPU execution because its current
contract colocates teacher and student on one supported GPU.

## See Also

- [Architecture](architecture.md)
- [Configuration](configuration.md)
- [Evaluation and Export](evaluation-and-export.md)
- [Response Distillation](response_distillation.md)
- [Logit Distillation](logit_distillation.md)
- [Command Reference](../COMMAND.md)

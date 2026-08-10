# SLM Distillation

Training, alignment, evaluation, and export workflows for compact language
models.

## Overview

SLM Distillation turns published teacher-response and preference datasets into
swappable student checkpoints. It provides two independent training branches:
offline response distillation and local logit distillation, each followed by
optional DPO alignment, evaluation, and export. Synthetic data generation lives
in [`slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data).

## Architecture

~~~text
published SFT data ──> response distillation ──┐
                                              ├─> DPO ─> evaluation ─> export
local teacher + SFT data ─> logit distillation┘
                         published preference data
~~~

The default student is `HuggingFaceTB/SmolLM2-135M-Instruct`. The response
branch consumes saved teacher text without loading a teacher; the logit branch
loads `HuggingFaceTB/SmolLM2-1.7B-Instruct` locally and requires compatible
teacher and student tokenizers. See the
[architecture guide](docs/architecture.md) for component boundaries and
checkpoint handoffs.

## Features

- Offline response distillation with response-only causal language-model loss
- Local teacher-to-student logit distillation with hard-label and KL losses
- DPO alignment for response-distilled and logit-distilled checkpoints
- Hugging Face or local model references with configurable dataset schemas
- Input, tokenizer, dataset, and checkpoint validation before expensive work
- Base/distilled/DPO evaluation on response and preference signals
- Hugging Face model-card export and optional Hub upload
- Checksummed artifact packaging with optional S3 handoff
- Isolated smoke paths that never reuse full-run checkpoints

## Getting Started

Requirements:

- Python 3.12+
- CUDA-capable GPU for training
- Hugging Face access for model and dataset downloads

~~~bash
git clone https://github.com/tohio/slm-distillation.git
cd slm-distillation

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

Validate and smoke-test the response branch:

~~~bash
make validate-response-inputs RESPONSE_DATA_LIMIT=100
make train-response-smoke
make train-dpo-smoke
make eval-response-smoke
~~~

Run the production response branch after the smoke path passes:

~~~bash
make train-response
make train-dpo
make eval-response
make export
~~~

See [Training](docs/training.md) for both end-to-end branches and
[COMMAND.md](COMMAND.md) for the complete Make target and variable reference.

## Project Structure

~~~text
.
├── configs/        YAML contracts for training, evaluation, export, and artifacts
├── data/           Dataset ownership notes; datasets remain on Hugging Face
├── distill/        Core Python package
├── docs/           Architecture and workflow documentation
├── scripts/        Command-line entry points used by the Makefile
└── tests/          Unit and contract tests
~~~

Folder-specific conventions are documented in
[`distill/README.md`](distill/README.md),
[`configs/README.md`](configs/README.md),
[`scripts/README.md`](scripts/README.md), and
[`data/README.md`](data/README.md).

## Documentation

Use the [documentation index](docs/README.md) to find architecture, training,
configuration, evaluation, export, and logit-distillation guides.

## Testing

~~~bash
make test
~~~

The suite validates configuration contracts, model resolution, response and
preference schemas, training data preparation, loss behavior, branch
handoffs, evaluation output, export orchestration, and artifact integrity. See
the unchanged [test guide](tests/README.md) for the detailed coverage list.

## Status

| Component | Status |
|---|---|
| Published dataset consumption | Implemented |
| Response distillation | Implemented |
| Local logit distillation | Implemented |
| Response and logit DPO branches | Implemented |
| Response and preference evaluation | Implemented |
| Hugging Face export | Implemented |
| Artifact packaging and S3 handoff | Implemented |
| Synthetic dataset production | External: `slm-synthetic-data` |

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Released under the MIT License.

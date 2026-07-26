# Evaluation

## Purpose

`distill/eval` owns response generation, reference-based metrics, preference
accuracy, model comparisons, and result serialization. It does not define
training claims or create held-out datasets.

## Contents

~~~text
eval/
├── compare_outputs.py
└── run_eval.py
~~~

## Key Files

| File | Responsibility |
|---|---|
| `compare_outputs.py` | Text normalization, exact match, token F1, completeness, and length metrics |
| `run_eval.py` | Input validation, generation, preference scoring, comparisons, and output files |

## How It Fits In

The response and logit evaluation configs each compare a base model, a
distilled checkpoint, and a DPO checkpoint. Smoke configs consume only smoke
checkpoints; full configs consume only full checkpoints.

## Usage/API

~~~bash
make eval-response-dry-run
make eval-logit-dry-run
make eval-response
make eval-logit
~~~

See [Evaluation and Export](../../docs/evaluation-and-export.md) for metrics and
output locations.

## Conventions

- Treat the first configured model as the comparison baseline.
- Write predictions as JSONL and aggregate results as JSON.
- Use deterministic generation unless the config explicitly enables sampling.
- Validate response and preference schemas before model evaluation.

## Gotchas

Bundled configs sample published training splits for pipeline validation. Use
independent held-out datasets for final quality claims.

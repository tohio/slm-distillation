# Data

## Purpose

`data` documents dataset ownership and local data policy. Training datasets are
loaded from Hugging Face by `distill/data`; this folder does not generate,
vendor, cache, or publish dataset rows.

## Contents

~~~text
data/
└── README.md
~~~

## Key Files

There are no local dataset files by design.

| Published dataset | Use |
|---|---|
| `tohio/slm-synthetic-distillation-sft` | Response and logit distillation; response evaluation |
| `tohio/slm-synthetic-distillation-dpo` | DPO training; preference evaluation |

## How It Fits In

YAML files under `configs/` declare dataset IDs, splits, and field mappings.
`distill/data/response.py` and `distill/data/preference.py` load and validate
them.

## Usage/API

~~~bash
make validate-response-inputs RESPONSE_DATA_LIMIT=100
make validate-logit-inputs LOGIT_DATA_LIMIT=100
~~~

Preference validation occurs through the matching DPO validation target after
its upstream checkpoint exists.

## Conventions

- Keep published data external to the repository.
- Configure schemas instead of hard-coding provider-specific columns.
- Keep dataset-generation logic in `slm-synthetic-data`.
- Use independent held-out data for final evaluation claims.

## Gotchas

The bundled evaluation configs use sampled training rows for pipeline
validation; they are not a held-out benchmark.

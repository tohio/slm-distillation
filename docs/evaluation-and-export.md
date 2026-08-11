# Evaluation and Export

Evaluation signals, result files, model-card generation, Hugging Face upload,
and artifact handoff for both branches.

## Evaluation

Each branch compares three checkpoints:

1. the base SmolLM2-135M-Instruct model;
2. the branch-specific distilled checkpoint;
3. the branch-specific DPO checkpoint.

Response generation reports:

- exact match;
- normalized exact match;
- token F1;
- non-empty output rate;
- average output length.

Preference evaluation reports chosen-over-rejected accuracy using conditional
completion log probabilities. Comparison deltas use the first configured model
as the baseline.

Each configured model is loaded once, reused for generation and preference
scoring, and released before the next model is loaded.

Evaluation writes one prediction JSONL file per model and a branch summary:

| Branch | Result |
|---|---|
| Response | `runs/smollm2-135m-response-distilled/eval/results.json` |
| Logit | `runs/smollm2-135m-logit-distilled/eval/results.json` |
| Response smoke | `runs/smoke/smollm2-135m-response-distilled/eval/results.json` |
| Logit smoke | `runs/smoke/smollm2-135m-logit-distilled/eval/results.json` |

The bundled evaluation configs sample the published training datasets to
validate the pipeline. Use held-out datasets for final model comparisons.

## Model Cards

Export validates the final model and tokenizer paths, then writes a model card
containing source checkpoint, teacher provenance, distillation type, DPO
status, dataset IDs, and evaluation result location.

~~~bash
make export
make export-logit
~~~

These commands validate local export inputs and write model cards. They do not
upload unless the config enables pushing or an explicit push target is used.

## Hugging Face Upload

~~~bash
make export-push
make export-logit-push
~~~

Push targets require `HF_TOKEN`. They create the configured model repository
when needed, upload the checkpoint and tokenizer, and publish the generated
model card as `README.md`.

## Artifact Handoff

Package and verify branch outputs:

~~~bash
make pack-artifacts
make verify-artifacts

make pack-artifacts-logit
make verify-artifacts-logit
~~~

Optional S3 transfer:

~~~bash
make push-artifacts
make pull-artifacts

make push-artifacts-logit
make pull-artifacts-logit
~~~

`make pack-artifacts` stages the configured files, writes a SHA-256 manifest,
and creates a local `.tar.gz`. `make push-artifacts` stages the same files but
uploads each file and `manifest.json` as an individual S3 object; it does not
upload the `.tar.gz`.

Each branch includes the pre-DPO final checkpoint, DPO final checkpoint,
top-level evaluation results, and model card. Published datasets, model-cache
copies of the base or teacher, intermediate checkpoints, and nested evaluation
prediction files are excluded. S3 is optional and independent of Hugging Face
model export. With `delete_remote_extra: true`, a push removes stale objects
under that branch's configured S3 run prefix.

## See Also

- [Architecture](architecture.md)
- [Training](training.md)
- [Configuration](configuration.md)
- [`distill/eval/README.md`](../distill/eval/README.md)
- [`distill/export/README.md`](../distill/export/README.md)
- [`distill/artifacts/README.md`](../distill/artifacts/README.md)

# Export

Export and model-card helpers for final distilled model variants.

## Files

| File | Purpose |
|---|---|
| `model_card.py` | Builds and writes model card content. |
| `export_model.py` | Builds export plans and writes model cards. |

## Inputs

| Input | Purpose |
|---|---|
| Final checkpoint path | Post-DPO model checkpoint. |
| Tokenizer path | Tokenizer to include with exported model. |
| Export config | Model name, target repo, model card metadata, and export flags. |

## Output

The export stage writes a model card and prepares the final model export plan.

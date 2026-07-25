# Export

Model-card generation and final checkpoint export.

The export command validates checkpoint and tokenizer paths, writes the
provenance model card, and optionally creates and uploads the configured
Hugging Face model repository. A push requires `HF_TOKEN`.

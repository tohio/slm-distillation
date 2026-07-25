# Tests

Run the suite with:

```bash
make test
```

Coverage includes:

- response model, dataset, formatting, and training configuration
- local and Hugging Face model reference resolution
- response-dataset schema conversion and validation
- response-only label masking and sequence truncation
- preference-dataset schema conversion and validation
- DPO, logit, evaluation, and export configuration
- tokenizer compatibility
- response metrics and branch checkpoint comparisons
- model-card generation and Hugging Face export orchestration
- artifact packaging, checksums, and handoff
- environment loading

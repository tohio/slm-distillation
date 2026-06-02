# Generation

Prompt loading and teacher response generation.

## Files

| File | Purpose |
|---|---|
| `prompts.py` | Loads prompt seed JSONL files. |
| `records.py` | Defines raw teacher response records and JSONL IO. |
| `generate_responses.py` | Converts prompts into raw teacher response records. |
| `batch_runner.py` | Reserved for batching/concurrency support. |

## Flow

```text
prompt JSONL
  -> provider.generate()
  -> raw teacher response JSONL
```

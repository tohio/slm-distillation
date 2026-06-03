from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.generation.adaptive import is_retryable_provider_error, sleep
from distill.generation.prompts import PromptRecord
from distill.providers.base import GenerationRequest, GenerationResponse


@dataclass(frozen=True)
class HostedGenerationResult:
    output_path: str
    written: int
    skipped: int
    errors: int


@dataclass(frozen=True)
class BatchGenerationItem:
    prompt: PromptRecord
    output: str
    response: GenerationResponse | None
    error: str | None = None


class HostedBatchStore:
    def __init__(self, output_path: Path, run_key: str):
        self.output_path = output_path

        # Normal repo outputs use runs/ so make clean-generated clears state.
        # Pytest tmp outputs keep manifests beside the tmp output to avoid
        # cross-test contamination from repeated raw.jsonl filenames.
        if output_path.is_absolute() and "pytest-" in str(output_path):
            root = output_path.parent / ".hosted_generation"
        else:
            root = Path("runs") / "hosted_generation"

        self.batch_dir = root / run_key / "batches"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_id: int) -> Path:
        return self.batch_dir / f"batch_{batch_id:09d}.json"

    def completed_batch_ids(self) -> set[int]:
        return {
            int(path.stem.split("_")[1])
            for path in self.batch_dir.glob("batch_*.json")
            if "_" in path.stem
        }

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_completed(
        self,
        *,
        batch_id: int,
        prompt_ids: list[str],
        records: list[dict[str, Any]],
        telemetry: dict[str, Any],
    ) -> None:
        self._atomic_json(
            self._path(batch_id),
            {
                "batch_id": int(batch_id),
                "prompt_ids": prompt_ids,
                "records": records,
                "telemetry": telemetry,
            },
        )

    def materialize_raw(self, preexisting_records: list[dict[str, Any]] | None = None) -> int:
        temp = self.output_path.with_suffix(".tmp")
        rows = 0
        with temp.open("w", encoding="utf-8") as handle:
            for record in preexisting_records or []:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
            for path in sorted(self.batch_dir.glob("batch_*.json")):
                for record in self._load(path).get("records", []):
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.output_path)
        return rows

    def telemetry_summary(self) -> dict[str, Any]:
        result = {
            "batches": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retry_count": 0,
            "retryable_provider_retries": 0,
            "adaptive_peak_in_flight_limit": 0,
            "adaptive_window_increases": 0,
            "adaptive_window_decreases": 0,
        }
        for path in sorted(self.batch_dir.glob("batch_*.json")):
            result["batches"] += 1
            telemetry = self._load(path).get("telemetry", {}) or {}
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "retry_count",
                "retryable_provider_retries",
                "adaptive_window_increases",
                "adaptive_window_decreases",
            ):
                result[key] += int(telemetry.get(key, 0) or 0)
            result["adaptive_peak_in_flight_limit"] = max(
                result["adaptive_peak_in_flight_limit"],
                int(telemetry.get("adaptive_peak_in_flight_limit", 0) or 0),
            )
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
        return result


def build_batch_prompt(prompts: list[PromptRecord]) -> str:
    items = [{"id": p.id, "category": p.category, "prompt": p.prompt} for p in prompts]
    return (
        "Generate one supervised distillation output for each input item below. "
        "Preserve each id exactly and return items in the same order. "
        "Return only a JSON object with key `items`. "
        "Each item must contain exactly the original id and an output string. "
        "Each output must be concise, complete, directly useful, and free of unrelated filler.\n\n"
        "INPUT ITEMS:\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


def _chunked(items: list[PromptRecord], size: int) -> list[list[PromptRecord]]:
    if size <= 0:
        raise ValueError("batch_size must be greater than zero")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_json_value(text: str) -> Any:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").strip()
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        object_start, object_end = raw.find("{"), raw.rfind("}")
        array_start, array_end = raw.find("["), raw.rfind("]")
        if object_start != -1 and object_end > object_start:
            return json.loads(raw[object_start : object_end + 1])
        if array_start != -1 and array_end > array_start:
            return json.loads(raw[array_start : array_end + 1])
        raise


def parse_batch_response(*, prompts: list[PromptRecord], response_text: str) -> dict[str, str]:
    parsed = _extract_json_value(response_text)
    items = parsed.get("items") if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else None
    if not isinstance(items, list):
        raise ValueError("batch response must be a JSON object with items array")
    if len(items) != len(prompts):
        raise ValueError(
            f"batch response item count mismatch: expected={len(prompts)}, got={len(items)}"
        )

    outputs: dict[str, str] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"batch response item {idx} must be an object")
        expected = prompts[idx].id
        if item.get("id") != expected:
            raise ValueError(
                f"batch response id mismatch at index {idx}: expected={expected}, got={item.get('id')}"
            )
        output = item.get("output")
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f"batch response item {idx} has empty output")
        outputs[expected] = output.strip()
    return outputs


def _record(
    *,
    prompt: PromptRecord,
    output: str,
    teacher_model: str,
    provider_name: str,
    response: GenerationResponse | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "id": prompt.id,
        "prompt_id": prompt.id,
        "category": prompt.category,
        "prompt": prompt.prompt,
        "teacher": teacher_model,
        "teacher_model": teacher_model,
        "teacher_provider": provider_name,
        "provider": provider_name,
        "model": teacher_model,
        "response": output,
        "output": output,
        "input_tokens": response.input_tokens if response else None,
        "output_tokens": response.output_tokens if response else None,
        "metadata": prompt.metadata,
        "raw": response.raw if response else None,
        "error": error,
    }


def _generate_batch_once(
    *,
    provider: Any,
    prompts: list[PromptRecord],
    teacher_model: str,
    provider_name: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
) -> list[BatchGenerationItem]:
    if getattr(provider, "provider_name", "") == "dry_run":
        return [
            BatchGenerationItem(
                prompt=p,
                output=f"Dry run response for: {p.prompt}",
                response=GenerationResponse(
                    text="{}",
                    model=teacher_model,
                    provider=provider_name,
                    input_tokens=len(p.prompt.split()),
                    output_tokens=8,
                    raw={"dry_run": True},
                ),
            )
            for p in prompts
        ]

    response = provider.generate(
        GenerationRequest(
            prompt=build_batch_prompt(prompts),
            model=teacher_model,
            max_output_tokens=max_output_tokens * len(prompts),
            temperature=temperature,
            top_p=top_p,
        )
    )
    try:
        outputs = parse_batch_response(prompts=prompts, response_text=response.text)
    except Exception:
        if len(prompts) != 1:
            raise
        plain = response.text.strip()
        if not plain:
            raise
        outputs = {prompts[0].id: plain}

    return [BatchGenerationItem(prompt=p, output=outputs[p.id], response=response) for p in prompts]


def _generate_batch_with_split(**kwargs: Any) -> list[BatchGenerationItem]:
    prompts = kwargs["prompts"]
    min_batch_size = kwargs["min_batch_size"]
    continue_on_error = kwargs["continue_on_error"]

    try:
        call_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"min_batch_size", "continue_on_error"}
        }
        return _generate_batch_once(**call_kwargs)
    except Exception as exc:
        # Retryable provider failures should be handled by the outer batch
        # scheduler so the whole batch can be requeued instead of recursively
        # split into many smaller failed requests.
        if is_retryable_provider_error(exc):
            raise

        if len(prompts) > min_batch_size:
            mid = len(prompts) // 2
            left = dict(kwargs, prompts=prompts[:mid])
            right = dict(kwargs, prompts=prompts[mid:])
            return _generate_batch_with_split(**left) + _generate_batch_with_split(**right)

        if not continue_on_error:
            raise

        return [BatchGenerationItem(prompt=p, output="", response=None, error=str(exc)) for p in prompts]


def _telemetry(items: list[BatchGenerationItem]) -> dict[str, Any]:
    out = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "retry_count": 0,
        "retryable_provider_retries": 0,
        "adaptive_peak_in_flight_limit": 0,
        "adaptive_window_increases": 0,
        "adaptive_window_decreases": 0,
    }
    for item in items:
        if item.response is None:
            continue
        out["prompt_tokens"] += int(item.response.input_tokens or 0)
        out["completion_tokens"] += int(item.response.output_tokens or 0)
        raw = item.response.raw if isinstance(item.response.raw, dict) else {}
        telemetry = raw.get("telemetry", {}) if isinstance(raw, dict) else {}
        for key in (
            "retry_count",
            "retryable_provider_retries",
            "adaptive_window_increases",
            "adaptive_window_decreases",
        ):
            out[key] += int(telemetry.get(key, 0) or 0)
        out["adaptive_peak_in_flight_limit"] = max(
            out["adaptive_peak_in_flight_limit"],
            int(telemetry.get("adaptive_peak_in_flight_limit", 0) or 0),
        )
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def _error_records(
    *,
    prompts: list[PromptRecord],
    error: Exception,
    teacher_model: str,
    provider_name: str,
) -> list[BatchGenerationItem]:
    return [
        BatchGenerationItem(prompt=p, output="", response=None, error=str(error))
        for p in prompts
    ]


def _control(controls: Any, name: str, default: Any) -> Any:
    return getattr(controls, name, default) if controls is not None else default


def _run_key(provider_name: str, teacher_model: str, output_path: Path) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in teacher_model)
    if output_path.is_absolute() and "pytest-" in str(output_path):
        safe_path = str(output_path.parent).replace("/", "_").replace(":", "_")
        return f"{provider_name}_{safe}_{safe_path}_{output_path.stem}"
    return f"{provider_name}_{safe}_{output_path.stem}"


def _load_existing_records(output_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not output_path.exists():
        return [], set()

    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            prompt_id = row.get("prompt_id") or row.get("id")
            if isinstance(prompt_id, str) and prompt_id:
                ids.add(prompt_id)
                records.append(row)
    return records, ids


def run_hosted_generation(
    *,
    provider: Any,
    prompts: list[PromptRecord],
    output_path: str | Path,
    teacher_model: str,
    provider_name: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    controls: Any,
    continue_on_error: bool,
    batch_size: int | None = None,
    min_batch_size: int | None = None,
    parallel_requests: int | None = None,
    progress_interval: int | None = None,
    resume: bool = True,
) -> HostedGenerationResult:
    output = Path(output_path)
    effective_batch_size = int(batch_size or _control(controls, "batch_size", 1) or 1)
    effective_min_batch_size = int(min_batch_size or _control(controls, "min_batch_size", 1) or 1)
    effective_parallel = int(
        parallel_requests
        or _control(controls, "parallel_requests", None)
        or _control(controls, "concurrency", 1)
        or 1
    )
    effective_progress_interval = int(
        progress_interval or _control(controls, "progress_interval", 25) or 25
    )
    max_requeues = int(_control(controls, "max_requeues", 3) or 0)
    requeue_delay = float(_control(controls, "exhausted_retryable_requeue_delay_seconds", 0.0) or 0.0)

    if effective_batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if effective_min_batch_size <= 0:
        raise ValueError("min_batch_size must be greater than zero")
    if effective_min_batch_size > effective_batch_size:
        raise ValueError("min_batch_size cannot be greater than batch_size")
    if effective_parallel <= 0:
        raise ValueError("parallel_requests must be greater than zero")

    store = HostedBatchStore(output, _run_key(provider_name, teacher_model, output))
    completed_ids = store.completed_batch_ids() if resume else set()

    preexisting_records: list[dict[str, Any]] = []
    preexisting_ids: set[str] = set()
    if resume and not completed_ids:
        preexisting_records, preexisting_ids = _load_existing_records(output)

    if completed_ids and resume:
        store.materialize_raw()

    prompts_to_run = [prompt for prompt in prompts if prompt.id not in preexisting_ids]
    all_batches = list(enumerate(_chunked(prompts_to_run, effective_batch_size)))
    pending = [(batch_id, batch) for batch_id, batch in all_batches if batch_id not in completed_ids]
    skipped = len(preexisting_ids) + sum(
        len(batch) for batch_id, batch in all_batches if batch_id in completed_ids
    )
    total = sum(len(batch) for _, batch in pending)

    if total == 0:
        if preexisting_records:
            store.materialize_raw(preexisting_records=preexisting_records)
        return HostedGenerationResult(str(output), 0, skipped, 0)

    print(
        f"Hosted generation plan: prompts={len(prompts)} skipped={skipped} remaining={total} "
        f"batches={len(pending)}/{len(all_batches)} batch_size={effective_batch_size} "
        f"min_batch_size={effective_min_batch_size} parallel_requests={effective_parallel}",
        flush=True,
    )

    written = errors = completed = 0
    next_progress = effective_progress_interval
    start = time.time()
    requeues: dict[int, int] = {}

    def submit(executor: ThreadPoolExecutor, batch_id: int, batch: list[PromptRecord]):
        return executor.submit(
            _generate_batch_with_split,
            provider=provider,
            prompts=batch,
            teacher_model=teacher_model,
            provider_name=provider_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            min_batch_size=effective_min_batch_size,
            continue_on_error=continue_on_error,
        )

    with ThreadPoolExecutor(max_workers=effective_parallel) as executor:
        futures = {submit(executor, batch_id, batch): (batch_id, batch) for batch_id, batch in pending}

        while futures:
            for future in as_completed(list(futures)):
                batch_id, batch = futures.pop(future)

                try:
                    items = future.result()
                except Exception as exc:
                    if is_retryable_provider_error(exc) and requeues.get(batch_id, 0) < max_requeues:
                        requeues[batch_id] = requeues.get(batch_id, 0) + 1
                        if requeue_delay > 0:
                            sleep(requeue_delay)
                        futures[submit(executor, batch_id, batch)] = (batch_id, batch)
                        continue

                    if not continue_on_error:
                        raise
                    items = _error_records(
                        prompts=batch,
                        error=exc,
                        teacher_model=teacher_model,
                        provider_name=provider_name,
                    )

                records = []
                for item in items:
                    written += 1
                    completed += 1
                    if item.error:
                        errors += 1
                    records.append(
                        _record(
                            prompt=item.prompt,
                            output=item.output,
                            teacher_model=teacher_model,
                            provider_name=provider_name,
                            response=item.response,
                            error=item.error,
                        )
                    )

                telemetry = _telemetry(items)
                telemetry["retryable_provider_retries"] += requeues.get(batch_id, 0)
                store.write_completed(
                    batch_id=batch_id,
                    prompt_ids=[p.id for p in batch],
                    records=records,
                    telemetry=telemetry,
                )

                if completed >= next_progress or completed >= total:
                    elapsed = max(0.001, time.time() - start)
                    metrics = store.telemetry_summary()
                    print(
                        f"Hosted generation progress: batches={metrics['batches']}/{len(pending)} "
                        f"last_batch_records={len(items)} completed={completed}/{total} "
                        f"written={written} errors={errors} skipped={skipped} "
                        f"elapsed_seconds={elapsed:.1f} records_per_second={completed / elapsed:.2f} "
                        f"prompt_tokens={metrics['prompt_tokens']} completion_tokens={metrics['completion_tokens']} "
                        f"retryable_provider_retries={metrics['retryable_provider_retries']} "
                        f"adaptive_peak_in_flight_limit={metrics['adaptive_peak_in_flight_limit']}",
                        flush=True,
                    )
                    while next_progress <= completed:
                        next_progress += effective_progress_interval
                break

    total_materialized = store.materialize_raw(preexisting_records=preexisting_records)
    elapsed = max(0.001, time.time() - start)
    metrics = store.telemetry_summary()
    print(
        f"Hosted generation completed: records={total_materialized} new_records={written} "
        f"errors={errors} elapsed_seconds={elapsed:.1f} records_per_second={written / elapsed:.2f} "
        f"prompt_tokens={metrics['prompt_tokens']} completion_tokens={metrics['completion_tokens']} "
        f"retry_count={metrics['retry_count']} retryable_provider_retries={metrics['retryable_provider_retries']} "
        f"adaptive_window_increases={metrics['adaptive_window_increases']} "
        f"adaptive_window_decreases={metrics['adaptive_window_decreases']} "
        f"adaptive_peak_in_flight_limit={metrics['adaptive_peak_in_flight_limit']}",
        flush=True,
    )
    return HostedGenerationResult(str(output), written, skipped, errors)


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from distill.generation.hosted_controls import hosted_controls_from_mapping
from distill.generation.hosted_runner import run_hosted_generation, write_jsonl
from distill.generation.prompts import PromptRecord, load_merged_prompt_records
from distill.providers.base import GenerationResponse
from distill.providers.groq import GroqProvider
from distill.providers.openrouter import OpenRouterProvider
from distill.utils.config import load_response_distill_config, load_teachers_config


class DryRunProvider:
    provider_name = "dry_run"

    def generate(self, request: Any) -> GenerationResponse:
        return GenerationResponse(
            text=f"Dry run response for: {request.prompt}",
            model=request.model,
            provider=self.provider_name,
            input_tokens=len(request.prompt.split()),
            output_tokens=8,
            raw={"dry_run": True},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw teacher responses.")
    parser.add_argument(
        "--config",
        default="configs/response_distill.yaml",
        help="Path to response distillation config YAML.",
    )
    parser.add_argument(
        "--teachers",
        default="configs/teachers.yaml",
        help="Path to teacher registry YAML.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional prompt limit for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write deterministic dry-run records without calling a hosted API.",
    )
    return parser.parse_args()


def _teacher_map(teachers_config: Any) -> dict[str, Any]:
    candidates = [
        getattr(teachers_config, "teachers", None),
        getattr(teachers_config, "models", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    if isinstance(teachers_config, dict):
        raw = teachers_config.get("teachers") or teachers_config
        if isinstance(raw, dict):
            return raw
    raise TypeError("Unable to resolve teacher registry mapping")


def _teacher_field(teacher: Any, field: str) -> Any:
    if isinstance(teacher, dict):
        return teacher[field]
    return getattr(teacher, field)


def _provider_for_name(provider_name: str, dry_run: bool) -> Any:
    if dry_run:
        return DryRunProvider()

    if provider_name == "openrouter":
        return OpenRouterProvider()

    if provider_name == "groq":
        return GroqProvider()

    raise SystemExit(f"Unsupported provider for generation: {provider_name}")


def _provider_generation_config(config_path: str, provider_name: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    providers = raw.get("providers", {}) or {}
    provider_cfg = providers.get(provider_name, {}) or {}
    return provider_cfg.get("generation", {}) or {}


def main() -> None:
    args = parse_args()

    run_config = load_response_distill_config(args.config)
    teachers_config = load_teachers_config(args.teachers)
    teacher = _teacher_map(teachers_config)[run_config.teacher_name]

    teacher_provider = _teacher_field(teacher, "provider")
    teacher_model = _teacher_field(teacher, "model")

    prompts = load_merged_prompt_records(run_config.data.prompts_paths)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    provider = _provider_for_name(teacher_provider, args.dry_run)

    if args.dry_run:
        controls = hosted_controls_from_mapping(
            {
                "concurrency": 1,
                "adaptive_concurrency_enabled": False,
                "retry_backoff_initial_seconds": 0.0,
                "retry_backoff_max_seconds": 0.0,
                "retry_jitter_ratio": 0.0,
            }
        )
    else:
        controls = hosted_controls_from_mapping(
            _provider_generation_config(args.config, teacher_provider)
        )

    result = run_hosted_generation(
        provider=provider,
        prompts=prompts,
        output_path=run_config.data.raw_teacher_path,
        teacher_model=teacher_model,
        provider_name=teacher_provider,
        max_output_tokens=run_config.distillation.max_output_tokens,
        temperature=run_config.distillation.temperature,
        top_p=run_config.distillation.top_p,
        controls=controls,
        continue_on_error=run_config.distillation.continue_on_error,
    )

    print(f"Wrote raw teacher records: {result.written}")
    if result.errors:
        print(f"Generation errors: {result.errors}")
    print(f"Output: {result.output_path}")


if __name__ == "__main__":
    main()

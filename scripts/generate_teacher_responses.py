from __future__ import annotations

import argparse

from distill.generation.generate_responses import (
    generate_teacher_records,
    write_teacher_records,
)
from distill.generation.prompts import load_merged_prompt_records
from distill.providers.base import GenerationRequest, GenerationResponse
from distill.providers.openrouter import OpenRouterProvider
from distill.utils.config import (
    load_response_distill_config,
    load_teachers_config,
)


class DryRunProvider:
    provider_name = "dry_run"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text=f"[dry-run] {request.prompt}",
            model=request.model,
            provider=self.provider_name,
            input_tokens=None,
            output_tokens=None,
            raw={"dry_run": True},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher responses.")
    parser.add_argument("--config", required=True, help="Response distill config YAML.")
    parser.add_argument(
        "--teachers",
        default="configs/teachers.yaml",
        help="Teachers config YAML.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of prompts to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic local responses without calling the provider API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_config = load_response_distill_config(args.config)
    teachers_config = load_teachers_config(args.teachers)

    teacher_key = run_config.teacher_name
    if teacher_key not in teachers_config.teachers:
        raise SystemExit(f"Unknown teacher in config: {teacher_key}")

    teacher = teachers_config.teachers[teacher_key]

    if teacher.provider != "openrouter":
        raise SystemExit(f"Unsupported provider for generation: {teacher.provider}")

    prompts = load_merged_prompt_records(run_config.data.prompts_paths)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    provider = DryRunProvider() if args.dry_run else OpenRouterProvider()

    records = generate_teacher_records(
        prompts=prompts,
        provider=provider,
        teacher_model=teacher.model,
        max_output_tokens=run_config.distillation.max_output_tokens,
        temperature=run_config.distillation.temperature,
        top_p=run_config.distillation.top_p,
        max_retries=run_config.distillation.max_retries,
        retry_delay_seconds=run_config.distillation.retry_delay_seconds,
        continue_on_error=run_config.distillation.continue_on_error,
    )

    count = write_teacher_records(run_config.data.raw_teacher_path, records)

    print(f"Wrote raw teacher records: {count}")
    print(f"Output: {run_config.data.raw_teacher_path}")


if __name__ == "__main__":
    main()

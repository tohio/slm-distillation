from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from distill.generation.records import TeacherResponseRecord


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class ValidatedTeacherResponse:
    record: TeacherResponseRecord
    validation: ValidationResult


REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i’m unable",
    "i am unable",
    "sorry, but",
    "as an ai",
)


def validate_teacher_response(
    record: TeacherResponseRecord,
    require_non_empty_output: bool = True,
    reject_refusals_when_not_expected: bool = True,
    reject_code_fences_for_function_body_tasks: bool = True,
) -> ValidationResult:
    response = record.response.strip()

    if require_non_empty_output and not response:
        return ValidationResult(accepted=False, reason="empty_response")

    if reject_refusals_when_not_expected:
        lowered = response.lower()
        if any(marker in lowered for marker in REFUSAL_MARKERS):
            if not record.metadata.get("allow_refusal", False):
                return ValidationResult(accepted=False, reason="unexpected_refusal")

    if reject_code_fences_for_function_body_tasks:
        task_type = record.metadata.get("task_type")
        if task_type == "function_body" and "```" in response:
            return ValidationResult(accepted=False, reason="code_fence_in_function_body")

    return ValidationResult(accepted=True)


def split_validated_records(
    records: Iterable[TeacherResponseRecord],
    require_non_empty_output: bool = True,
    reject_refusals_when_not_expected: bool = True,
    reject_code_fences_for_function_body_tasks: bool = True,
) -> tuple[list[ValidatedTeacherResponse], list[ValidatedTeacherResponse]]:
    accepted: list[ValidatedTeacherResponse] = []
    rejected: list[ValidatedTeacherResponse] = []

    for record in records:
        result = validate_teacher_response(
            record,
            require_non_empty_output=require_non_empty_output,
            reject_refusals_when_not_expected=reject_refusals_when_not_expected,
            reject_code_fences_for_function_body_tasks=reject_code_fences_for_function_body_tasks,
        )

        item = ValidatedTeacherResponse(record=record, validation=result)

        if result.accepted:
            accepted.append(item)
        else:
            rejected.append(item)

    return accepted, rejected

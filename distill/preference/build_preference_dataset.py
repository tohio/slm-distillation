from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from distill.generation.records import TeacherResponseRecord, load_teacher_response_records
from distill.preference.records import (
    PreferenceRecord,
    RejectedPreferenceRecord,
    write_preference_records,
    write_rejected_preference_records,
)
from distill.utils.config import PreferenceConfig, load_preference_config


@dataclass(frozen=True)
class PreferenceBuildResult:
    accepted: int
    rejected: int
    output_path: str
    rejected_path: str


def _load_rejected_response_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Rejected response file not found: {input_path}")

    records = load_teacher_response_records(input_path)
    return {record.prompt_id: record.response for record in records}


def _build_pair(
    record: TeacherResponseRecord,
    rejected_response: str,
    config: PreferenceConfig,
) -> PreferenceRecord | RejectedPreferenceRecord:
    chosen = record.response.strip()
    rejected = rejected_response.strip()

    metadata = {
        **record.metadata,
        "prompt_id": record.prompt_id,
        "category": record.category,
        "teacher": record.teacher,
        "provider": record.provider,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
    }

    if config.preference.require_non_empty_chosen and not chosen:
        return RejectedPreferenceRecord(
            prompt=record.prompt,
            chosen=chosen,
            rejected=rejected,
            reason="empty_chosen",
            metadata=metadata,
        )

    if config.preference.require_non_empty_rejected and not rejected:
        return RejectedPreferenceRecord(
            prompt=record.prompt,
            chosen=chosen,
            rejected=rejected,
            reason="empty_rejected",
            metadata=metadata,
        )

    if config.preference.reject_identical_pairs and chosen == rejected:
        return RejectedPreferenceRecord(
            prompt=record.prompt,
            chosen=chosen,
            rejected=rejected,
            reason="identical_pair",
            metadata=metadata,
        )

    return PreferenceRecord(
        prompt=record.prompt,
        chosen=chosen,
        rejected=rejected,
        metadata=metadata,
    )


def build_preference_dataset_from_config(config: PreferenceConfig) -> PreferenceBuildResult:
    teacher_records = load_teacher_response_records(config.source.validated_teacher_path)
    rejected_map = _load_rejected_response_map(config.source.rejected_response_path)

    accepted_records: list[PreferenceRecord] = []
    rejected_records: list[RejectedPreferenceRecord] = []

    for record in teacher_records:
        rejected_response = rejected_map.get(
            record.prompt_id,
            config.preference.default_rejected_response,
        )
        pair = _build_pair(record, rejected_response, config)
        if isinstance(pair, PreferenceRecord):
            accepted_records.append(pair)
        else:
            rejected_records.append(pair)

    accepted_count = write_preference_records(config.data.output_path, accepted_records)
    rejected_count = write_rejected_preference_records(
        config.data.rejected_path,
        rejected_records,
    )

    return PreferenceBuildResult(
        accepted=accepted_count,
        rejected=rejected_count,
        output_path=config.data.output_path,
        rejected_path=config.data.rejected_path,
    )


def build_preference_dataset(config_path: str) -> PreferenceBuildResult:
    return build_preference_dataset_from_config(load_preference_config(config_path))

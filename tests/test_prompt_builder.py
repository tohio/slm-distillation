from pathlib import Path

from distill.generation.prompt_builder import (
    build_prompt_records,
    build_prompts_from_config,
    load_seed_prompts,
)


def test_build_prompt_records_are_diverse() -> None:
    records = build_prompt_records(target_records=100, include_seed_prompts=False)

    assert len(records) == 100
    assert len({record.prompt for record in records}) == 100
    assert len({record.category for record in records}) >= 6
    assert len({record.metadata["family"] for record in records}) >= 6


def test_build_prompts_from_config_writes_jsonl(tmp_path: Path) -> None:
    config_path = tmp_path / "prompts.yaml"
    output_path = tmp_path / "built_prompts.jsonl"

    config_path.write_text(
        f"""
prompts:
  output_path: {output_path}
  seed_paths: []
  default_target_records: 25
  include_seed_prompts: true
""",
        encoding="utf-8",
    )

    result = build_prompts_from_config(config_path)

    assert result.records == 25
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").count("\n") == 25


def test_load_seed_prompts_accepts_instruction_field(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text('{"instruction": "Explain backups.", "category": "ops"}\n', encoding="utf-8")

    records = load_seed_prompts([str(seed_path)])

    assert len(records) == 1
    assert records[0].prompt == "Explain backups."
    assert records[0].category == "ops"

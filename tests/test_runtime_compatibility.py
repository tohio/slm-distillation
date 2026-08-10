from pathlib import Path

from transformers import TrainingArguments
from trl import DPOConfig


def test_transformers_warmup_fraction_contract(tmp_path: Path) -> None:
    arguments = TrainingArguments(
        output_dir=str(tmp_path / "response"),
        warmup_steps=0.03,
        report_to=[],
    )

    assert arguments.get_warmup_steps(100) == 3


def test_trl_warmup_fraction_contract(tmp_path: Path) -> None:
    arguments = DPOConfig(
        output_dir=str(tmp_path / "dpo"),
        warmup_steps=0.03,
        report_to=[],
    )

    assert arguments.get_warmup_steps(100) == 3

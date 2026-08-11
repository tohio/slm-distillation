from __future__ import annotations

import subprocess


def _make_dry_run(target: str, *variables: str) -> str:
    result = subprocess.run(
        ["make", "--no-print-directory", "-n", target, *variables],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_optional_training_arguments_do_not_emit_blank_continuations() -> None:
    for target in ("train-response", "train-logit", "train-dpo", "train-dpo-logit"):
        output = _make_dry_run(target)
        assert not any(line.strip() == "\\" for line in output.splitlines())


def test_optional_training_arguments_are_preserved() -> None:
    output = _make_dry_run(
        "train-dpo",
        "DPO_MAX_STEPS=250",
        "DPO_MAX_TRAIN_SAMPLES=2000",
        "DPO_RESUME_FROM_CHECKPOINT=runs/test/checkpoint-100",
    )

    assert "--max-steps 250" in output
    assert "--max-train-samples 2000" in output
    assert "--resume-from-checkpoint runs/test/checkpoint-100" in output

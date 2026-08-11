from __future__ import annotations

from typing import Any


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 0.0001 or abs(value) >= 10000):
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def format_training_log(
    stage: str,
    *,
    step: int,
    max_steps: int,
    logs: dict[str, Any],
) -> list[str]:
    """Format compact console lines while Trainer retains complete metrics."""
    useful_keys = {
        "loss",
        "learning_rate",
        "grad_norm",
        "rewards/margins",
        "loss/hard",
        "loss/soft_kl",
    }
    if not useful_keys.intersection(logs):
        return []

    common_keys = (
        ("epoch", "epoch"),
        ("loss", "loss"),
        ("learning_rate", "lr"),
        ("grad_norm", "grad"),
    )
    progress = f"[{stage}] step {step}"
    if max_steps > 0:
        progress += f"/{max_steps}"
    common = [progress]
    for key, label in common_keys:
        if key in logs:
            common.append(f"{label} {_format_metric(logs[key])}")

    if "rewards/margins" in logs:
        dpo_keys = (
            ("rewards/chosen", "chosen"),
            ("rewards/rejected", "rejected"),
            ("rewards/margins", "margin"),
            ("rewards/accuracies", "accuracy"),
            ("entropy", "entropy"),
        )
        values = ["reward"] + [
            f"{label} {_format_metric(logs[key])}"
            for key, label in dpo_keys
            if key in logs
        ]
        common.extend(values)
    elif "loss/hard" in logs or "loss/soft_kl" in logs:
        values = ["components"]
        if "loss/hard" in logs:
            values.append(f"hard {_format_metric(logs['loss/hard'])}")
        if "loss/soft_kl" in logs:
            values.append(f"soft_kl {_format_metric(logs['loss/soft_kl'])}")
        common.extend(values)
    return [" | ".join(common)]


def build_compact_logging_callback(stage: str) -> Any:
    from transformers import TrainerCallback

    class CompactLoggingCallback(TrainerCallback):
        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            del args, control, kwargs
            if not state.is_world_process_zero or not logs:
                return
            for line in format_training_log(
                stage,
                step=int(state.global_step),
                max_steps=int(state.max_steps),
                logs=logs,
            ):
                print(line, flush=True)

    return CompactLoggingCallback()


def install_compact_logging(trainer: Any, *, stage: str) -> None:
    from transformers.trainer_callback import PrinterCallback

    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(build_compact_logging_callback(stage))

from distill.utils.logging import format_training_log


def test_format_training_log_keeps_response_metrics_compact() -> None:
    lines = format_training_log(
        "response-distill",
        step=10,
        max_steps=100,
        logs={
            "loss": 1.23456,
            "grad_norm": 2.5,
            "learning_rate": 0.00002,
            "epoch": 0.1,
        },
    )

    assert lines == [
        "[response-distill] step 10/100 | epoch 0.1 | loss 1.2346 | "
        "lr 2.000e-05 | grad 2.5"
    ]


def test_format_training_log_keeps_dpo_metrics_on_one_line() -> None:
    lines = format_training_log(
        "response-dpo",
        step=20,
        max_steps=100,
        logs={
            "loss": 0.5,
            "rewards/chosen": 0.2,
            "rewards/rejected": -0.4,
            "rewards/margins": 0.6,
            "rewards/accuracies": 0.8,
            "entropy": 1.1,
        },
    )

    assert len(lines) == 1
    assert "loss 0.5" in lines[0]
    assert "reward | chosen 0.2" in lines[0]
    assert "rejected -0.4" in lines[0]


def test_format_training_log_shows_logit_loss_components() -> None:
    lines = format_training_log(
        "logit-distill",
        step=5,
        max_steps=10,
        logs={"loss": 5.0, "loss/hard": 1.5, "loss/soft_kl": 8.5},
    )

    assert lines[0].endswith("components | hard 1.5 | soft_kl 8.5")


def test_format_training_log_suppresses_metricless_completion_event() -> None:
    lines = format_training_log(
        "response-dpo",
        step=5,
        max_steps=5,
        logs={"epoch": 0.625, "total_flos": 1000.0},
    )

    assert lines == []

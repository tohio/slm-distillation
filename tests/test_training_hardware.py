from __future__ import annotations

from typing import Any

import pytest

from distill.utils.hardware import validate_single_cuda_gpu


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool,
        device_count: int,
        device_name: str = "NVIDIA A100 80GB PCIe",
    ) -> None:
        self._available = available
        self._device_count = device_count
        self._device_name = device_name

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self._device_name


def fake_torch(cuda: FakeCuda) -> Any:
    return type("FakeTorch", (), {"cuda": cuda})()


def test_single_cuda_gpu_contract_accepts_one_visible_gpu() -> None:
    name = validate_single_cuda_gpu(
        fake_torch(FakeCuda(available=True, device_count=1)),
        stage="Training",
        allowed_gpu_classes=["a100"],
    )

    assert name == "NVIDIA A100 80GB PCIe"


@pytest.mark.parametrize("device_count", [0, 2, 8])
def test_single_cuda_gpu_contract_rejects_non_single_counts(
    device_count: int,
) -> None:
    with pytest.raises(RuntimeError, match="exactly one visible CUDA GPU"):
        validate_single_cuda_gpu(
            fake_torch(FakeCuda(available=True, device_count=device_count)),
            stage="Training",
        )


def test_single_cuda_gpu_contract_requires_cuda() -> None:
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        validate_single_cuda_gpu(
            fake_torch(FakeCuda(available=False, device_count=0)),
            stage="Training",
        )


def test_single_cuda_gpu_contract_rejects_unsupported_gpu_class() -> None:
    with pytest.raises(RuntimeError, match="Unsupported GPU"):
        validate_single_cuda_gpu(
            fake_torch(
                FakeCuda(
                    available=True,
                    device_count=1,
                    device_name="NVIDIA RTX A6000",
                )
            ),
            stage="Logit distillation",
            allowed_gpu_classes=["a100", "h200"],
        )

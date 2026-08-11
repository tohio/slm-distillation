from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def validate_single_cuda_gpu(
    torch: Any,
    *,
    stage: str,
    allowed_gpu_classes: Sequence[str] | None = None,
) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{stage} requires a CUDA GPU")

    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count != 1:
        raise RuntimeError(
            f"{stage} requires exactly one visible CUDA GPU; "
            "set CUDA_VISIBLE_DEVICES to one GPU"
        )

    gpu_name = str(torch.cuda.get_device_name(0))
    if allowed_gpu_classes is None:
        return gpu_name

    normalized_name = gpu_name.lower().replace(" ", "")
    normalized_classes = [
        gpu_class.lower().replace(" ", "")
        for gpu_class in allowed_gpu_classes
    ]
    if not any(
        gpu_class in normalized_name for gpu_class in normalized_classes
    ):
        raise RuntimeError(f"Unsupported GPU for {stage}: {gpu_name}")
    return gpu_name

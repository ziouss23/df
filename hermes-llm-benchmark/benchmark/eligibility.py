from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelEligibility:
    model_id: str
    eligible: bool
    reason: str
    estimated_vram_gb: Optional[float]
    available_vram_gb: float
    required_free_vram_gb: float


def estimate_vram_gb(
    parameter_billions: float,
    bytes_per_parameter: float = 2.0,
) -> float:
    """
    Conservative baseline estimate for model weights.

    This does NOT estimate total runtime VRAM.
    KV cache, activations and framework overhead are handled
    separately by the safety margin.
    """
    return parameter_billions * bytes_per_parameter


def check_model_eligibility(
    model_id: str,
    parameter_billions: float,
    available_vram_gb: float,
    minimum_free_vram_gb: float = 6.0,
    bytes_per_parameter: float = 2.0,
) -> ModelEligibility:
    estimated = estimate_vram_gb(
        parameter_billions,
        bytes_per_parameter,
    )

    usable_vram = available_vram_gb - minimum_free_vram_gb

    if available_vram_gb <= 0:
        return ModelEligibility(
            model_id=model_id,
            eligible=False,
            reason="no CUDA GPU detected",
            estimated_vram_gb=estimated,
            available_vram_gb=available_vram_gb,
            required_free_vram_gb=minimum_free_vram_gb,
        )

    if estimated > usable_vram:
        return ModelEligibility(
            model_id=model_id,
            eligible=False,
            reason=(
                f"estimated weight VRAM {estimated:.1f} GB exceeds "
                f"usable VRAM {usable_vram:.1f} GB after "
                f"{minimum_free_vram_gb:.1f} GB safety margin"
            ),
            estimated_vram_gb=estimated,
            available_vram_gb=available_vram_gb,
            required_free_vram_gb=minimum_free_vram_gb,
        )

    return ModelEligibility(
        model_id=model_id,
        eligible=True,
        reason="model fits estimated VRAM budget",
        estimated_vram_gb=estimated,
        available_vram_gb=available_vram_gb,
        required_free_vram_gb=minimum_free_vram_gb,
    )

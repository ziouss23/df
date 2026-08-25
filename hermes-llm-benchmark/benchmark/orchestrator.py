from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from benchmark.campaign import CampaignLoader, ModelSpec
from benchmark.hardware import detect_hardware


@dataclass(frozen=True)
class Eligibility:
    model_id: str
    eligible: bool
    reason: str


class BenchmarkOrchestrator:
    """
    HERMES LLM Benchmark V5.3 orchestrator.

    Responsibilities:
    - load campaign configuration
    - inspect hardware
    - determine model eligibility
    - prepare campaign execution

    GPU execution is deliberately separated from this first layer.
    """

    def __init__(self, manifests_dir: str = "manifests"):
        self.loader = CampaignLoader(manifests_dir)

        self.campaign = self.loader.campaign()
        self.models = self.loader.enabled_models()

    def preflight(self) -> dict[str, Any]:
        hardware = detect_hardware()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware,
            "cuda_required": self.campaign.require_cuda,
            "preflight_pass": (
                not self.campaign.require_cuda
                or bool(hardware.get("cuda_available"))
            ),
        }

    def eligible_model(
        self,
        model: ModelSpec,
        hardware: dict[str, Any],
    ) -> Eligibility:

        if not model.enabled:
            return Eligibility(
                model.id,
                False,
                "model_disabled",
            )

        if self.campaign.require_cuda and not hardware.get(
            "cuda_available",
            False,
        ):
            return Eligibility(
                model.id,
                False,
                "cuda_required_but_unavailable",
            )

        free_vram = float(
            hardware.get("vram_free_gb", 0.0)
        )

        if free_vram < self.campaign.minimum_free_vram_gb:
            return Eligibility(
                model.id,
                False,
                (
                    "insufficient_free_vram:"
                    f"{free_vram:.2f}GB"
                ),
            )

        return Eligibility(
            model.id,
            True,
            "eligible",
        )

    def eligibility_report(
        self,
        hardware: dict[str, Any],
    ) -> list[Eligibility]:

        return [
            self.eligible_model(model, hardware)
            for model in self.models
        ]

    def plan(self) -> dict[str, Any]:
        preflight = self.preflight()
        hardware = preflight["hardware"]

        eligibility = self.eligibility_report(hardware)

        return {
            "schema_version": "5.3",
            "timestamp": preflight["timestamp"],
            "preflight_pass": preflight["preflight_pass"],
            "hardware": hardware,
            "campaign": {
                "warmups": self.campaign.warmups,
                "measured_runs": self.campaign.measured_runs,
                "max_cv": self.campaign.max_cv,
                "suspect_cv": self.campaign.suspect_cv,
                "confidence_level": self.campaign.confidence_level,
                "minimum_free_vram_gb": (
                    self.campaign.minimum_free_vram_gb
                ),
            },
            "models": [
                {
                    "id": item.model_id,
                    "eligible": item.eligible,
                    "reason": item.reason,
                }
                for item in eligibility
            ],
        }


def main() -> None:
    orchestrator = BenchmarkOrchestrator()

    plan = orchestrator.plan()

    import json

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()

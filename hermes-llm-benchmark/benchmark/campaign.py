from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_id: str
    revision: str
    enabled: bool


@dataclass(frozen=True)
class CampaignSpec:
    version: str
    warmups: int
    measured_runs: int
    max_cv: float
    suspect_cv: float
    confidence_level: float
    minimum_free_vram_gb: float
    require_cuda: bool
    stop_on_load_failure: bool
    stop_on_invalid_metrics: bool
    save_raw_results: bool


class CampaignLoader:
    def __init__(self, manifests_dir: str | Path = "manifests"):
        self.root = Path(manifests_dir)

    def _load(self, name: str) -> dict[str, Any]:
        path = self.root / name

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid manifest: {path}")

        return data

    def models(self) -> list[ModelSpec]:
        data = self._load("models.yaml")

        if data.get("version") != "5.3":
            raise ValueError("models.yaml version must be 5.3")

        result = []

        for model in data.get("models", []):
            result.append(
                ModelSpec(
                    id=model["id"],
                    hf_id=model["hf_id"],
                    revision=model.get("revision", "main"),
                    enabled=bool(model.get("enabled", False)),
                )
            )

        return result

    def campaign(self) -> CampaignSpec:
        data = self._load("gates.yaml")

        if data.get("version") != "5.3":
            raise ValueError("gates.yaml version must be 5.3")

        campaign = data["campaign"]
        statistics = data["statistics"]
        hardware = data["hardware"]
        execution = data["execution"]

        return CampaignSpec(
            version=data["version"],
            warmups=int(campaign["warmups"]),
            measured_runs=int(campaign["measured_runs"]),
            max_cv=float(statistics["max_cv"]),
            suspect_cv=float(statistics["suspect_cv"]),
            confidence_level=float(statistics["confidence_level"]),
            minimum_free_vram_gb=float(hardware["minimum_free_vram_gb"]),
            require_cuda=bool(hardware["require_cuda"]),
            stop_on_load_failure=bool(execution["stop_on_load_failure"]),
            stop_on_invalid_metrics=bool(
                execution["stop_on_invalid_metrics"]
            ),
            save_raw_results=bool(execution["save_raw_results"]),
        )

    def enabled_models(self) -> list[ModelSpec]:
        return [model for model in self.models() if model.enabled]

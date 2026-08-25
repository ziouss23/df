from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LMEvalStatus:
    installed: bool
    executable: str | None
    version: str | None


class LMEvalAdapter:
    """Thin adapter around lm-evaluation-harness."""

    def status(self) -> LMEvalStatus:
        executable = shutil.which("lm_eval")

        if not executable:
            return LMEvalStatus(False, None, None)

        version = None

        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                version = result.stdout.strip() or result.stderr.strip()

        except (OSError, subprocess.SubprocessError):
            pass

        return LMEvalStatus(
            installed=True,
            executable=executable,
            version=version,
        )

    def validate_cli(self) -> bool:
        executable = shutil.which("lm_eval")

        if not executable:
            return False

        try:
            result = subprocess.run(
                [executable, "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False

        return (
            result.returncode == 0
            and "Language Model Evaluation Harness" in result.stdout
        )

    def build_command(
        self,
        model: str,
        tasks: list[str],
        *,
        num_fewshot: int = 0,
        limit: int | None = None,
        output_path: str = "results/raw/lm_eval",
    ) -> list[str]:

        command = [
            "lm_eval",
            "run",
            "--model",
            "vllm",
            "--model_args",
            f"pretrained={model}",
            "--tasks",
            ",".join(tasks),
            "--num_fewshot",
            str(num_fewshot),
            "--output_path",
            output_path,
        ]

        if limit is not None:
            command.extend(["--limit", str(limit)])

        return command

    def run(self, command: list[str]) -> dict[str, Any]:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": command,
        }

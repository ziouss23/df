from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VLLMStatus:
    installed: bool
    executable: str | None
    version: str | None


class VLLMAdapter:
    """Thin adapter around the installed vLLM CLI."""

    def status(self) -> VLLMStatus:
        executable = shutil.which("vllm")

        if not executable:
            return VLLMStatus(False, None, None)

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

        return VLLMStatus(
            installed=True,
            executable=executable,
            version=version,
        )

    def validate_cli(self) -> bool:
        """Verify that vLLM exposes the serving benchmark command."""
        executable = shutil.which("vllm")

        if not executable:
            return False

        try:
            result = subprocess.run(
                [executable, "bench", "serve", "--help"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return False

        return (
            result.returncode == 0
            and "Benchmark the online serving throughput" in result.stdout
        )

    def build_benchmark_command(
        self,
        model: str,
        *,
        input_len: int = 1024,
        output_len: int = 128,
        num_prompts: int = 1000,
        num_warmups: int = 2,
        backend: str = "vllm",
        request_rate: float | None = None,
        save_result: bool = True,
        save_detailed: bool = True,
        result_path: str | None = None,
    ) -> list[str]:
        """Build a real vLLM 0.27.x benchmark command."""

        command = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            backend,
            "--model",
            model,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            "--num-prompts",
            str(num_prompts),
            "--num-warmups",
            str(num_warmups),
        ]

        if request_rate is not None:
            command.extend(["--request-rate", str(request_rate)])

        if save_result:
            command.append("--save-result")

        if save_detailed:
            command.append("--save-detailed")

        if result_path:
            command.extend(["--result-filename", result_path])

        return command

    def run(self, command: list[str]) -> dict[str, Any]:
        """Execute a validated command and return raw process information."""

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

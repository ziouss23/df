import platform
import subprocess
from datetime import datetime, timezone


def detect_platform():
    import os

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"

    if os.getenv("COLAB_GPU"):
        return "colab"

    if os.getenv("SPACE_ID"):
        return "huggingface"

    if os.getenv("CODESPACES"):
        return "codespace"

    return "unknown"


def detect_hardware():
    """
    Retourne un fingerprint matériel.
    Fonctionne avec ou sans GPU CUDA.
    """

    result = {
        "platform": detect_platform(),
        "architecture": platform.machine(),
        "os": platform.system(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "cuda_version": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        import torch

        result["pytorch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()

        if result["cuda_available"]:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)

            result["gpu_name"] = props.name
            result["vram_total_gb"] = round(
                props.total_memory / (1024 ** 3), 3
            )

    except ImportError:
        result["pytorch_version"] = None

    return result


def preflight(require_cuda=False):
    """
    Vérifie si l'environnement est compatible avec l'exécution demandée.
    """

    hardware = detect_hardware()

    if require_cuda and not hardware["cuda_available"]:
        return False, hardware, "CUDA GPU requis mais absent."

    return True, hardware, "Environment OK."


if __name__ == "__main__":
    import json

    hardware = detect_hardware()

    print(json.dumps(hardware, indent=2))

    ok, _, message = preflight(require_cuda=False)

    print()
    print("PREFLIGHT:", "PASS" if ok else "FAIL")
    print(message)

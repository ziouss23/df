from datetime import datetime, timezone


SCHEMA_VERSION = "1.0"


def normalize_performance(raw: dict) -> dict:
    """
    Normalise les métriques de performance provenant de vLLM.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "type": "performance",
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "ttft_ms": raw.get("ttft_mean_ms"),
        "tpot_ms": raw.get("tpot_mean_ms"),
        "itl_ms": raw.get("itl_mean_ms"),
        "e2e_ms": raw.get("e2e_mean_ms"),

        "decode_tokens_per_sec": raw.get(
            "decode_tokens_per_sec"
        ),
        "prefill_tokens_per_sec": raw.get(
            "prefill_tokens_per_sec"
        ),
        "requests_per_sec": raw.get(
            "requests_per_sec"
        ),

        "percentiles": raw.get(
            "percentiles", {}
        ),
    }


def normalize_quality(raw: dict) -> dict:
    """
    Normalise les résultats lm-evaluation-harness.
    """

    scores = raw.get("scores", {})

    valid_scores = {
        name: value
        for name, value in scores.items()
        if isinstance(value, (int, float))
    }

    average = (
        sum(valid_scores.values()) / len(valid_scores)
        if valid_scores
        else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "type": "quality",
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "scores": valid_scores,
        "average_score": average,

        "tasks": raw.get("tasks", []),
        "num_fewshot": raw.get("num_fewshot"),
    }


def normalize_hermes(raw: dict) -> dict:
    """
    Normalise un résultat HERMES-001.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "type": "hermes",

        "workload_id": raw.get(
            "workload_id",
            "HERMES-001"
        ),

        "total_tokens": raw.get(
            "total_tokens"
        ),

        "total_ttft_ms": raw.get(
            "total_ttft_ms"
        ),

        "max_vram_peak_gb": raw.get(
            "max_vram_peak_gb"
        ),

        "steps": raw.get(
            "steps",
            []
        ),
    }

import json
from pathlib import Path
from datetime import datetime, timezone


def append_jsonl(path: str | Path, record: dict) -> None:
    """
    Ajoute un résultat dans un fichier JSONL.
    Une ligne = un événement/run.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = dict(record)

    record.setdefault(
        "logged_at",
        datetime.now(timezone.utc).isoformat()
    )

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":")
            )
            + "\n"
        )


def read_jsonl(path: str | Path) -> list[dict]:
    """
    Recharge tous les records d'un fichier JSONL.
    """

    path = Path(path)

    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide ligne {line_number}: {path}"
                ) from exc

    return records

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Any) -> None:
    ensure_parent(path)
    Path(path).write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def read_json(path: str | Path) -> Any:
    return orjson.loads(Path(path).read_bytes())


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    serialized = b"".join(orjson.dumps(row) + b"\n" for row in rows)
    Path(path).write_bytes(serialized)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_bytes().splitlines():
        if line.strip():
            rows.append(orjson.loads(line))
    return rows

"""Small shared I/O helpers for the reference-only frontend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")


def new_output(path: Path) -> Path:
    result = path.resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result

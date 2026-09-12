"""Resolve relocated, digest-bound reference inputs without rewriting evidence."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from .common import load_json, require, sha256_file

SCHEMA = {"name": "hbserve.reference_artifact_map", "version": 1}
_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("reference_artifacts", default=None)


def contained_path(root: Path, raw: str) -> Path:
    require(isinstance(raw, str) and bool(raw), "artifact path must be nonempty")
    relative = Path(raw)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "artifact path must be relative and cannot traverse parents")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "artifact path escapes its directory")
    return path


@contextmanager
def artifact_context(path: Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    path = path.resolve()
    document = load_json(path)
    require(document.get("schema") == SCHEMA, "unsupported artifact-map schema")
    rows = document.get("artifacts")
    require(isinstance(rows, dict) and bool(rows), "artifact map is empty")
    resolved, verified = {}, {}
    for original, row in rows.items():
        require(isinstance(original, str) and bool(original) and isinstance(row, dict),
                "malformed artifact-map entry")
        source = contained_path(path.parent, row.get("path"))
        require(source.is_file(), f"missing mapped artifact: {source}")
        require(type(row.get("bytes")) is int and row["bytes"] >= 0,
                "mapped artifact size must be a nonnegative integer")
        if source not in verified:
            verified[source] = (source.stat().st_size, sha256_file(source))
        require(verified[source] == (row["bytes"], row.get("sha256")),
                f"mapped artifact size/digest differs: {original}")
        resolved[original] = source
    state = {"paths": resolved, "receipt": {
        "artifact_map_sha256": sha256_file(path), "bindings_verified": len(rows),
        "unique_files_verified": len(verified), "source_bytes_rewritten": False,
    }}
    token = _ACTIVE.set(state)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def resolve_input(raw: str | Path) -> Path:
    state = _ACTIVE.get()
    if state is None:
        return Path(raw).resolve()
    key = str(raw)
    require(key in state["paths"], f"input is not in the explicit artifact map: {key}")
    return state["paths"][key]


def artifact_receipt() -> dict[str, Any] | None:
    state = _ACTIVE.get()
    return dict(state["receipt"]) if state is not None else None

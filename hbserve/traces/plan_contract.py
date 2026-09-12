"""Reference plan validation without importing a historical simulator client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import load_json, require, sha256_file
from ._reference.full_model_trace_plan import PLAN_SCHEMA

SUPPORTED_KINDS = {"weight", "kv_cache", "activation", "anonymous"}


def uint(value: Any, label: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum,
            f"{label} must be an integer >= {minimum}")
    return value


def validate_plan(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    path = plan_path.resolve()
    plan = load_json(path)
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported lazy-plan schema")
    objects = plan.get("objects")
    require(isinstance(objects, list) and objects, "plan has no object table")
    seen = set()
    extents = []
    for index, row in enumerate(objects):
        require(isinstance(row, dict), f"plan object {index} is not an object")
        require(uint(row.get("target_object_index"), "target object index") == index,
                "plan target object indices are not dense")
        name = row.get("object_id")
        require(isinstance(name, str) and name and name not in seen,
                f"plan object {index} has a duplicate or malformed id")
        require(row.get("kind") in SUPPORTED_KINDS, f"plan object {name} has unknown kind")
        begin = uint(row.get("logical_address"), f"{name} logical address")
        size = uint(row.get("bytes"), f"{name} bytes", minimum=1)
        require(begin <= 2**64 - size, f"plan object {name} overflows u64")
        seen.add(name)
        extents.append((begin, begin + size, name))
    extents.sort()
    for left, right in zip(extents, extents[1:]):
        require(left[1] <= right[0], f"plan objects {left[2]} and {right[2]} overlap")
    return plan, objects, sha256_file(path)

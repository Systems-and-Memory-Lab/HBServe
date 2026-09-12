#!/usr/bin/env python3
"""Strict coordinate rules for the prefill shape-aware shadow generator.

The primary phase-rule catalog intentionally remains unchanged.  This module
defines a separate, small rule language for retrospectively diagnosed tensor
geometry.  Every byte-valued field must be non-negative and sector aligned;
unknown or extra fields are rejected so a malformed descriptor cannot silently
fall back to an affine interpretation.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any


RULE_SCHEMA = {"name": "hbfsim.shape_aware_cta_coordinate_rule", "version": 1}
SUPPORTED_KINDS = {"affine", "modulo", "quotient", "mixed_radix", "cyclic"}

_COMMON_FIELDS = {"schema", "kind", "base_bytes", "sector_bytes"}
_KIND_FIELDS = {
    "affine": {"x_stride_bytes"},
    "modulo": {"period", "stride_bytes"},
    "quotient": {"divisor", "stride_bytes"},
    "mixed_radix": {"radix", "inner_stride_bytes", "outer_stride_bytes"},
    "cyclic": {"period_bytes", "stride_bytes"},
}
_COORDINATE = struct.Struct("<Q")


class ShapeAwareRuleError(ValueError):
    """Raised when a shape-aware rule cannot be applied without guessing."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ShapeAwareRuleError(message)


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label}: integer required",
    )
    result = int(value)
    require(result >= minimum, f"{label}: must be at least {minimum}")
    return result


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized rule after strict schema and alignment checks."""

    require(isinstance(rule, dict), "shape-aware rule must be an object")
    require(rule.get("schema") == RULE_SCHEMA, "unsupported shape-aware rule schema")
    kind = rule.get("kind")
    require(isinstance(kind, str) and kind in SUPPORTED_KINDS,
            f"unsupported shape-aware rule kind {kind!r}")
    expected_fields = _COMMON_FIELDS | _KIND_FIELDS[kind]
    actual_fields = set(rule)
    require(
        actual_fields == expected_fields,
        "shape-aware rule fields differ: "
        f"missing={sorted(expected_fields - actual_fields)}, "
        f"extra={sorted(actual_fields - expected_fields)}",
    )

    sector_bytes = integer(rule["sector_bytes"], "sector bytes", minimum=1)
    require(
        sector_bytes & (sector_bytes - 1) == 0,
        "sector bytes must be a power of two",
    )
    normalized: dict[str, Any] = {
        "schema": RULE_SCHEMA,
        "kind": kind,
        "base_bytes": integer(rule["base_bytes"], "base bytes"),
        "sector_bytes": sector_bytes,
    }

    byte_fields: list[str]
    if kind == "affine":
        byte_fields = ["x_stride_bytes"]
    elif kind == "modulo":
        normalized["period"] = integer(rule["period"], "modulo period", minimum=2)
        byte_fields = ["stride_bytes"]
    elif kind == "quotient":
        normalized["divisor"] = integer(
            rule["divisor"], "quotient divisor", minimum=1
        )
        byte_fields = ["stride_bytes"]
    elif kind == "mixed_radix":
        normalized["radix"] = integer(rule["radix"], "mixed-radix radix", minimum=2)
        byte_fields = ["inner_stride_bytes", "outer_stride_bytes"]
    else:
        normalized["period_bytes"] = integer(
            rule["period_bytes"], "cyclic period bytes", minimum=sector_bytes
        )
        byte_fields = ["stride_bytes"]

    for field in byte_fields:
        value = integer(rule[field], field.replace("_", " "))
        require(value % sector_bytes == 0, f"{field} is not sector aligned")
        normalized[field] = value
    require(
        normalized["base_bytes"] % sector_bytes == 0,
        "base_bytes is not sector aligned",
    )
    if kind == "cyclic":
        require(
            normalized["period_bytes"] % sector_bytes == 0,
            "period_bytes is not sector aligned",
        )
        require(
            normalized["base_bytes"] < normalized["period_bytes"],
            "cyclic base_bytes must lie inside the byte period",
        )
    return normalized


def coordinate(rule: dict[str, Any], cta_x: int) -> int:
    """Compute the object-relative coordinate for one non-negative CTA x."""

    normalized = validate_rule(rule)
    x = integer(cta_x, "CTA x")
    return _coordinate_validated(normalized, x)


def _coordinate_validated(normalized: dict[str, Any], x: int) -> int:
    """Fast coordinate path for a rule and CTA x already validated by callers."""

    base = int(normalized["base_bytes"])
    kind = str(normalized["kind"])
    if kind == "affine":
        return base + x * int(normalized["x_stride_bytes"])
    if kind == "modulo":
        return base + (x % int(normalized["period"])) * int(
            normalized["stride_bytes"]
        )
    if kind == "quotient":
        return base + (x // int(normalized["divisor"])) * int(
            normalized["stride_bytes"]
        )
    if kind == "mixed_radix":
        radix = int(normalized["radix"])
        return (
            base
            + (x % radix) * int(normalized["inner_stride_bytes"])
            + (x // radix) * int(normalized["outer_stride_bytes"])
        )
    if kind == "cyclic":
        return (
            base + x * int(normalized["stride_bytes"])
        ) % int(normalized["period_bytes"])
    raise AssertionError("validated shape-aware rule has an unknown kind")


def coordinate_from_validated_rule(rule: dict[str, Any], cta_x: int) -> int:
    """Fast path for callers that already ran :func:`validate_rule`.

    This deliberately performs no schema checks and is intended for the inner
    request-generation loop after the enclosing generator plan has passed its
    fail-closed preparation gate.
    """

    return _coordinate_validated(rule, cta_x)


def coordinate_period_ctas(rule: dict[str, Any]) -> int:
    """Return the finite CTA phase period needed for bounded validation.

    Affine and quotient coordinates have no repeating address phase, so their
    extrema are covered by endpoints and the returned period is one.  A
    mixed-radix rule repeats its inner phase every radix CTAs while its outer
    term remains monotonic.  A cyclic byte coordinate repeats after
    ``period_bytes / gcd(period_bytes, stride_bytes)`` CTAs.
    """

    normalized = validate_rule(rule)
    kind = str(normalized["kind"])
    if kind in {"affine", "quotient"}:
        return 1
    if kind == "modulo":
        return int(normalized["period"])
    if kind == "mixed_radix":
        return int(normalized["radix"])
    if kind == "cyclic":
        return int(normalized["period_bytes"]) // math.gcd(
            int(normalized["period_bytes"]), int(normalized["stride_bytes"])
        )
    raise AssertionError("validated shape-aware rule has an unknown kind")


def validate_full_grid_bounds(
    *,
    rule: dict[str, Any],
    grid_cta_x: int,
    footprint_span_bytes: int,
    object_extent_bytes: int,
) -> dict[str, Any]:
    """Exhaustively validate every coordinate and return a stable receipt."""

    normalized = validate_rule(rule)
    grid = integer(grid_cta_x, "grid CTA x", minimum=1)
    footprint = integer(
        footprint_span_bytes, "footprint span bytes", minimum=1
    )
    extent = integer(object_extent_bytes, "object extent bytes", minimum=1)
    sector = int(normalized["sector_bytes"])
    require(footprint % sector == 0, "footprint span is not sector aligned")
    require(extent % sector == 0, "object extent is not sector aligned")

    digest = hashlib.sha256()
    minimum_coordinate: int | None = None
    maximum_coordinate = -1
    maximum_end = -1
    maximum_end_cta_x = -1
    for cta_x in range(grid):
        value = _coordinate_validated(normalized, cta_x)
        require(value >= 0, f"CTA x={cta_x}: coordinate is negative")
        require(value % sector == 0,
                f"CTA x={cta_x}: coordinate is not sector aligned")
        end = value + footprint
        require(
            end <= extent,
            f"CTA x={cta_x}: footprint end {end} escapes object extent {extent}",
        )
        digest.update(_COORDINATE.pack(value))
        minimum_coordinate = (
            value if minimum_coordinate is None else min(minimum_coordinate, value)
        )
        maximum_coordinate = max(maximum_coordinate, value)
        if end > maximum_end:
            maximum_end = end
            maximum_end_cta_x = cta_x

    return {
        "status": "PASS_EXHAUSTIVE_FULL_GRID_BOUNDS",
        "grid_cta_x": grid,
        "coordinates_checked": grid,
        "sector_bytes": sector,
        "footprint_span_bytes": footprint,
        "object_extent_bytes": extent,
        "minimum_coordinate_bytes": int(minimum_coordinate or 0),
        "maximum_coordinate_bytes": maximum_coordinate,
        "maximum_footprint_end_bytes": maximum_end,
        "maximum_footprint_end_cta_x": maximum_end_cta_x,
        "coordinate_stream_sha256": digest.hexdigest(),
        "rule_sha256": canonical_sha256(normalized),
    }

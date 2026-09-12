#!/usr/bin/env python3
"""Shared validation helpers for fixed-stride compact trace templates."""

from __future__ import annotations

from typing import Any

from hbserve.traces._reference.compact_request_template import require


def object_table(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    objects = manifest.get("objects")
    require(isinstance(objects, list) and objects, "template has no object table")
    require(
        [int(item["template_object_index"]) for item in objects]
        == list(range(len(objects))),
        "template object indices are not dense",
    )
    return objects


def repeat_descriptor(
    manifest: dict[str, Any], objects: list[dict[str, Any]]
) -> tuple[int, list[int]]:
    parameterization = manifest.get("parameterization")
    require(isinstance(parameterization, dict), "template is not parameterized")
    count = int(parameterization.get("repeat_count", 0))
    raw = parameterization.get("object_offset_stride_bytes_by_source_name")
    require(count > 0 and isinstance(raw, dict), "invalid repeat parameterization")
    names = [str(item["source_name"]) for item in objects]
    require(set(raw) == set(names), "repeat stride map must cover every object exactly")
    strides = [int(raw[name]) for name in names]
    require(all(value >= 0 for value in strides), "repeat strides must be nonnegative")
    ordering = parameterization.get(
        "ordering", "CTA-group-major; observed order inside first group"
    )
    require(
        ordering
        in {
            "CTA-group-major; observed order inside first group",
            "instance-major",
        },
        "only instance-major fixed-stride parameterization is supported",
    )
    return count, strides

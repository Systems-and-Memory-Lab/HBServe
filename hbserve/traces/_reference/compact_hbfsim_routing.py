#!/usr/bin/env python3
"""Placement routing shared by compact HBFSim replay frontends.

Only objects routed to HBF receive device-local logical addresses. They are
packed at page granularity so holes occupied by HBM-resident objects in the
source GPU virtual-address space never inflate HBF's initial population.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from hbserve.traces._reference.compact_request_template import require
from hbserve.traces._reference.replay_compact_hbfsim import POLICIES, target_for_kind


ROUTING_SCHEMA = {
    "name": "hbfsim.compact_ordered_stream_routing",
    "version": 1,
}

LAYOUT_POLICIES = ("source-address", "semantic-key")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _semantic_key(obj: dict[str, Any]) -> str:
    value = obj.get("semantic_key")
    require(
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value),
        f"object {obj.get('object_id')!r} lacks a stable semantic key",
    )
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def build_routing(
    *, objects: list[dict[str, Any]], policy: str, hbf_page_bytes: int,
    layout_policy: str = "source-address",
    explicit_object_layout: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build target choices and target-local object address spaces.

    ``source-address`` preserves the historical behavior: HBM objects retain
    plan addresses and HBF objects are packed in source-address order.
    ``semantic-key`` packs both target namespaces by a stable semantic key so
    independently captured plans receive the same placement without deriving
    a device color from either run's GPU virtual addresses.
    """

    require(policy in POLICIES, "unsupported routing policy")
    require(
        isinstance(hbf_page_bytes, int)
        and not isinstance(hbf_page_bytes, bool)
        and hbf_page_bytes > 0,
        "HBF page bytes must be a positive integer",
    )
    require(layout_policy in LAYOUT_POLICIES, "unsupported routing layout policy")
    rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    target_by_index: list[str] = [""] * len(objects)
    hbf_base_by_index: list[int | None] = [None] * len(objects)
    target_base_by_index: list[int | None] = [None] * len(objects)

    if explicit_object_layout is not None:
        object_ids = [str(obj["object_id"]) for obj in objects]
        require(len(set(object_ids)) == len(object_ids), "object ids must be unique")
        require(
            set(explicit_object_layout) == set(object_ids),
            "explicit routing layout must cover every object exactly once",
        )
        occupied: dict[str, list[tuple[int, int, str]]] = {
            "HBM": [], "HBF_LOGICAL": []
        }
        mapping_rows = []
        for index, obj in enumerate(objects):
            object_id = str(obj["object_id"])
            placement = explicit_object_layout[object_id]
            require(isinstance(placement, Mapping),
                    f"explicit placement for {object_id} is malformed")
            target = target_for_kind(str(obj["kind"]), policy)
            require(placement.get("target") == target,
                    f"explicit placement target changed for {object_id}")
            target_begin = placement.get("base")
            capacity = placement.get("capacity_bytes")
            require(
                isinstance(target_begin, int)
                and not isinstance(target_begin, bool)
                and target_begin >= 0
                and target_begin % hbf_page_bytes == 0,
                f"explicit placement base is invalid for {object_id}",
            )
            require(
                isinstance(capacity, int)
                and not isinstance(capacity, bool)
                and capacity >= int(obj["bytes"]),
                f"explicit placement capacity is too small for {object_id}",
            )
            target_by_index[index] = target
            target_base_by_index[index] = target_begin
            if target == "HBF_LOGICAL":
                hbf_base_by_index[index] = target_begin
            occupied[target].append(
                (target_begin, target_begin + capacity, object_id)
            )
            stable_row = {
                "object_id": object_id,
                "kind": str(obj["kind"]),
                "target": target,
                "target_logical_address": target_begin,
                "capacity_bytes": capacity,
            }
            mapping_rows.append(stable_row)
            row = {
                "object_index": index,
                **stable_row,
                "semantic_key": obj.get("semantic_key"),
                "source_logical_address": int(obj["logical_address"]),
                "bytes": int(obj["bytes"]),
            }
            all_rows.append(row)
            if target == "HBF_LOGICAL":
                rows.append(
                    {
                        "object_index": index,
                        "object_id": object_id,
                        "kind": str(obj["kind"]),
                        "source_logical_address": int(obj["logical_address"]),
                        "target_hbf_logical_address": target_begin,
                        "bytes": int(obj["bytes"]),
                        "capacity_bytes": capacity,
                    }
                )
        for target, extents in occupied.items():
            extents.sort()
            for left, right in zip(extents, extents[1:]):
                require(left[1] <= right[0],
                        f"explicit {target} slots overlap: {left[2]} and {right[2]}")
        hbf_end = max((end for begin, end, _name in occupied["HBF_LOGICAL"]),
                      default=0)
        hbf_pages = _align_up(hbf_end, hbf_page_bytes) // hbf_page_bytes
        encoded = json.dumps(
            sorted(mapping_rows, key=lambda row: row["object_id"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return {
            "schema": ROUTING_SCHEMA,
            "policy": policy,
            "layout_policy": "explicit-object-sidecar",
            "hbf_page_bytes": hbf_page_bytes,
            "target_by_index": target_by_index,
            "target_base_by_index": target_base_by_index,
            "hbf_base_by_index": hbf_base_by_index,
            "hbf_objects": rows,
            "target_objects": all_rows,
            "hbf_object_count": len(rows),
            "hbf_object_bytes": sum(int(row["bytes"]) for row in rows),
            "initial_hbf_logical_first_lpn": 0,
            "initial_hbf_logical_pages": hbf_pages,
            "mapping_table_sha256": hashlib.sha256(encoded).hexdigest(),
            "enable_hbm": "HBM" in target_by_index,
            "enable_hbf": "HBF_LOGICAL" in target_by_index,
        }

    indexed = list(enumerate(objects))
    if layout_policy == "semantic-key":
        semantic_keys = [_semantic_key(obj) for obj in objects]
        require(
            len(set(semantic_keys)) == len(semantic_keys),
            "semantic routing keys must be unique",
        )
        ordered = sorted(
            indexed,
            key=lambda item: (
                target_for_kind(str(item[1]["kind"]), policy),
                _semantic_key(item[1]),
            ),
        )
    else:
        ordered = sorted(
            indexed,
            key=lambda item: (
                int(item[1]["logical_address"]), str(item[1]["object_id"])
            ),
        )

    cursors = {"HBM": 0, "HBF_LOGICAL": 0}
    for index, obj in ordered:
        target = target_for_kind(str(obj["kind"]), policy)
        target_by_index[index] = target
        if layout_policy == "source-address" and target == "HBM":
            target_begin = int(obj["logical_address"])
        else:
            target_begin = _align_up(cursors[target], hbf_page_bytes)
            cursors[target] = target_begin + int(obj["bytes"])
        target_base_by_index[index] = target_begin
        row = {
            "object_index": index,
            "object_id": str(obj["object_id"]),
            "kind": str(obj["kind"]),
            "semantic_key": obj.get("semantic_key"),
            "source_logical_address": int(obj["logical_address"]),
            "target": target,
            "target_logical_address": target_begin,
            "bytes": int(obj["bytes"]),
        }
        all_rows.append(row)
        if target == "HBF_LOGICAL":
            hbf_base_by_index[index] = target_begin
            rows.append(
                {
                    "object_index": index,
                    "object_id": str(obj["object_id"]),
                    "kind": str(obj["kind"]),
                    "source_logical_address": int(obj["logical_address"]),
                    "target_hbf_logical_address": target_begin,
                    "bytes": int(obj["bytes"]),
                }
            )
    require(all(target_by_index), "routing did not classify every plan object")
    require(all(value is not None for value in target_base_by_index),
            "routing did not place every plan object")
    hbf_cursor = cursors["HBF_LOGICAL"]
    hbf_pages = (
        _align_up(hbf_cursor, hbf_page_bytes) // hbf_page_bytes if rows else 0
    )
    digest_rows = rows if layout_policy == "source-address" else all_rows
    encoded = json.dumps(
        digest_rows, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return {
        "schema": ROUTING_SCHEMA,
        "policy": policy,
        "layout_policy": layout_policy,
        "hbf_page_bytes": hbf_page_bytes,
        "target_by_index": target_by_index,
        "target_base_by_index": target_base_by_index,
        "hbf_base_by_index": hbf_base_by_index,
        "hbf_objects": rows,
        "target_objects": all_rows,
        "hbf_object_count": len(rows),
        "hbf_object_bytes": sum(int(row["bytes"]) for row in rows),
        "initial_hbf_logical_first_lpn": 0,
        "initial_hbf_logical_pages": hbf_pages,
        "mapping_table_sha256": hashlib.sha256(encoded).hexdigest(),
        "enable_hbm": "HBM" in target_by_index,
        "enable_hbf": "HBF_LOGICAL" in target_by_index,
    }

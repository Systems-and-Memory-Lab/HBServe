#!/usr/bin/env python3
"""Normalize legacy inference traces into a placement-neutral address contract.

The contract's *target* boundary is an issued GPU global-memory load/store
lane address/range.  Existing Hugging Face and SGLang captures do not reach
that boundary: they contain semantic tensor extents and runtime paged-KV slot
extents, respectively.  The adapter keeps those records useful without
silently promoting them to instruction-observed accesses.

Every field that can affect replay carries an ``observed``, ``derived``, or
``hypothetical`` evidence label.  Placement is deliberately absent and must
be added by a later routing step.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import lzma
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, TextIO

try:
    import orjson as _orjson
except ImportError:  # The artifact remains usable with the Python standard library.
    _orjson = None


SCHEMA_NAME = "hbfsim.gpu_address_trace"
SCHEMA_VERSION = 1
TARGET_BOUNDARY = "gpu_global_memory_issued_load_store"
HF_BOUNDARY = "semantic_tensor_extent"
SGLANG_BOUNDARY = "runtime_paged_kv_slot_extent"
ALLOWED_BOUNDARIES = {TARGET_BOUNDARY, HF_BOUNDARY, SGLANG_BOUNDARY}
EVIDENCE_LEVELS = {"observed", "derived", "hypothetical"}
PAGE_BYTES = 4096
SOURCE_KINDS = {"hf-canonical-v1", "sglang-paged-canonical-v1"}
PLACEMENT_KEYS = {
    "target",
    "placement",
    "route",
    "routing",
    "tier",
    "policy",
}


class TraceContractError(ValueError):
    """Raised when an input or normalized trace violates the contract."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TraceContractError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_jsonl_text(path: Path) -> TextIO:
    """Open a canonical JSONL trace without hiding its compression boundary."""
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="strict")
    return path.open("rt", encoding="utf-8", errors="strict")


def open_jsonl_binary(path: Path) -> BinaryIO:
    """Open JSONL as bytes so an available SIMD parser avoids text copies."""
    if path.suffix == ".xz":
        return lzma.open(path, "rb")
    return path.open("rb")


def json_loads_object(raw: bytes, *, path: Path, line_number: int) -> dict[str, Any]:
    """Parse one JSON object with identical fail-closed behavior.

    ``orjson`` is an optional acceleration for multi-billion-byte traces.  It
    changes only the parser implementation; the accepted top-level record
    type and all downstream schema checks remain the same.
    """
    try:
        item = _orjson.loads(raw) if _orjson is not None else json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TraceContractError(f"{path}:{line_number}: invalid JSON") from error
    require(
        isinstance(item, dict),
        f"{path}:{line_number}: every record must be an object",
    )
    return item


def iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from plain or XZ JSONL without retaining trace events."""
    with open_jsonl_binary(path) as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            yield json_loads_object(raw, path=path, line_number=line_number)


def align_up(value: int, alignment: int = PAGE_BYTES) -> int:
    return (value + alignment - 1) // alignment * alignment


def positive_int(value: Any, label: str) -> int:
    require(
        not isinstance(value, bool) and isinstance(value, int) and value > 0,
        f"{label} must be a positive integer",
    )
    return int(value)


def nonnegative_number(value: Any, label: str) -> int | float:
    require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value >= 0,
        f"{label} must be a nonnegative number",
    )
    return value


def read_jsonl(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    objects: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    phase = "header"
    for item in iter_jsonl_records(path):
        record_type = item.get("record_type")
        if record_type == "header":
            require(header is None and phase == "header", "header must be first")
            header = item
            phase = "objects"
        elif record_type == "object":
            require(
                header is not None and phase == "objects",
                "objects must precede events",
            )
            objects.append(item)
        elif record_type == "event":
            require(header is not None, "events cannot precede the header")
            phase = "events"
            events.append(item)
        else:
            raise TraceContractError(f"{path}: unknown record_type {record_type!r}")
    require(header is not None, f"{path}: missing header")
    require(objects, f"{path}: missing objects")
    require(events, f"{path}: missing events")
    return header, objects, events


def detect_source_kind(
    header: dict[str, Any], events: Iterable[dict[str, Any]]
) -> str:
    schema = header.get("schema")
    require(
        schema
        == {"name": "hbfsim.hardware_validated_canonical_trace", "version": 1},
        "only hbfsim.hardware_validated_canonical_trace v1 is supported",
    )
    memory_events = [event for event in events if event.get("op") in {"R", "W"}]
    require(memory_events, "source has no memory events")
    if any("slot_start" in event or "slot_count" in event for event in memory_events):
        require(
            all(event.get("kind") == "kv_cache" for event in memory_events),
            "SGLang paged input unexpectedly contains non-KV memory events",
        )
        return "sglang-paged-canonical-v1"
    if all("gpu_virtual_address" in event for event in memory_events):
        return "hf-canonical-v1"
    raise TraceContractError(
        "cannot distinguish HF semantic events from SGLang paged-KV events"
    )


def source_object_bytes(item: dict[str, Any]) -> int:
    candidates = [
        item[key] for key in ("bytes", "logical_bytes") if item.get(key) is not None
    ]
    require(candidates, f"object {item.get('object_id')!r} has no byte extent")
    values = [positive_int(value, f"object {item.get('object_id')} bytes") for value in candidates]
    require(
        len(set(values)) == 1,
        f"object {item.get('object_id')!r} disagrees on bytes/logical_bytes",
    )
    return values[0]


def observed_raw_extents(item: dict[str, Any], object_bytes: int) -> list[dict[str, Any]]:
    raw: list[tuple[str, int, int]] = []
    if item.get("gpu_virtual_address") is not None:
        raw.append(
            (
                "live",
                positive_int(
                    item["gpu_virtual_address"],
                    f"object {item.get('object_id')} GPU virtual address",
                ),
                object_bytes,
            )
        )
    for role in ("before", "after"):
        descriptor = item.get(role)
        if descriptor is None:
            continue
        require(isinstance(descriptor, dict), f"object {item.get('object_id')} {role} is invalid")
        raw.append(
            (
                role,
                positive_int(
                    descriptor.get("gpu_virtual_address"),
                    f"object {item.get('object_id')} {role} GPU virtual address",
                ),
                positive_int(
                    descriptor.get("bytes"),
                    f"object {item.get('object_id')} {role} bytes",
                ),
            )
        )
    require(raw, f"object {item.get('object_id')!r} has no observed GPU VA")

    grouped: dict[tuple[int, int], list[str]] = {}
    for role, address, byte_count in raw:
        grouped.setdefault((address, byte_count), []).append(role)
    extents = []
    for ordinal, ((address, byte_count), roles) in enumerate(
        sorted(grouped.items(), key=lambda value: (value[0][0], value[0][1]))
    ):
        extents.append(
            {
                "extent_id": f"raw{ordinal:02d}",
                "gpu_virtual_address": address,
                "bytes": byte_count,
                "roles": sorted(roles),
                "field_evidence": {
                    "gpu_virtual_address": {
                        "level": "observed",
                        "source": "live tensor/pool data pointer",
                    },
                    "bytes": {
                        "level": "derived",
                        "source": "live tensor shape, stride, dtype, or reported nbytes",
                    },
                },
            }
        )
    return extents


def mapping_rows(objects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "object_id": item["object_id"],
            "logical_address": item["logical_address"],
            "bytes": item["bytes"],
            "raw_gpu_va_extents": item["raw_gpu_va_extents"],
        }
        for item in objects
    ]


def normalize_objects(
    source_objects: list[dict[str, Any]], source_kind: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    del source_kind
    by_id: dict[str, dict[str, Any]] = {}
    for source in source_objects:
        object_id = source.get("object_id")
        require(
            isinstance(object_id, str) and object_id and object_id not in by_id,
            f"duplicate or malformed object id {object_id!r}",
        )
        by_id[object_id] = source

    logical_cursor = 0
    normalized: list[dict[str, Any]] = []
    normalized_by_id: dict[str, dict[str, Any]] = {}
    for object_id in sorted(by_id):
        source = by_id[object_id]
        byte_count = source_object_bytes(source)
        logical_cursor = align_up(logical_cursor)
        raw_extents = observed_raw_extents(source, byte_count)
        metadata = {
            key: source[key]
            for key in (
                "name",
                "aliases",
                "layer",
                "component",
                "side",
                "shape",
                "stride",
                "dtype",
                "contiguous",
                "capacity_tokens",
                "active_tokens_before",
                "active_tokens_after",
                "allocator_page_size_tokens",
                "bytes_per_token",
            )
            if key in source
        }
        item = {
            "record_type": "object",
            "object_id": object_id,
            "kind": source.get("kind"),
            "bytes": byte_count,
            "logical_address": logical_cursor,
            "raw_gpu_va_extents": raw_extents,
            "source_logical_address": source.get("logical_address"),
            "metadata": metadata,
            "field_evidence": {
                "kind": {
                    "level": "derived",
                    "source": "source runtime object classification",
                },
                "bytes": {
                    "level": "derived",
                    "source": "source live tensor/pool descriptor",
                },
                "logical_address": {
                    "level": "derived",
                    "source": "object-id-sorted 4-KiB-aligned normalization",
                },
                "raw_gpu_va_extents": {
                    "level": "observed",
                    "source": "source live allocation descriptors",
                },
            },
            "source_object_sha256": sha256_value(source),
        }
        normalized.append(item)
        normalized_by_id[object_id] = item
        logical_cursor += align_up(byte_count)
    return normalized, normalized_by_id


def field_evidence(level: str, source: str) -> dict[str, str]:
    require(level in EVIDENCE_LEVELS, f"unknown evidence level {level}")
    return {"level": level, "source": source}


def matching_extent(
    obj: dict[str, Any], raw_address: int, byte_count: int, object_offset: int
) -> str:
    matches = []
    for extent in obj["raw_gpu_va_extents"]:
        base = extent["gpu_virtual_address"]
        if (
            raw_address == base + object_offset
            and object_offset + byte_count <= extent["bytes"]
        ):
            matches.append(extent["extent_id"])
    require(
        matches,
        f"event range at GPU VA {raw_address} does not match object {obj['object_id']}",
    )
    return sorted(matches)[0]


def normalize_memory_event(
    source: dict[str, Any],
    *,
    sequence_index: int,
    source_kind: str,
    source_objects: dict[str, dict[str, Any]],
    normalized_objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    event_id = source.get("event_id")
    object_id = source.get("object_id")
    require(isinstance(event_id, str) and event_id, "memory event has no event id")
    require(
        isinstance(object_id, str)
        and object_id in source_objects
        and object_id in normalized_objects,
        f"event {event_id} references an unknown object",
    )
    source_obj = source_objects[object_id]
    normalized_obj = normalized_objects[object_id]
    byte_count = positive_int(source.get("bytes"), f"event {event_id} bytes")
    operation = source.get("op")
    require(operation in {"R", "W"}, f"event {event_id} has invalid operation")

    source_logical = source.get("logical_address")
    source_object_logical = source_obj.get("logical_address")
    if source.get("object_offset") is not None:
        object_offset = nonnegative_number(
            source["object_offset"], f"event {event_id} object offset"
        )
        require(isinstance(object_offset, int), f"event {event_id} offset must be integer")
    else:
        require(
            isinstance(source_logical, int) and isinstance(source_object_logical, int),
            f"event {event_id} cannot derive its object offset",
        )
        object_offset = source_logical - source_object_logical
    require(object_offset >= 0, f"event {event_id} has a negative object offset")
    require(
        object_offset + byte_count <= normalized_obj["bytes"],
        f"event {event_id} escapes object {object_id}",
    )
    if isinstance(source_logical, int) and isinstance(source_object_logical, int):
        require(
            source_logical == source_object_logical + object_offset,
            f"event {event_id} source logical address disagrees with object offset",
        )

    if source_kind == "hf-canonical-v1":
        boundary = HF_BOUNDARY
        raw_address = positive_int(
            source.get("gpu_virtual_address"),
            f"event {event_id} GPU virtual address",
        )
        address_source = (
            "semantic object offset added to a same-run live tensor allocation; "
            "not an observed instruction address"
        )
        operation_source = "model-stage semantic access rule"
        byte_source = "semantic live-tensor extent rule"
    else:
        boundary = SGLANG_BOUNDARY
        # The legacy SGLang events contain observed slots but no event GPU VA.
        # Pick the live pool extent and make the derivation explicit.
        extents = normalized_obj["raw_gpu_va_extents"]
        require(len(extents) == 1, f"SGLang object {object_id} has ambiguous raw extents")
        raw_address = extents[0]["gpu_virtual_address"] + object_offset
        address_source = (
            "observed SGLang pool base plus observed allocator slot times "
            "derived bytes-per-token; not an observed instruction address"
        )
        operation_source = "runtime hook role (kv_indices read or out_cache_loc write)"
        byte_source = "observed slot count times derived KV bytes per token"
    raw_extent_id = matching_extent(
        normalized_obj, raw_address, byte_count, object_offset
    )

    metadata = {
        key: source[key]
        for key in (
            "phase",
            "stage",
            "layer",
            "component",
            "side",
            "slot_start",
            "slot_count",
            "timing_semantics",
            "evidence",
        )
        if key in source
    }
    issue_ns = nonnegative_number(source.get("issue_ns", 0), f"event {event_id} issue_ns")
    dependencies = source.get("dependencies", [])
    require(
        isinstance(dependencies, list)
        and all(isinstance(value, str) and value for value in dependencies),
        f"event {event_id} has invalid dependencies",
    )
    return {
        "record_type": "event",
        "sequence_index": sequence_index,
        "event_id": event_id,
        "event_type": "memory_access",
        "kind": source.get("kind"),
        "access_boundary": boundary,
        "op": operation,
        "raw_gpu_virtual_address": raw_address,
        "raw_extent_id": raw_extent_id,
        "logical_address": normalized_obj["logical_address"] + object_offset,
        "bytes": byte_count,
        "object_id": object_id,
        "object_offset": object_offset,
        "issue_ns": issue_ns,
        "duration_ns": 0,
        "dependencies": list(dependencies),
        "metadata": metadata,
        "field_evidence": {
            "access_boundary": field_evidence(
                "derived", "source frontend classification"
            ),
            "op": field_evidence("derived", operation_source),
            "raw_gpu_virtual_address": field_evidence("derived", address_source),
            "logical_address": field_evidence(
                "derived", "normalized object base plus validated object offset"
            ),
            "bytes": field_evidence("derived", byte_source),
            "issue_ns": field_evidence(
                "derived", "memory event placed at its enclosing measured stage start"
            ),
            "dependencies": field_evidence(
                "derived", "source semantic stage dependency DAG"
            ),
            "object_attribution": field_evidence(
                "derived", "validated containment in a same-run live object extent"
            ),
        },
        "source_event_sha256": sha256_value(source),
    }


def normalize_nonmemory_event(
    source: dict[str, Any], *, sequence_index: int, source_kind: str
) -> dict[str, Any]:
    event_id = source.get("event_id")
    require(isinstance(event_id, str) and event_id, "non-memory event has no event id")
    dependencies = source.get("dependencies", [])
    require(
        isinstance(dependencies, list)
        and all(isinstance(value, str) and value for value in dependencies),
        f"event {event_id} has invalid dependencies",
    )
    duration_ns = nonnegative_number(
        source.get("duration_ns", 0), f"event {event_id} duration_ns"
    )
    issue_ns = nonnegative_number(source.get("issue_ns", 0), f"event {event_id} issue_ns")
    event_type = "compute_interval" if source.get("kind") == "compute" else "fence"
    duration_observed = event_type == "compute_interval"
    issue_observed = (
        source_kind == "hf-canonical-v1"
        and source.get("evidence") == "hardware_cuda_event_stage_span"
    )
    metadata = {
        key: source[key]
        for key in ("phase", "stage", "layer", "timing_semantics", "evidence")
        if key in source
    }
    return {
        "record_type": "event",
        "sequence_index": sequence_index,
        "event_id": event_id,
        "event_type": event_type,
        "kind": source.get("kind"),
        "issue_ns": issue_ns,
        "duration_ns": duration_ns,
        "dependencies": list(dependencies),
        "metadata": metadata,
        "field_evidence": {
            "issue_ns": field_evidence(
                "observed" if issue_observed else "derived",
                (
                    "CUDA-event stage start relative to the captured origin"
                    if issue_observed
                    else "source stage ordering or accumulated measured durations"
                ),
            ),
            "duration_ns": field_evidence(
                "observed" if duration_observed else "derived",
                (
                    "CUDA-event measured stage/attention envelope"
                    if duration_observed
                    else "zero-duration semantic fence"
                ),
            ),
            "dependencies": field_evidence(
                "derived", "source semantic stage dependency DAG"
            ),
        },
        "source_event_sha256": sha256_value(source),
    }


def boundary_coverage(boundaries: set[str]) -> str:
    if boundaries == {TARGET_BOUNDARY}:
        return "complete"
    if TARGET_BOUNDARY in boundaries:
        return "partial"
    return "none"


def contract_boundary_coverage(
    boundaries: set[str], contract: dict[str, Any]
) -> str:
    """Combine emitted-event boundaries with explicitly omitted source ops."""
    emitted = boundary_coverage(boundaries)
    source_coverage = contract.get("source_operation_coverage")
    if source_coverage is None:
        return emitted
    require(isinstance(source_coverage, dict), "invalid source operation coverage")
    status = source_coverage.get("status")
    require(status in {"complete", "partial"}, "invalid source operation coverage status")
    if status == "partial" and TARGET_BOUNDARY in boundaries:
        return "partial"
    return emitted


def normalize(
    source_path: Path, *, source_kind: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_header, source_objects_list, source_events = read_jsonl(source_path)
    detected_kind = detect_source_kind(source_header, source_events)
    if source_kind is not None:
        require(source_kind in SOURCE_KINDS, f"unknown source kind {source_kind}")
        require(
            source_kind == detected_kind,
            f"requested source kind {source_kind} disagrees with detected {detected_kind}",
        )
    else:
        source_kind = detected_kind

    normalized_objects, normalized_by_id = normalize_objects(
        source_objects_list, source_kind
    )
    source_objects = {item["object_id"]: item for item in source_objects_list}
    normalized_events = []
    for sequence_index, source in enumerate(source_events):
        if source.get("op") in {"R", "W"}:
            event = normalize_memory_event(
                source,
                sequence_index=sequence_index,
                source_kind=source_kind,
                source_objects=source_objects,
                normalized_objects=normalized_by_id,
            )
        else:
            require(
                source.get("op") is None,
                f"event {source.get('event_id')} has an unsupported operation",
            )
            event = normalize_nonmemory_event(
                source, sequence_index=sequence_index, source_kind=source_kind
            )
        normalized_events.append(event)

    boundaries = {
        event["access_boundary"]
        for event in normalized_events
        if event["event_type"] == "memory_access"
    }
    map_rows = mapping_rows(normalized_objects)
    mapping_sha256 = sha256_value(map_rows)
    source_sha256 = sha256_file(source_path)
    header = {
        "record_type": "header",
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "classification": "placement-neutral address-trace proxy",
        "trace_contract": {
            "target_boundary": {
                "name": TARGET_BOUNDARY,
                "definition": (
                    "executed GPU global-memory load/store lane virtual address "
                    "and access width before L1/L2 filtering"
                ),
                "excludes": [
                    "shared/local memory",
                    "GPU L2 miss requests",
                    "memory-controller transactions",
                    "physical HBM or HBF addresses",
                ],
            },
            "actual_memory_event_boundaries": sorted(boundaries),
            "target_boundary_coverage": boundary_coverage(boundaries),
            "placement_neutral": True,
            "evidence_levels": ["observed", "derived", "hypothetical"],
        },
        "address_contract": {
            "raw_address_space": "same-run process-local GPU virtual address",
            "logical_address_space": "portable object-offset-preserving virtual address",
            "logical_mapping": {
                "scheme": "object-id-sorted 4-KiB-aligned packing",
                "alignment_bytes": PAGE_BYTES,
                "mapping_table_sha256": mapping_sha256,
                "population_bytes": (
                    0
                    if not normalized_objects
                    else align_up(
                        normalized_objects[-1]["logical_address"]
                        + normalized_objects[-1]["bytes"]
                    )
                ),
            },
        },
        "source": {
            "kind": source_kind,
            "trace": str(source_path.resolve()),
            "trace_sha256": source_sha256,
            "schema": source_header.get("schema"),
            "classification": source_header.get("classification"),
        },
        "not_claimed": [
            "complete instruction-observed load/store trace",
            "GPU L2 miss trace",
            "memory-controller request trace",
            "direct HBF hardware trace",
            "HBM/HBF placement",
        ],
    }
    records = [header] + normalized_objects + normalized_events
    summary = validate_records(records)
    manifest = {
        "schema": {"name": "hbfsim.gpu_address_trace_manifest", "version": 1},
        "status": "pass",
        "source_kind": source_kind,
        "source_trace": str(source_path.resolve()),
        "source_trace_sha256": source_sha256,
        "mapping_table_sha256": mapping_sha256,
        **summary,
    }
    return records, manifest


def find_placement_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key.lower() in PLACEMENT_KEYS:
                found.append(child_path)
            found.extend(find_placement_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_placement_keys(child, f"{path}[{index}]"))
    return found


def validate_evidence(
    record_id: str, evidence: Any, required_fields: Iterable[str]
) -> None:
    require(isinstance(evidence, dict), f"record {record_id} has no field evidence")
    for field in required_fields:
        item = evidence.get(field)
        require(isinstance(item, dict), f"record {record_id} lacks evidence for {field}")
        require(
            item.get("level") in EVIDENCE_LEVELS,
            f"record {record_id} has invalid evidence level for {field}",
        )
        require(
            isinstance(item.get("source"), str) and item["source"],
            f"record {record_id} has no evidence source for {field}",
        )


def require_sha256(value: Any, label: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA-256 digest",
    )


def validate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Validate a trace in one pass.

    The iterable form matters for instruction traces: a bounded layer can
    already contain millions of lane events.  Objects are retained, while
    events are checked and discarded as they arrive.
    """
    iterator = iter(records)
    try:
        header = next(iterator)
    except StopIteration as error:
        raise TraceContractError("header must be first") from error
    require(header.get("record_type") == "header", "header must be first")
    require(
        header.get("schema") == {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "wrong normalized trace schema",
    )
    contract = header.get("trace_contract")
    require(isinstance(contract, dict), "header has no trace contract")
    require(contract.get("placement_neutral") is True, "trace is not placement-neutral")
    require(
        (contract.get("target_boundary") or {}).get("name") == TARGET_BOUNDARY,
        "wrong target boundary",
    )
    placement_keys = find_placement_keys(header)
    require(not placement_keys, f"placement fields are forbidden: {placement_keys[:3]}")

    objects: dict[str, dict[str, Any]] = {}
    saw_event = False
    mapping_checked = False
    previous_logical_end = 0
    seen_events: set[str] = set()
    boundaries: Counter[str] = Counter()
    evidence_levels: Counter[str] = Counter()
    memory_bytes = 0
    memory_events = 0
    event_count = 0
    sequential_lane_ids = (
        contract.get("event_id_scheme") == "lane_{sequence_index:012d}"
    )

    def check_mapping() -> None:
        nonlocal mapping_checked
        require(objects, "normalized trace has no objects")
        expected_mapping_sha256 = sha256_value(mapping_rows(objects.values()))
        mapping = (header.get("address_contract") or {}).get("logical_mapping") or {}
        require(
            mapping.get("mapping_table_sha256") == expected_mapping_sha256,
            "logical mapping hash mismatch",
        )
        require(
            mapping.get("population_bytes") == align_up(previous_logical_end),
            "logical population size mismatch",
        )
        mapping_checked = True

    for record in iterator:
        placement_keys = find_placement_keys(record)
        require(
            not placement_keys,
            f"placement fields are forbidden: {placement_keys[:3]}",
        )
        if record.get("record_type") == "object":
            require(not saw_event, "objects must precede events")
            object_id = record.get("object_id")
            require(
                isinstance(object_id, str) and object_id and object_id not in objects,
                f"duplicate or malformed object {object_id!r}",
            )
            address = record.get("logical_address")
            byte_count = record.get("bytes")
            require(
                isinstance(address, int)
                and not isinstance(address, bool)
                and address >= 0
                and address % PAGE_BYTES == 0,
                f"object {object_id} has invalid normalized address",
            )
            positive_int(byte_count, f"object {object_id} bytes")
            require(address >= previous_logical_end, f"object {object_id} overlaps its predecessor")
            previous_logical_end = address + byte_count
            extents = record.get("raw_gpu_va_extents")
            require(isinstance(extents, list) and extents, f"object {object_id} has no raw extents")
            validate_evidence(
                object_id,
                record.get("field_evidence"),
                ("kind", "bytes", "logical_address", "raw_gpu_va_extents"),
            )
            require_sha256(
                record.get("source_object_sha256"),
                f"object {object_id} source hash",
            )
            extent_ids = set()
            for extent in extents:
                extent_id = extent.get("extent_id")
                require(
                    isinstance(extent_id, str) and extent_id not in extent_ids,
                    f"object {object_id} has duplicate raw extent ids",
                )
                extent_ids.add(extent_id)
                positive_int(extent.get("gpu_virtual_address"), f"object {object_id} raw address")
                positive_int(extent.get("bytes"), f"object {object_id} raw bytes")
                validate_evidence(
                    f"{object_id}/{extent_id}",
                    extent.get("field_evidence"),
                    ("gpu_virtual_address", "bytes"),
                )
            objects[object_id] = record
        elif record.get("record_type") == "event":
            if not saw_event:
                check_mapping()
                saw_event = True
            event = record
            event_id = event.get("event_id")
            require(
                isinstance(event_id, str) and event_id,
                f"malformed event {event_id!r}",
            )
            if sequential_lane_ids:
                require(
                    event_id == f"lane_{event_count:012d}",
                    f"event {event_id!r} violates the sequential lane id scheme",
                )
            else:
                require(
                    event_id not in seen_events,
                    f"duplicate event {event_id!r}",
                )
            require(
                event.get("sequence_index") == event_count,
                f"event {event_id} has a noncontiguous sequence index",
            )
            require_sha256(
                event.get("source_event_sha256"), f"event {event_id} source hash"
            )
            dependencies = event.get("dependencies")
            require(
                isinstance(dependencies, list)
                and all(isinstance(value, str) and value for value in dependencies),
                f"event {event_id} has invalid dependencies",
            )
            if sequential_lane_ids:
                require(
                    not dependencies,
                    f"raw lane event {event_id} unexpectedly encodes dependencies",
                )
            else:
                missing = [
                    dependency
                    for dependency in dependencies
                    if dependency not in seen_events
                ]
                require(
                    not missing,
                    f"event {event_id} has forward/missing dependencies {missing}",
                )
            nonnegative_number(
                event.get("duration_ns"), f"event {event_id} duration_ns"
            )
            if event.get("event_type") == "memory_access":
                memory_events += 1
                boundary = event.get("access_boundary")
                require(
                    boundary in ALLOWED_BOUNDARIES,
                    f"event {event_id} has invalid boundary",
                )
                boundaries[boundary] += 1
                require(
                    event.get("op") in {"R", "W"},
                    f"event {event_id} has invalid op",
                )
                byte_count = positive_int(
                    event.get("bytes"), f"event {event_id} bytes"
                )
                memory_bytes += byte_count
                object_id = event.get("object_id")
                require(
                    object_id in objects,
                    f"event {event_id} has an unknown object",
                )
                obj = objects[object_id]
                offset = event.get("object_offset")
                require(
                    isinstance(offset, int)
                    and not isinstance(offset, bool)
                    and offset >= 0
                    and offset + byte_count <= obj["bytes"],
                    f"event {event_id} escapes its object",
                )
                require(
                    event.get("logical_address") == obj["logical_address"] + offset,
                    f"event {event_id} has an inconsistent logical address",
                )
                raw_address = event.get("raw_gpu_virtual_address")
                positive_int(raw_address, f"event {event_id} raw GPU VA")
                extent = next(
                    (
                        value
                        for value in obj["raw_gpu_va_extents"]
                        if value["extent_id"] == event.get("raw_extent_id")
                    ),
                    None,
                )
                require(
                    extent is not None,
                    f"event {event_id} has an unknown raw extent",
                )
                require(
                    raw_address == extent["gpu_virtual_address"] + offset
                    and offset + byte_count <= extent["bytes"],
                    f"event {event_id} does not map to its declared raw extent",
                )
                required = (
                    "access_boundary",
                    "op",
                    "raw_gpu_virtual_address",
                    "logical_address",
                    "bytes",
                    "issue_ns",
                    "dependencies",
                    "object_attribution",
                )
                validate_evidence(event_id, event.get("field_evidence"), required)
                issue_ns = event.get("issue_ns")
                issue_evidence = event["field_evidence"]["issue_ns"]
                if issue_ns is None:
                    require(
                        boundary == TARGET_BOUNDARY
                        and issue_evidence.get("level") == "derived"
                        and issue_evidence.get("status") == "unknown",
                        f"event {event_id} has an unmarked unknown issue time",
                    )
                else:
                    nonnegative_number(issue_ns, f"event {event_id} issue_ns")
                if boundary == TARGET_BOUNDARY:
                    op_evidence = event["field_evidence"]["op"]
                    metadata = event.get("metadata")
                    mref = (
                        metadata.get("memory_reference_metadata")
                        if isinstance(metadata, dict)
                        else None
                    )
                    opcode = metadata.get("opcode") if isinstance(metadata, dict) else None
                    derived_ldgsts_global_read = (
                        op_evidence.get("level") == "derived"
                        and isinstance(opcode, str)
                        and opcode.upper().startswith("LDGSTS")
                        and isinstance(mref, dict)
                        and mref.get("format") == "MEM_META_V1"
                        and mref.get("instruction_memory_space_name")
                        == "GLOBAL_TO_SHARED"
                        and mref.get("mref_memory_space_name") == "GLOBAL"
                        and mref.get("num_mref") == 2
                        and mref.get("address_mref_index") == 1
                        and mref.get("nvbit_is_load") is True
                        and mref.get("nvbit_is_store") is True
                        and mref.get("effective_mref_operation") == "R"
                        and event.get("op") == "R"
                    )
                    require(
                        op_evidence.get("level") == "observed"
                        or derived_ldgsts_global_read,
                        f"issued load/store event {event_id} has an unsupported "
                        "non-observed op",
                    )
                    for field in ("raw_gpu_virtual_address", "bytes"):
                        require(
                            event["field_evidence"][field]["level"] == "observed",
                            f"issued load/store event {event_id} has non-observed {field}",
                        )
            else:
                require(
                    event.get("event_type") in {"compute_interval", "fence"},
                    f"event {event_id} has invalid event type",
                )
                validate_evidence(
                    event_id,
                    event.get("field_evidence"),
                    ("issue_ns", "duration_ns", "dependencies"),
                )
                nonnegative_number(event.get("issue_ns"), f"event {event_id} issue_ns")
            for evidence in event["field_evidence"].values():
                evidence_levels[evidence["level"]] += 1
            if not sequential_lane_ids:
                seen_events.add(event_id)
            event_count += 1
        else:
            raise TraceContractError(
                f"unknown normalized record type {record.get('record_type')!r}"
            )
    require(objects, "normalized trace has no objects")
    require(event_count, "normalized trace has no events")
    if not mapping_checked:
        check_mapping()

    actual_boundaries = sorted(boundaries)
    require(
        contract.get("actual_memory_event_boundaries") == actual_boundaries,
        "header boundary census disagrees with events",
    )
    expected_coverage = contract_boundary_coverage(set(boundaries), contract)
    require(
        contract.get("target_boundary_coverage") == expected_coverage,
        "header target-boundary coverage disagrees with events",
    )
    return {
        "objects": len(objects),
        "events": event_count,
        "memory_events": memory_events,
        "memory_bytes": memory_bytes,
        "memory_events_by_boundary": dict(sorted(boundaries.items())),
        "field_evidence_levels": dict(sorted(evidence_levels.items())),
        "target_boundary_coverage": expected_coverage,
    }


def validate_trace_path(path: Path) -> dict[str, Any]:
    """Streaming validation for a plain or XZ canonical trace."""
    return validate_records(iter_jsonl_records(path))


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-kind", choices=sorted(SOURCE_KINDS))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an already-normalized hbfsim.gpu_address_trace v1 file.",
    )
    args = parser.parse_args()
    if args.validate_only:
        header, objects, events = read_jsonl(args.input)
        records = [header] + objects + events
        summary = validate_records(records)
        print(json.dumps(summary, sort_keys=True))
        return

    records, manifest = normalize(args.input.resolve(), source_kind=args.source_kind)
    write_records(args.output.resolve(), records)
    manifest["output"] = str(args.output.resolve())
    manifest["output_sha256"] = sha256_file(args.output.resolve())
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

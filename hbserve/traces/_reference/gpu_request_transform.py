#!/usr/bin/env python3
"""Reference issued-lane -> GPU device-request transform.

This module is deliberately a small, evidence-preserving reference path.  It
does not claim to reproduce an NVIDIA cache hierarchy.  It consumes an
``hbfsim.gpu_address_trace`` v1 file whose target-boundary events are grouped
by one executed warp memory instruction, coalesces lane byte ranges into
configurable aligned sectors, optionally filters them through a configurable
set-associative LRU cache, and emits a placement-neutral
``hbfsim.gpu_device_request_trace`` v1 file.

The reference model exists to close and test the trace plumbing.  Architecture
specific coalescing/cache models must remain named, independently calibrated
replacements rather than silently inheriting this model's output.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import lzma
from pathlib import Path
from typing import Any, Iterable

from hbserve.traces._reference.load_store_trace_contract import (
    TARGET_BOUNDARY,
    TraceContractError,
    validate_records as validate_address_records,
)


ADDRESS_SCHEMA = {"name": "hbfsim.gpu_address_trace", "version": 1}
DEVICE_SCHEMA = {"name": "hbfsim.gpu_device_request_trace", "version": 1}
DEVICE_BOUNDARY = "gpu_device_request_before_placement"
READ_ADMISSION_POLICIES = ("uniform-mru", "sass-ef-lru")
CACHE_ADDRESS_SPACES = ("logical", "raw-gpu-va")


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stream_context = (
        lzma.open(path, "rt", encoding="utf-8", errors="strict")
        if path.suffix == ".xz"
        else path.open("rt", encoding="utf-8", errors="strict")
    )
    with stream_context as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TraceContractError(f"{path}:{line_number} is not an object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for record in records:
            stream.write(canonical_json(record) + b"\n")


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TraceContractError(f"{label} must be a positive integer")
    return value


def nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceContractError(f"{label} must be a nonnegative integer")
    return value


def aligned_floor(value: int, alignment: int) -> int:
    return value // alignment * alignment


@dataclass(frozen=True)
class LaneAccess:
    event_id: str
    instruction_group_id: str
    object_id: str
    raw_extent_id: str
    kind: str
    op: str
    lane_id: int
    raw_address: int
    logical_address: int
    object_offset: int
    byte_count: int
    issue_ns: int | float | None
    dependencies: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SectorRequest:
    instruction_group_id: str
    object_id: str
    raw_extent_id: str
    kind: str
    op: str
    raw_address: int
    logical_address: int
    object_offset: int
    byte_count: int
    issue_ns: int | float | None
    dependencies: tuple[str, ...]
    source_lane_event_ids: tuple[str, ...]
    source_lane_bytes: int
    metadata: dict[str, Any]


def _lane_access(event: dict[str, Any]) -> LaneAccess:
    event_id = event.get("event_id")
    metadata = event.get("metadata")
    if not isinstance(event_id, str) or not event_id:
        raise TraceContractError("lane event has no event_id")
    if event.get("event_type") != "memory_access":
        raise TraceContractError(f"event {event_id} is not a memory access")
    if event.get("access_boundary") != TARGET_BOUNDARY:
        raise TraceContractError(
            f"event {event_id} is not at the issued load/store boundary"
        )
    if not isinstance(metadata, dict):
        raise TraceContractError(f"event {event_id} has no metadata")
    group_id = metadata.get("instruction_group_id")
    lane_id = metadata.get("lane_id")
    if not isinstance(group_id, str) or not group_id:
        raise TraceContractError(
            f"event {event_id} has no instruction_group_id metadata"
        )
    lane_id = nonnegative_int(lane_id, f"event {event_id} lane_id")
    if lane_id >= 32:
        raise TraceContractError(f"event {event_id} lane_id is outside a warp")
    dependencies = event.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) and item for item in dependencies
    ):
        raise TraceContractError(f"event {event_id} has invalid dependencies")
    return LaneAccess(
        event_id=event_id,
        instruction_group_id=group_id,
        object_id=str(event["object_id"]),
        raw_extent_id=str(event["raw_extent_id"]),
        kind=str(event.get("kind")),
        op=str(event["op"]),
        lane_id=lane_id,
        raw_address=positive_int(
            event.get("raw_gpu_virtual_address"), f"event {event_id} raw address"
        ),
        logical_address=nonnegative_int(
            event.get("logical_address"), f"event {event_id} logical address"
        ),
        object_offset=nonnegative_int(
            event.get("object_offset"), f"event {event_id} object offset"
        ),
        byte_count=positive_int(event.get("bytes"), f"event {event_id} bytes"),
        issue_ns=event.get("issue_ns", 0),
        dependencies=tuple(dependencies),
        metadata=metadata,
    )


def group_lane_events(events: Iterable[dict[str, Any]]) -> list[list[LaneAccess]]:
    """Group consecutive lane records belonging to one dynamic warp instruction."""
    groups: list[list[LaneAccess]] = []
    current: list[LaneAccess] = []
    completed: set[str] = set()
    for event in events:
        lane = _lane_access(event)
        if current and lane.instruction_group_id != current[0].instruction_group_id:
            completed.add(current[0].instruction_group_id)
            groups.append(current)
            current = []
        if lane.instruction_group_id in completed:
            raise TraceContractError(
                f"instruction group {lane.instruction_group_id} is not contiguous"
            )
        current.append(lane)
    if current:
        groups.append(current)
    for group in groups:
        lane_ids = [lane.lane_id for lane in group]
        if len(lane_ids) != len(set(lane_ids)):
            raise TraceContractError(
                f"instruction group {group[0].instruction_group_id} repeats a lane"
            )
    return groups


def _extent_map(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(extent["extent_id"]): extent
        for extent in obj.get("raw_gpu_va_extents", [])
    }


def coalesce_group(
    group: list[LaneAccess],
    *,
    objects: dict[str, dict[str, Any]],
    sector_bytes: int,
) -> list[SectorRequest]:
    """Union lane byte ranges into aligned sectors, split at object extents.

    This is a named reference transform, not an Ada coalescer.  A sector is
    emitted once for each object/raw-extent/op tuple touched by the dynamic
    warp instruction.  Sector fetches are clipped at allocation boundaries so
    the placement-neutral object mapping remains valid.
    """
    positive_int(sector_bytes, "sector_bytes")
    if sector_bytes & (sector_bytes - 1):
        raise TraceContractError("sector_bytes must be a power of two")
    if not group:
        return []
    reference = group[0]
    for lane in group[1:]:
        if lane.op != reference.op:
            raise TraceContractError(
                f"instruction group {reference.instruction_group_id} mixes operations"
            )
        for key in ("kernel_id", "raw_record_index", "pc"):
            left = reference.metadata.get(key)
            right = lane.metadata.get(key)
            if left is not None and right is not None and left != right:
                raise TraceContractError(
                    f"instruction group {reference.instruction_group_id} mixes {key}"
                )

    partitions: dict[tuple[str, str, str], list[LaneAccess]] = {}
    for lane in group:
        partitions.setdefault((lane.object_id, lane.raw_extent_id, lane.op), []).append(lane)

    requests: list[SectorRequest] = []
    for (object_id, extent_id, operation), lanes in sorted(partitions.items()):
        obj = objects.get(object_id)
        if obj is None:
            raise TraceContractError(f"unknown object {object_id}")
        extent = _extent_map(obj).get(extent_id)
        if extent is None:
            raise TraceContractError(f"unknown extent {object_id}/{extent_id}")
        raw_begin = int(extent["gpu_virtual_address"])
        raw_end = raw_begin + int(extent["bytes"])
        object_bytes = int(obj["bytes"])
        sector_sources: dict[int, list[LaneAccess]] = {}
        source_lane_bytes: Counter[int] = Counter()
        for lane in lanes:
            if lane.raw_address < raw_begin or lane.raw_address + lane.byte_count > raw_end:
                raise TraceContractError(f"lane {lane.event_id} escapes its raw extent")
            first = aligned_floor(lane.raw_address, sector_bytes)
            last = aligned_floor(lane.raw_address + lane.byte_count - 1, sector_bytes)
            sector = first
            while sector <= last:
                sector_sources.setdefault(sector, []).append(lane)
                overlap_begin = max(sector, lane.raw_address)
                overlap_end = min(sector + sector_bytes, lane.raw_address + lane.byte_count)
                source_lane_bytes[sector] += max(0, overlap_end - overlap_begin)
                sector += sector_bytes
        for sector, source_lanes in sorted(sector_sources.items()):
            request_raw_begin = max(sector, raw_begin)
            request_raw_end = min(sector + sector_bytes, raw_end)
            object_offset = request_raw_begin - raw_begin
            byte_count = request_raw_end - request_raw_begin
            if object_offset + byte_count > object_bytes:
                raise TraceContractError(
                    f"coalesced request escapes logical object {object_id}"
                )
            source_ids = tuple(sorted({lane.event_id for lane in source_lanes}))
            dependencies = tuple(
                dict.fromkeys(
                    dependency
                    for lane in source_lanes
                    for dependency in lane.dependencies
                )
            )
            issue_values = {lane.issue_ns for lane in source_lanes}
            if None in issue_values and len(issue_values) != 1:
                raise TraceContractError(
                    f"instruction group {reference.instruction_group_id} mixes "
                    "known and unknown issue times"
                )
            issue_ns = None if issue_values == {None} else min(issue_values)
            requests.append(
                SectorRequest(
                    instruction_group_id=reference.instruction_group_id,
                    object_id=object_id,
                    raw_extent_id=extent_id,
                    kind=str(obj.get("kind")),
                    op=operation,
                    raw_address=request_raw_begin,
                    logical_address=int(obj["logical_address"]) + object_offset,
                    object_offset=object_offset,
                    byte_count=byte_count,
                    issue_ns=issue_ns,
                    dependencies=dependencies,
                    source_lane_event_ids=source_ids,
                    source_lane_bytes=source_lane_bytes[sector],
                    metadata={
                        key: reference.metadata[key]
                        for key in (
                            "kernel_ordinal",
                            "kernel_id",
                            "kernel_name",
                            "raw_record_index",
                            "cta",
                            "warp_in_cta",
                            "pc",
                            "opcode",
                            "active_mask",
                        )
                        if key in reference.metadata
                    },
                )
            )
    return sorted(requests, key=lambda item: (item.raw_address, item.object_id, item.op))


@dataclass
class CacheEntry:
    tag: int
    present_sectors: set[int] = field(default_factory=set)
    dirty_sectors: dict[int, SectorRequest] = field(default_factory=dict)


class ReferenceLRUCache:
    """Small set-associative sector cache used only as a reference transform."""

    def __init__(
        self,
        *,
        capacity_bytes: int,
        line_bytes: int,
        sector_bytes: int,
        associativity: int,
        write_policy: str,
        write_allocate: bool,
        read_admission: str = "uniform-mru",
        address_space: str = "logical",
        write_miss_fetch: bool = True,
    ) -> None:
        positive_int(capacity_bytes, "cache capacity")
        positive_int(line_bytes, "cache line bytes")
        positive_int(sector_bytes, "cache sector bytes")
        positive_int(associativity, "cache associativity")
        if line_bytes % sector_bytes:
            raise TraceContractError("cache line must contain whole sectors")
        lines = capacity_bytes // line_bytes
        if lines < associativity or lines % associativity:
            raise TraceContractError(
                "cache capacity must contain an integral number of sets"
            )
        if write_policy not in {"write-through", "write-back"}:
            raise TraceContractError(f"unsupported write policy {write_policy}")
        if read_admission not in READ_ADMISSION_POLICIES:
            raise TraceContractError(
                f"unsupported read-admission policy {read_admission}"
            )
        if address_space not in CACHE_ADDRESS_SPACES:
            raise TraceContractError(f"unsupported cache address space {address_space}")
        if not isinstance(write_miss_fetch, bool):
            raise TraceContractError("write_miss_fetch must be boolean")
        self.line_bytes = line_bytes
        self.sector_bytes = sector_bytes
        self.associativity = associativity
        self.set_count = lines // associativity
        self.write_policy = write_policy
        self.write_allocate = write_allocate
        self.read_admission = read_admission
        self.address_space = address_space
        self.write_miss_fetch = write_miss_fetch
        self.sets: list[OrderedDict[int, CacheEntry]] = [
            OrderedDict() for _ in range(self.set_count)
        ]
        self.stats: Counter[str] = Counter()

    def _address(self, request: SectorRequest) -> int:
        return (
            request.logical_address
            if self.address_space == "logical"
            else request.raw_address
        )

    def _location(self, request: SectorRequest) -> tuple[int, int, int]:
        address = self._address(request)
        line_number = address // self.line_bytes
        set_index = line_number % self.set_count
        tag = line_number // self.set_count
        sector_index = (address % self.line_bytes) // self.sector_bytes
        return set_index, tag, sector_index

    def _evict_first(self, request: SectorRequest) -> bool:
        opcode_tokens = str(request.metadata.get("opcode", "")).upper().split(".")
        return (
            request.op == "R"
            and self.read_admission == "sass-ef-lru"
            and "EF" in opcode_tokens
        )

    @staticmethod
    def _touch(
        cache_set: OrderedDict[int, CacheEntry], tag: int, *, low_priority: bool
    ) -> None:
        cache_set.move_to_end(tag, last=not low_priority)

    def _insert(self, request: SectorRequest) -> tuple[CacheEntry, list[SectorRequest]]:
        set_index, tag, _ = self._location(request)
        cache_set = self.sets[set_index]
        evictions: list[SectorRequest] = []
        if tag in cache_set:
            entry = cache_set.pop(tag)
            cache_set[tag] = entry
            return entry, evictions
        if len(cache_set) >= self.associativity:
            _, victim = cache_set.popitem(last=False)
            self.stats["line_evictions"] += 1
            if victim.dirty_sectors:
                evictions.extend(victim.dirty_sectors.values())
                self.stats["dirty_sector_writebacks"] += len(victim.dirty_sectors)
        entry = CacheEntry(tag=tag)
        cache_set[tag] = entry
        return entry, evictions

    def access(self, request: SectorRequest) -> list[tuple[str, SectorRequest]]:
        set_index, tag, sector_index = self._location(request)
        cache_set = self.sets[set_index]
        entry = cache_set.get(tag)
        hit = entry is not None and sector_index in entry.present_sectors
        evict_first = self._evict_first(request)
        if evict_first:
            self.stats["sass_evict_first_read_accesses"] += 1
        operation_name = {"R": "read", "W": "write"}.get(request.op)
        if operation_name is None:
            raise TraceContractError(f"unsupported cache operation {request.op}")
        self.stats[f"{operation_name}_{'hits' if hit else 'misses'}"] += 1
        downstream: list[tuple[str, SectorRequest]] = []

        if request.op == "R":
            if hit:
                self._touch(cache_set, tag, low_priority=evict_first)
                return downstream
            entry, evictions = self._insert(request)
            downstream.extend(("dirty-eviction", value) for value in evictions)
            entry.present_sectors.add(sector_index)
            self._touch(cache_set, tag, low_priority=evict_first)
            downstream.append(("read-miss", request))
            return downstream

        if request.op != "W":
            raise TraceContractError(f"unsupported cache operation {request.op}")
        if self.write_policy == "write-through":
            if hit:
                cache_set.move_to_end(tag)
            elif self.write_allocate:
                entry, evictions = self._insert(request)
                downstream.extend(("dirty-eviction", value) for value in evictions)
                entry.present_sectors.add(sector_index)
            downstream.append(("write-through", request))
            return downstream

        # Reference write-back behavior.  A write-allocate miss models the
        # fill as a read before marking the sector dirty.  A no-allocate miss
        # bypasses as a write request.
        if not hit and not self.write_allocate:
            downstream.append(("write-around", request))
            return downstream
        if not hit:
            entry, evictions = self._insert(request)
            downstream.extend(("dirty-eviction", value) for value in evictions)
            if self.write_miss_fetch:
                downstream.append(("write-allocate-fill", request))
            else:
                self.stats["write_miss_allocations_without_fetch"] += 1
        assert entry is not None
        entry.present_sectors.add(sector_index)
        entry.dirty_sectors[sector_index] = request
        cache_set.move_to_end(tag)
        return downstream

    def drain(self) -> list[SectorRequest]:
        writebacks: list[SectorRequest] = []
        for cache_set in self.sets:
            for entry in cache_set.values():
                writebacks.extend(entry.dirty_sectors.values())
                entry.dirty_sectors.clear()
        self.stats["final_dirty_sector_writebacks"] += len(writebacks)
        return writebacks

    def state_summary(self) -> dict[str, Any]:
        resident_lines = 0
        resident_sectors = 0
        resident_dirty_sectors = 0
        for cache_set in self.sets:
            resident_lines += len(cache_set)
            for entry in cache_set.values():
                resident_sectors += len(entry.present_sectors)
                resident_dirty_sectors += len(entry.dirty_sectors)
        return {
            "resident_lines": resident_lines,
            "resident_sector_bytes": resident_sectors * self.sector_bytes,
            "resident_dirty_sector_bytes": (
                resident_dirty_sectors * self.sector_bytes
            ),
        }


def _device_event(
    request: SectorRequest,
    *,
    sequence_index: int,
    reason: str,
    operation: str | None = None,
    provenance_mode: str = "full",
) -> dict[str, Any]:
    if provenance_mode not in {"full", "compact"}:
        raise TraceContractError(f"unsupported provenance mode {provenance_mode}")
    op = operation or request.op
    event_id = f"device_{sequence_index:012d}"
    metadata = {
        **request.metadata,
        "access_boundary": DEVICE_BOUNDARY,
        "transform_reason": reason,
        "source_instruction_group_id": request.instruction_group_id,
        "source_lane_event_count": len(request.source_lane_event_ids),
        "source_lane_event_ids_sha256": sha256_value(
            list(request.source_lane_event_ids)
        ),
        "source_lane_bytes": request.source_lane_bytes,
        "raw_gpu_virtual_address": request.raw_address,
        "raw_extent_id": request.raw_extent_id,
    }
    if provenance_mode == "full":
        metadata["source_lane_event_ids"] = list(request.source_lane_event_ids)
    else:
        metadata["source_lane_event_id_first"] = request.source_lane_event_ids[0]
        metadata["source_lane_event_id_last"] = request.source_lane_event_ids[-1]
    if request.issue_ns is None:
        issue_ns: int | float = sequence_index
        issue_time_source = (
            "device request sequence index used as an ordinal-ns placeholder; "
            "not a measured issue time"
        )
        metadata["issue_time_status"] = "ordinal-placeholder"
    else:
        issue_ns = request.issue_ns
        issue_time_source = "numeric issue time preserved from source trace"
    source = {
        "event_id": event_id,
        "op": op,
        "logical_address": request.logical_address,
        "bytes": request.byte_count,
        "metadata": metadata,
    }
    return {
        "record_type": "event",
        "sequence_index": sequence_index,
        "event_id": event_id,
        "event_type": "memory_access",
        "kind": request.kind,
        "access_boundary": DEVICE_BOUNDARY,
        "op": op,
        "logical_address": request.logical_address,
        "bytes": request.byte_count,
        "object_id": request.object_id,
        "object_offset": request.object_offset,
        "issue_ns": issue_ns,
        "duration_ns": 0,
        # Raw trace dependencies often name lane events that disappear during
        # the transform.  Preserve them as provenance, not executable DAG
        # edges, until a separately validated timing composer maps them.
        "dependencies": [],
        "metadata": metadata,
        "field_evidence": {
            "access_boundary": {
                "level": "derived",
                "source": "named coalescing/cache transform",
            },
            "op": {"level": "derived", "source": "issued lane operation"},
            "logical_address": {
                "level": "derived",
                "source": "raw sector mapped through validated object extent",
            },
            "bytes": {
                "level": "derived",
                "source": "aligned sector intersection with object extent",
            },
            "issue_ns": {
                "level": "derived",
                "source": issue_time_source,
            },
            "dependencies": {
                "level": "derived",
                "source": "empty reference-transform DAG",
            },
        },
        "source_event_sha256": sha256_value(source),
    }


def transform(
    records: list[dict[str, Any]],
    *,
    source_path: Path,
    sector_bytes: int,
    cache_mode: str,
    cache_capacity_bytes: int,
    cache_line_bytes: int,
    cache_associativity: int,
    write_policy: str,
    write_allocate: bool,
    final_drain: bool,
    read_admission: str = "uniform-mru",
    cache_address_space: str = "logical",
    write_miss_fetch: bool = True,
    warmup_kernel_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nonnegative_int(warmup_kernel_count, "warmup kernel count")
    address_summary = validate_address_records(records)
    if address_summary["target_boundary_coverage"] != "complete":
        raise TraceContractError("transform requires complete target-boundary coverage")
    header = records[0]
    if header.get("schema") != ADDRESS_SCHEMA:
        raise TraceContractError("transform requires gpu_address_trace v1")
    objects_list = [item for item in records if item.get("record_type") == "object"]
    objects = {str(item["object_id"]): item for item in objects_list}
    lane_events = [
        item
        for item in records
        if item.get("record_type") == "event"
        and item.get("event_type") == "memory_access"
    ]
    groups = group_lane_events(lane_events)
    coalesced: list[SectorRequest] = []
    for group in groups:
        coalesced.extend(
            coalesce_group(group, objects=objects, sector_bytes=sector_bytes)
        )

    cache: ReferenceLRUCache | None = None
    if cache_mode == "reference-lru":
        cache = ReferenceLRUCache(
            capacity_bytes=cache_capacity_bytes,
            line_bytes=cache_line_bytes,
            sector_bytes=sector_bytes,
            associativity=cache_associativity,
            write_policy=write_policy,
            write_allocate=write_allocate,
            read_admission=read_admission,
            address_space=cache_address_space,
            write_miss_fetch=write_miss_fetch,
        )
    elif cache_mode != "bypass":
        raise TraceContractError(f"unsupported cache mode {cache_mode}")
    if warmup_kernel_count and cache is None:
        raise TraceContractError("warmup kernels require an enabled cache")

    event_specs: list[tuple[SectorRequest, str, str | None]] = []
    warmup_counts: Counter[str] = Counter()
    measurement_entry_cache_state: dict[str, Any] | None = None
    if cache is None:
        event_specs.extend((request, "cache-bypass", None) for request in coalesced)
    else:
        for request in coalesced:
            kernel_ordinal = request.metadata.get("kernel_ordinal")
            if warmup_kernel_count:
                if not isinstance(kernel_ordinal, int) or isinstance(
                    kernel_ordinal, bool
                ):
                    raise TraceContractError(
                        "warmup requires integer kernel_ordinal metadata"
                    )
            is_warmup = (
                isinstance(kernel_ordinal, int)
                and kernel_ordinal < warmup_kernel_count
            )
            if not is_warmup and measurement_entry_cache_state is None:
                measurement_entry_cache_state = cache.state_summary()
            for reason, downstream in cache.access(request):
                operation = "W" if reason == "dirty-eviction" else None
                if reason == "write-allocate-fill":
                    operation = "R"
                if is_warmup:
                    op = operation or downstream.op
                    warmup_counts["device_requests_suppressed"] += 1
                    warmup_counts["device_bytes_suppressed"] += downstream.byte_count
                    warmup_counts[f"device_{op.lower()}_bytes_suppressed"] += (
                        downstream.byte_count
                    )
                else:
                    event_specs.append((downstream, reason, operation))
        if final_drain:
            event_specs.extend(
                (request, "final-dirty-drain", "W") for request in cache.drain()
            )

    output_events = [
        _device_event(request, sequence_index=index, reason=reason, operation=operation)
        for index, (request, reason, operation) in enumerate(event_specs)
    ]
    source_lane_bytes = sum(int(event["bytes"]) for event in lane_events)
    coalesced_bytes = sum(request.byte_count for request in coalesced)
    source_sha256 = sha256_file(source_path)
    transform_config = {
        "coalescer": "aligned-sector-union-reference-v1",
        "sector_bytes": sector_bytes,
        "cache_mode": cache_mode,
        "cache_capacity_bytes": cache_capacity_bytes if cache is not None else 0,
        "cache_line_bytes": cache_line_bytes if cache is not None else 0,
        "cache_associativity": cache_associativity if cache is not None else 0,
        "write_policy": write_policy if cache is not None else "none",
        "write_allocate": write_allocate if cache is not None else False,
        "write_miss_fetch": write_miss_fetch if cache is not None else False,
        "read_admission": read_admission if cache is not None else "none",
        "cache_address_space": cache_address_space if cache is not None else "none",
        "warmup_kernel_count": warmup_kernel_count if cache is not None else 0,
        "warmup_output_policy": (
            "update-cache-and-suppress-device-events"
            if warmup_kernel_count
            else "none"
        ),
        "final_drain": final_drain if cache is not None else False,
        "unknown_issue_time_policy": (
            "device-request sequence index as ordinal-ns placeholder"
        ),
    }
    output_header = {
        "record_type": "header",
        "schema": DEVICE_SCHEMA,
        "classification": "derived reference GPU device-request trace",
        "trace_contract": {
            "actual_memory_event_boundary": DEVICE_BOUNDARY,
            "placement_neutral": True,
            "source_target_boundary": TARGET_BOUNDARY,
        },
        "source": {
            "trace": str(source_path.resolve()),
            "trace_sha256": source_sha256,
            "schema": header.get("schema"),
        },
        "transform": transform_config,
        "not_claimed": [
            "architecture-accurate NVIDIA coalescing",
            "complete or cycle-accurate Ada L1/L2 behavior",
            "production GPU issue timing",
            "physical HBM/HBF address",
            "real HBF device behavior",
        ],
    }
    output_records = [output_header, *objects_list, *output_events]
    summary = validate_device_records(output_records)
    summary.update(
        {
            "source_lane_events": len(lane_events),
            "source_instruction_groups": len(groups),
            "source_lane_bytes": source_lane_bytes,
            "coalesced_requests": len(coalesced),
            "coalesced_bytes": coalesced_bytes,
            "cache_stats": dict(sorted(cache.stats.items())) if cache else {},
            "cache_state": cache.state_summary() if cache else {},
            "measurement_entry_cache_state": measurement_entry_cache_state or {},
            "warmup": dict(sorted(warmup_counts.items())),
            "transform": transform_config,
        }
    )
    return output_records, summary


def validate_device_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    iterator = iter(records)
    try:
        header = next(iterator)
    except StopIteration as error:
        raise TraceContractError("device trace header must be first") from error
    if header.get("record_type") != "header":
        raise TraceContractError("device trace header must be first")
    if header.get("schema") != DEVICE_SCHEMA:
        raise TraceContractError("wrong device trace schema")
    contract = header.get("trace_contract")
    if not isinstance(contract, dict) or contract.get("placement_neutral") is not True:
        raise TraceContractError("device trace is not placement-neutral")
    if contract.get("actual_memory_event_boundary") != DEVICE_BOUNDARY:
        raise TraceContractError("device trace has the wrong memory boundary")
    objects: dict[str, dict[str, Any]] = {}
    saw_event = False
    event_count = 0
    total_bytes = 0
    # Event ids are not required to encode the sequence index, so exact
    # duplicate detection still needs one compact id set.  Do not retain the
    # event dictionaries themselves: a full trace can make that several GiB.
    seen_ids: set[str] = set()
    for record in iterator:
        if record.get("record_type") == "object":
            if saw_event:
                raise TraceContractError("device trace objects must precede events")
            object_id = record.get("object_id")
            if not isinstance(object_id, str) or not object_id or object_id in objects:
                raise TraceContractError(f"bad device trace object {object_id!r}")
            positive_int(record.get("bytes"), f"object {object_id} bytes")
            nonnegative_int(record.get("logical_address"), f"object {object_id} address")
            objects[object_id] = record
        elif record.get("record_type") == "event":
            saw_event = True
            event = record
            index = event_count
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in seen_ids:
                raise TraceContractError(f"bad device event id {event_id!r}")
            if event.get("sequence_index") != index:
                raise TraceContractError(f"device event {event_id} has bad sequence")
            if event.get("event_type") != "memory_access":
                raise TraceContractError(f"device event {event_id} is not memory")
            if event.get("access_boundary") != DEVICE_BOUNDARY:
                raise TraceContractError(f"device event {event_id} has bad boundary")
            if event.get("op") not in {"R", "W"}:
                raise TraceContractError(f"device event {event_id} has bad op")
            object_id = event.get("object_id")
            if object_id not in objects:
                raise TraceContractError(f"device event {event_id} has unknown object")
            byte_count = positive_int(
                event.get("bytes"), f"device event {event_id} bytes"
            )
            offset = nonnegative_int(
                event.get("object_offset"), f"device event {event_id} offset"
            )
            obj = objects[object_id]
            if offset + byte_count > int(obj["bytes"]):
                raise TraceContractError(f"device event {event_id} escapes object")
            if event.get("logical_address") != int(obj["logical_address"]) + offset:
                raise TraceContractError(f"device event {event_id} address mismatch")
            if event.get("dependencies") != []:
                raise TraceContractError(
                    f"device event {event_id} has an unvalidated dependency DAG"
                )
            issue_ns = event.get("issue_ns")
            if (
                isinstance(issue_ns, bool)
                or not isinstance(issue_ns, (int, float))
                or issue_ns < 0
            ):
                raise TraceContractError(
                    f"device event {event_id} has an invalid issue time"
                )
            total_bytes += byte_count
            seen_ids.add(event_id)
            event_count += 1
        else:
            raise TraceContractError("unknown device trace record")
    if not objects or event_count == 0:
        raise TraceContractError("device trace requires objects and events")
    return {
        "objects": len(objects),
        "device_requests": event_count,
        "device_request_bytes": total_bytes,
        "memory_event_boundary": DEVICE_BOUNDARY,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sector-bytes", type=int, default=32)
    parser.add_argument(
        "--cache-mode", choices=("bypass", "reference-lru"), default="bypass"
    )
    parser.add_argument("--cache-capacity-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--cache-line-bytes", type=int, default=128)
    parser.add_argument("--cache-associativity", type=int, default=16)
    parser.add_argument(
        "--write-policy", choices=("write-through", "write-back"), default="write-through"
    )
    parser.add_argument("--no-write-allocate", action="store_true")
    parser.add_argument("--no-write-miss-fetch", action="store_true")
    parser.add_argument(
        "--read-admission", choices=READ_ADMISSION_POLICIES, default="uniform-mru"
    )
    parser.add_argument(
        "--cache-address-space", choices=CACHE_ADDRESS_SPACES, default="logical"
    )
    parser.add_argument("--final-drain", action="store_true")
    parser.add_argument("--warmup-kernel-count", type=int, default=0)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    output, summary = transform(
        records,
        source_path=args.input,
        sector_bytes=args.sector_bytes,
        cache_mode=args.cache_mode,
        cache_capacity_bytes=args.cache_capacity_bytes,
        cache_line_bytes=args.cache_line_bytes,
        cache_associativity=args.cache_associativity,
        write_policy=args.write_policy,
        write_allocate=not args.no_write_allocate,
        write_miss_fetch=not args.no_write_miss_fetch,
        read_admission=args.read_admission,
        cache_address_space=args.cache_address_space,
        warmup_kernel_count=args.warmup_kernel_count,
        final_drain=args.final_drain,
    )
    write_jsonl(args.output, output)
    manifest = {
        "schema": {"name": "hbfsim.gpu_request_transform_manifest", "version": 1},
        "status": "pass",
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        **summary,
    }
    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

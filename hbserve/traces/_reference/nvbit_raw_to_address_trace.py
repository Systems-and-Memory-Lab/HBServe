#!/usr/bin/env python3
"""Stream raw Accel-Sim/NVBit v5 records into the GPU address contract.

The importer is intentionally placement-neutral.  It emits exactly one event
per active global-memory lane, preserves raw execution order and provenance,
and leaves coalescing, caches, address routing, and device requests to later
named transforms.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import lzma
from pathlib import Path
from typing import Any, BinaryIO, Callable

from hbserve.traces._reference.load_store_trace_contract import (
    PAGE_BYTES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TARGET_BOUNDARY,
    align_up,
    canonical_json,
    field_evidence,
    mapping_rows,
    sha256_file,
    sha256_value,
    validate_trace_path,
)
from hbserve.traces._reference.nvbit_v5_trace import (
    NVBitV5Instruction,
    NVBitV5Record,
    NVBitV5TraceError,
    discover_trace_files,
    iter_instruction_records,
    kernel_id_from_filename,
    read_trace_header,
)
from hbserve.traces._reference.summarize_nvbit_addresses import access_mode, address_space


class RawLaneImportError(ValueError):
    """Raised when a raw trace would require guessing or lose lane events."""


RMW_POLICIES = ("reject", "split-read-write")


def is_rmw_instruction(instruction: NVBitV5Instruction) -> bool:
    """Return whether this record is a true global RMW, excluding LDGSTS."""
    if instruction.opcode.upper().startswith("LDGSTS"):
        return False
    metadata = instruction.memory_reference_metadata
    if metadata is not None:
        return metadata.is_load and metadata.is_store
    return access_mode(instruction.opcode) == "read_write"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RawLaneImportError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RawLaneImportError(f"cannot read JSON object {path}") from error
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


class RangeIndex:
    """Small deterministic interval index supporting overlapping known ranges."""

    def __init__(self, ranges: list[dict[str, Any]]) -> None:
        self.ranges = sorted(
            ranges,
            key=lambda item: (item["begin"], item["end"], item["source_name"]),
        )
        self.begins = [item["begin"] for item in self.ranges]
        self.prefix_max_end: list[int] = []
        current = 0
        for item in self.ranges:
            current = max(current, item["end"])
            self.prefix_max_end.append(current)

    def match(self, address: int, byte_count: int) -> dict[str, Any] | None:
        access_end = address + byte_count
        cursor = bisect_right(self.begins, address) - 1
        matches: list[dict[str, Any]] = []
        while cursor >= 0 and self.prefix_max_end[cursor] >= access_end:
            item = self.ranges[cursor]
            if item["begin"] <= address and access_end <= item["end"]:
                matches.append(item)
            cursor -= 1
        if not matches:
            return None
        return min(
            matches,
            key=lambda item: (
                item["end"] - item["begin"],
                item["source_name"],
            ),
        )


def _source_kind(name: str, *, weight: bool) -> str:
    if weight:
        return "weight"
    if name.startswith("kv_"):
        return "kv_cache"
    return "activation"


def _tensor_ownership_descriptors(
    probe_manifest: dict[str, Any],
) -> list[tuple[str, bool, dict[str, Any]]]:
    """Load observed tensor-storage spans without inventing workspace owners.

    Multiple tensor storage identities can reuse or overlap one GPU VA during
    the capture.  Raw NVBit records have kernel order but no direct join to the
    host observation clock, so overlapping ownership records are collapsed
    into one honest VA span whose descriptor preserves all candidates.
    """

    ownership = probe_manifest.get("tensor_ownership")
    if ownership is None:
        return []
    require(isinstance(ownership, dict), "malformed tensor ownership registry")
    require(
        ownership.get("schema")
        == {"name": "hbfsim.same_run_tensor_ownership", "version": 1},
        "unsupported tensor ownership schema",
    )
    allocations = ownership.get("allocations")
    require(isinstance(allocations, list), "tensor ownership has no allocations")

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for allocation in allocations:
        require(isinstance(allocation, dict), "malformed tensor allocation")
        allocation_id = allocation.get("allocation_id")
        require(
            isinstance(allocation_id, str)
            and allocation_id
            and allocation_id not in seen_ids,
            f"duplicate or malformed tensor allocation id {allocation_id!r}",
        )
        seen_ids.add(allocation_id)
        require(
            allocation.get("evidence") == "observed_tensor_storage",
            f"tensor allocation {allocation_id} is not observation-backed",
        )
        begin = allocation.get("address_begin")
        end = allocation.get("address_end_exclusive")
        storage_base = allocation.get("storage_base")
        storage_nbytes = allocation.get("storage_nbytes")
        require(
            isinstance(begin, int)
            and not isinstance(begin, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and isinstance(storage_base, int)
            and not isinstance(storage_base, bool)
            and isinstance(storage_nbytes, int)
            and not isinstance(storage_nbytes, bool)
            and begin == storage_base
            and end == storage_base + storage_nbytes
            and 0 < begin < end,
            f"tensor allocation {allocation_id} has an invalid storage extent",
        )
        views = allocation.get("tensor_views")
        require(
            isinstance(views, list) and views,
            f"tensor allocation {allocation_id} has no observed tensor views",
        )
        for view in views:
            require(
                isinstance(view, dict)
                and view.get("storage_base") == storage_base
                and view.get("storage_nbytes") == storage_nbytes,
                f"tensor allocation {allocation_id} has an inconsistent view",
            )
        candidates.append(
            {
                "begin": begin,
                "end": end,
                "allocation": allocation,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["begin"],
            item["end"],
            item["allocation"]["allocation_id"],
        )
    )
    components: list[list[dict[str, Any]]] = []
    component_end = 0
    for candidate in candidates:
        if not components or candidate["begin"] >= component_end:
            components.append([candidate])
            component_end = candidate["end"]
            continue
        components[-1].append(candidate)
        component_end = max(component_end, candidate["end"])

    result: list[tuple[str, bool, dict[str, Any]]] = []
    for ordinal, component in enumerate(components):
        begin = min(item["begin"] for item in component)
        end = max(item["end"] for item in component)
        allocation_candidates = [item["allocation"] for item in component]
        descriptor = {
            "address_begin": begin,
            "address_end_exclusive": end,
            "storage_base": begin,
            "storage_nbytes": end - begin,
            "ownership_status": (
                "spatial_candidate_single_generation"
                if len(allocation_candidates) == 1
                else "spatial_candidate_multiple_generations_or_overlaps"
            ),
            "temporal_join_status": "unavailable",
            "allocation_candidates": allocation_candidates,
            "derivation": (
                "union of overlapping same-run PyTorch tensor storage extents; "
                "no allocator workspace inferred"
            ),
        }
        result.append(
            (f"tensor_ownership_span_{ordinal:06d}", False, descriptor)
        )
    return result


def load_known_ranges(probe_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        probe_manifest.get("schema")
        == {"name": "hbfsim.qwen_nvbit_layer_probe", "version": 1},
        "unsupported probe manifest schema",
    )
    source_objects = probe_manifest.get("objects")
    require(isinstance(source_objects, dict), "probe manifest has no object registry")
    weights = source_objects.get("weights")
    require(isinstance(weights, list), "probe manifest has no weight registry")

    raw: list[tuple[str, bool, dict[str, Any]]] = []
    for descriptor in weights:
        require(isinstance(descriptor, dict), "malformed weight descriptor")
        raw.append((str(descriptor.get("name", "")), True, descriptor))
    for name, descriptor in source_objects.items():
        if name == "weights":
            continue
        require(isinstance(descriptor, dict), f"malformed descriptor for {name}")
        raw.append((name, False, descriptor))
    raw.extend(_tensor_ownership_descriptors(probe_manifest))

    ranges: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for name, is_weight, descriptor in raw:
        require(name and name not in seen_names, f"duplicate or empty object name {name!r}")
        seen_names.add(name)
        begin = descriptor.get("address_begin")
        end = descriptor.get("address_end_exclusive")
        require(
            isinstance(begin, int)
            and not isinstance(begin, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 < begin < end,
            f"object {name} has an invalid raw GPU range",
        )
        ranges.append(
            {
                "source_name": name,
                "kind": _source_kind(name, weight=is_weight),
                "begin": begin,
                "end": end,
                "attribution": "known",
                "source_descriptor": descriptor,
            }
        )
    require(ranges, "probe manifest has no known object ranges")
    return sorted(
        ranges,
        key=lambda item: (item["begin"], item["end"], item["source_name"]),
    )


def _kernel_descriptors(trace_files: list[Path]) -> list[dict[str, Any]]:
    descriptors = []
    for ordinal, path in enumerate(trace_files):
        try:
            header = read_trace_header(path)
        except NVBitV5TraceError as error:
            raise RawLaneImportError(str(error)) from error
        filename_id = kernel_id_from_filename(path)
        require(
            header["kernel_id"] == filename_id,
            f"{path.name}: header/file kernel id mismatch",
        )
        descriptors.append(
            {
                "ordinal": ordinal,
                "path": path,
                "file": path.name,
                "sha256": sha256_file(path),
                **header,
            }
        )
    return descriptors


def _add_touched_pages(pages: set[int], address: int, byte_count: int) -> None:
    first = address // PAGE_BYTES
    last = (address + byte_count - 1) // PAGE_BYTES
    pages.update(range(first, last + 1))


def anonymous_ranges(touched_pages: set[int]) -> list[dict[str, Any]]:
    if not touched_pages:
        return []
    ordered = sorted(touched_pages)
    spans: list[tuple[int, int]] = []
    first = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        spans.append((first, previous + 1))
        first = previous = page
    spans.append((first, previous + 1))
    return [
        {
            "source_name": f"anonymous_va_span_{ordinal:06d}",
            "kind": "anonymous",
            "begin": first_page * PAGE_BYTES,
            "end": end_page * PAGE_BYTES,
            "attribution": "anonymous",
            "source_descriptor": {
                "derivation": "contiguous touched 4-KiB GPU-VA pages",
                "first_page": first_page,
                "end_exclusive_page": end_page,
            },
        }
        for ordinal, (first_page, end_page) in enumerate(spans)
    ]


def normalized_objects(
    known: list[dict[str, Any]], anonymous: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cursor = 0
    result = []
    for ordinal, source in enumerate([*known, *anonymous]):
        cursor = align_up(cursor)
        byte_count = source["end"] - source["begin"]
        object_id = (
            f"known_{ordinal:04d}"
            if source["attribution"] == "known"
            else f"anonymous_{ordinal - len(known):06d}"
        )
        if source["attribution"] == "known":
            raw_address_level = "observed"
            raw_address_source = "same-run live allocation descriptor"
            raw_bytes_level = "observed"
            raw_bytes_source = "same-run live allocation extent"
            kind_source = "probe object registry role"
        else:
            raw_address_level = "derived"
            raw_address_source = "4-KiB floor of instruction-observed unmatched VA"
            raw_bytes_level = "derived"
            raw_bytes_source = "merged contiguous touched unmatched VA pages"
            kind_source = "absence from the bounded same-run object registry"
        raw_extent = {
            "extent_id": "raw00",
            "gpu_virtual_address": source["begin"],
            "bytes": byte_count,
            "roles": [source["attribution"]],
            "field_evidence": {
                "gpu_virtual_address": field_evidence(
                    raw_address_level, raw_address_source
                ),
                "bytes": field_evidence(raw_bytes_level, raw_bytes_source),
            },
        }
        item = {
            "record_type": "object",
            "object_id": object_id,
            "kind": source["kind"],
            "bytes": byte_count,
            "logical_address": cursor,
            "raw_gpu_va_extents": [raw_extent],
            "source_logical_address": None,
            "metadata": {
                "source_name": source["source_name"],
                "attribution": source["attribution"],
                **(
                    {
                        "tensor_ownership_status": source[
                            "source_descriptor"
                        ]["ownership_status"],
                        "tensor_allocation_candidates": len(
                            source["source_descriptor"][
                                "allocation_candidates"
                            ]
                        ),
                        "tensor_temporal_join_status": source[
                            "source_descriptor"
                        ]["temporal_join_status"],
                    }
                    if "ownership_status" in source["source_descriptor"]
                    else {}
                ),
            },
            "field_evidence": {
                "kind": field_evidence("derived", kind_source),
                "bytes": field_evidence(raw_bytes_level, raw_bytes_source),
                "logical_address": field_evidence(
                    "derived", "deterministic 4-KiB-aligned object packing"
                ),
                "raw_gpu_va_extents": field_evidence(
                    raw_address_level, raw_address_source
                ),
            },
            "source_object_sha256": sha256_value(source["source_descriptor"]),
        }
        source["object_record"] = item
        result.append(item)
        cursor += align_up(byte_count)
    return result


LaneSink = Callable[
    [
        dict[str, Any],
        NVBitV5Record,
        NVBitV5Instruction,
        int,
        int,
        int,
        str,
        dict[str, Any] | None,
    ],
    None,
]


def scan_raw(
    kernels: list[dict[str, Any]],
    known_index: RangeIndex,
    *,
    lane_sink: LaneSink | None = None,
    rmw_policy: str = "reject",
) -> dict[str, Any]:
    """Scan raw files once; optionally consume each global lane event."""
    require(rmw_policy in RMW_POLICIES, f"unsupported RMW policy {rmw_policy!r}")
    totals: Counter[str] = Counter()
    kernel_summaries = []
    for kernel in kernels:
        counters: Counter[str] = Counter()
        try:
            records = iter_instruction_records(kernel["path"])
            for record in records:
                instruction = record.instruction
                totals["raw_instruction_records"] += 1
                counters["raw_instruction_records"] += 1
                if instruction.memory_width == 0:
                    continue
                totals["dynamic_memory_instructions"] += 1
                counters["dynamic_memory_instructions"] += 1
                metadata = instruction.memory_reference_metadata
                ldgsts_global_source = False
                if metadata is not None:
                    totals["v7_mref_metadata_records"] += 1
                    counters["v7_mref_metadata_records"] += 1
                    flag_class = (
                        "load_store"
                        if metadata.is_load and metadata.is_store
                        else "load_only"
                        if metadata.is_load
                        else "store_only"
                        if metadata.is_store
                        else "neither_load_nor_store"
                    )
                    totals[f"nvbit_{flag_class}_records"] += 1
                    counters[f"nvbit_{flag_class}_records"] += 1
                    space = metadata.mref_memory_space_name.lower()
                    mode = (
                        "read_write"
                        if metadata.is_load and metadata.is_store
                        else "read"
                        if metadata.is_load
                        else "write"
                        if metadata.is_store
                        else "unknown"
                    )
                    lane_records = len(instruction.addresses)
                    lane_width_bytes = lane_records * instruction.memory_width
                    totals[f"explicit_{space}_mref_records"] += 1
                    totals[f"explicit_{space}_mref_lanes"] += lane_records
                    totals[f"explicit_{space}_mref_width_bytes"] += lane_width_bytes
                    counters[f"explicit_{space}_mref_records"] += 1
                    counters[f"explicit_{space}_mref_lanes"] += lane_records
                    counters[f"explicit_{space}_mref_width_bytes"] += lane_width_bytes
                    if instruction.opcode.upper().startswith("LDGSTS"):
                        require(
                            metadata.instruction_memory_space_name
                            == "GLOBAL_TO_SHARED",
                            f"{kernel['file']}:{record.line_number}: LDGSTS v7 "
                            "record lacks GLOBAL_TO_SHARED instruction space",
                        )
                        require(
                            space in {"global", "shared"},
                            f"{kernel['file']}:{record.line_number}: LDGSTS v7 "
                            f"record has unexpected mref space {space}",
                        )
                        role = (
                            "global_source" if space == "global" else "shared_destination"
                        )
                        require(
                            metadata.is_load and metadata.is_store,
                            f"{kernel['file']}:{record.line_number}: LDGSTS v7 "
                            "GLOBAL_TO_SHARED record must preserve raw "
                            "isLoad=true,isStore=true flags",
                        )
                        # NVBit v1.8 exposes only instruction-level direction:
                        # LDGSTS is both a load and a store.  For this one
                        # explicitly labeled compound instruction, derive the
                        # operand direction from the v7 mref space plus the ISA
                        # role confirmed by the cp.async synthetic gate.  No
                        # numerical address-domain heuristic is involved.
                        mode = "read" if space == "global" else "write"
                        totals[f"labeled_ldgsts_{role}_records"] += 1
                        totals[f"labeled_ldgsts_{role}_lanes"] += lane_records
                        totals[f"labeled_ldgsts_{role}_bytes"] += lane_width_bytes
                        counters[f"labeled_ldgsts_{role}_records"] += 1
                        counters[f"labeled_ldgsts_{role}_lanes"] += lane_records
                        counters[f"labeled_ldgsts_{role}_bytes"] += lane_width_bytes
                        totals[f"effective_ldgsts_{space}_{mode}_records"] += 1
                        totals[f"effective_ldgsts_{space}_{mode}_lanes"] += lane_records
                        totals[f"effective_ldgsts_{space}_{mode}_bytes"] += lane_width_bytes
                        counters[f"effective_ldgsts_{space}_{mode}_records"] += 1
                        counters[f"effective_ldgsts_{space}_{mode}_lanes"] += lane_records
                        counters[f"effective_ldgsts_{space}_{mode}_bytes"] += lane_width_bytes
                elif instruction.opcode.upper().startswith("LDGSTS"):
                    # NVBit v5 emits separate address records for LDGSTS's
                    # shared destination and global source but does not tag
                    # the operand.  Shared records are zero-based offsets
                    # (for example 0x0 and 0x240); preserve them as a census
                    # rather than pretending they are GPU VAs.
                    lane_records = len(instruction.addresses)
                    lane_width_bytes = lane_records * instruction.memory_width
                    if not instruction.addresses:
                        totals["inactive_ldgsts_records"] += 1
                        counters["inactive_ldgsts_records"] += 1
                        continue
                    shared_bytes = kernel.get("shared_memory_bytes")
                    shared_base = kernel.get("shared_memory_base_address")
                    require(
                        isinstance(shared_bytes, int) and shared_bytes > 0,
                        f"{kernel['file']}: LDGSTS needs a positive shmem header",
                    )

                    def is_shared_address(address: int) -> bool:
                        if 0 <= address < shared_bytes:
                            return True
                        return (
                            isinstance(shared_base, int)
                            and shared_base <= address < shared_base + shared_bytes
                        )

                    shared_flags = [
                        is_shared_address(address) for address in instruction.addresses
                    ]
                    require(
                        all(shared_flags) or not any(shared_flags),
                        f"{kernel['file']}:{record.line_number}: mixed LDGSTS address domains",
                    )
                    if all(shared_flags):
                        totals["omitted_ldgsts_shared_destination_records"] += 1
                        totals["omitted_ldgsts_shared_destination_lanes"] += lane_records
                        totals["omitted_ldgsts_shared_destination_width_bytes"] += lane_width_bytes
                        counters["omitted_ldgsts_shared_destination_records"] += 1
                        counters["omitted_ldgsts_shared_destination_lanes"] += lane_records
                        counters["omitted_ldgsts_shared_destination_width_bytes"] += lane_width_bytes
                        continue
                    totals["imported_ldgsts_global_source_records"] += 1
                    totals["imported_ldgsts_global_source_lanes"] += lane_records
                    totals["imported_ldgsts_global_source_bytes"] += lane_width_bytes
                    counters["imported_ldgsts_global_source_records"] += 1
                    counters["imported_ldgsts_global_source_lanes"] += lane_records
                    counters["imported_ldgsts_global_source_bytes"] += lane_width_bytes
                    ldgsts_global_source = True
                if metadata is None:
                    space = (
                        "global"
                        if ldgsts_global_source
                        else address_space(instruction.opcode)
                    )
                    mode = (
                        "read"
                        if ldgsts_global_source
                        else access_mode(instruction.opcode)
                    )
                totals[f"{space}_memory_instructions"] += 1
                counters[f"{space}_memory_instructions"] += 1
                if space != "global":
                    continue
                if mode == "read_write":
                    classification_source = (
                        "v7 raw isLoad/isStore flags are both true"
                        if metadata is not None
                        else "v5 opcode classification is read-modify-write"
                    )
                    if rmw_policy == "reject":
                        raise RawLaneImportError(
                            f"{kernel['file']}:{record.line_number}: atomic/RMW opcode "
                            f"{instruction.opcode} is unsupported by v1 R/W events; "
                            f"{classification_source}"
                        )
                    lane_count = len(instruction.addresses)
                    lane_bytes = lane_count * instruction.memory_width
                    totals["original_rmw_instructions"] += 1
                    totals["original_rmw_lanes"] += lane_count
                    totals["original_rmw_lane_bytes"] += lane_bytes
                    counters["original_rmw_instructions"] += 1
                    counters["original_rmw_lanes"] += lane_count
                    counters["original_rmw_lane_bytes"] += lane_bytes
                    effective_modes = ("read", "write")
                else:
                    require(
                        mode in {"read", "write"},
                        f"{kernel['file']}:{record.line_number}: cannot classify global "
                        f"opcode {instruction.opcode}",
                    )
                    effective_modes = (mode,)

                # An atomic is lowered as one complete warp read group followed
                # by one complete warp write group.  Keeping the groups separate
                # preserves the source RMW order while remaining representable in
                # the v1 pure-R/pure-W event contract.
                for effective_mode in effective_modes:
                    operation = "R" if effective_mode == "read" else "W"
                    totals[f"{effective_mode}_global_instructions"] += 1
                    counters[f"{effective_mode}_global_instructions"] += 1
                    for lane_ordinal, (lane_id, address) in enumerate(
                        zip(instruction.active_lanes, instruction.addresses, strict=True)
                    ):
                        require(
                            address > 0,
                            f"{kernel['file']}:{record.line_number}: active global lane "
                            "has a non-positive GPU VA",
                        )
                        known = known_index.match(address, instruction.memory_width)
                        attribution = "known" if known is not None else "anonymous"
                        byte_count = instruction.memory_width
                        totals["global_lane_accesses"] += 1
                        totals["global_lane_bytes"] += byte_count
                        totals[f"{effective_mode}_lane_accesses"] += 1
                        totals[f"{effective_mode}_lane_bytes"] += byte_count
                        totals[f"{attribution}_lane_accesses"] += 1
                        totals[f"{attribution}_lane_bytes"] += byte_count
                        counters["global_lane_accesses"] += 1
                        counters["global_lane_bytes"] += byte_count
                        counters[f"{effective_mode}_lane_accesses"] += 1
                        counters[f"{effective_mode}_lane_bytes"] += byte_count
                        counters[f"{attribution}_lane_accesses"] += 1
                        counters[f"{attribution}_lane_bytes"] += byte_count
                        if mode == "read_write":
                            expanded = f"expanded_{effective_mode}_events"
                            totals[expanded] += 1
                            counters[expanded] += 1
                        if (
                            known is not None
                            and "ownership_status" in known["source_descriptor"]
                        ):
                            totals["tensor_spatial_candidate_lane_accesses"] += 1
                            totals["tensor_spatial_candidate_lane_bytes"] += byte_count
                            counters["tensor_spatial_candidate_lane_accesses"] += 1
                            counters["tensor_spatial_candidate_lane_bytes"] += byte_count
                        if lane_sink is not None:
                            lane_sink(
                                kernel,
                                record,
                                instruction,
                                lane_ordinal,
                                lane_id,
                                address,
                                operation,
                                known,
                            )
        except NVBitV5TraceError as error:
            raise RawLaneImportError(str(error)) from error
        ldgsts_operand_census: dict[str, dict[str, int]] = {}
        for public_metric, counter_suffix in (
            ("records", "records"),
            ("active_lanes", "lanes"),
            ("width_bytes", "bytes"),
        ):
            ldgsts_operand_census[public_metric] = {
                "global_source": counters.get(
                    f"labeled_ldgsts_global_source_{counter_suffix}", 0
                ),
                "shared_destination": counters.get(
                    f"labeled_ldgsts_shared_destination_{counter_suffix}", 0
                ),
            }
        has_labeled_ldgsts = any(
            pair["global_source"] or pair["shared_destination"]
            for pair in ldgsts_operand_census.values()
        )
        census_balanced = all(
            pair["global_source"] == pair["shared_destination"]
            for pair in ldgsts_operand_census.values()
        )
        if has_labeled_ldgsts:
            totals["kernels_with_labeled_ldgsts"] += 1
            if census_balanced:
                totals["balanced_labeled_ldgsts_kernels"] += 1
            else:
                totals["unbalanced_labeled_ldgsts_kernels"] += 1
                counters["unbalanced_labeled_ldgsts_operand_census"] += 1
        kernel_summaries.append(
            {
                "ordinal": kernel["ordinal"],
                "file": kernel["file"],
                "kernel_id": kernel["kernel_id"],
                **dict(sorted(counters.items())),
                **(
                    {
                        "labeled_ldgsts_operand_census": {
                            **ldgsts_operand_census,
                            "balanced": census_balanced,
                        }
                    }
                    if has_labeled_ldgsts
                    else {}
                ),
            }
        )

    require(totals["global_memory_instructions"] > 0, "trace has no global instructions")
    require(totals["global_lane_accesses"] > 0, "trace has no active global lanes")
    require(
        totals["known_lane_accesses"] + totals["anonymous_lane_accesses"]
        == totals["global_lane_accesses"],
        "known/anonymous lane count does not conserve the global total",
    )
    require(
        totals["known_lane_bytes"] + totals["anonymous_lane_bytes"]
        == totals["global_lane_bytes"],
        "known/anonymous bytes do not conserve the global total",
    )
    require(
        totals["read_lane_accesses"] + totals["write_lane_accesses"]
        == totals["global_lane_accesses"],
        "read/write lane count does not conserve the global total",
    )
    require(
        totals["read_lane_bytes"] + totals["write_lane_bytes"]
        == totals["global_lane_bytes"],
        "read/write bytes do not conserve the global total",
    )
    return {
        "totals": dict(sorted(totals.items())),
        "kernels": kernel_summaries,
    }


def _mapping_header(
    probe_manifest_path: Path,
    probe_manifest: dict[str, Any],
    kernels: list[dict[str, Any]],
    order_evidence: dict[str, Any],
    objects: list[dict[str, Any]],
    first_pass: dict[str, Any],
) -> dict[str, Any]:
    mapping_sha256 = sha256_value(mapping_rows(objects))
    population_bytes = align_up(
        objects[-1]["logical_address"] + objects[-1]["bytes"]
    )
    kernel_rows = [
        {
            key: kernel[key]
            for key in (
                "ordinal",
                "file",
                "sha256",
                "kernel_id",
                "kernel_name",
                "cuda_stream_id",
                "accelsim_tracer_version",
                "nvbit_version",
                "grid_dim",
                "block_dim",
                "shared_memory_bytes",
                "shared_memory_base_address",
            )
        }
        for kernel in kernels
    ]
    omitted_ldgsts = first_pass["totals"].get(
        "omitted_ldgsts_shared_destination_records", 0
    )
    labeled_ldgsts = first_pass["totals"].get(
        "labeled_ldgsts_global_source_records", 0
    ) + first_pass["totals"].get("labeled_ldgsts_shared_destination_records", 0)
    unbalanced_labeled_ldgsts = first_pass["totals"].get(
        "unbalanced_labeled_ldgsts_kernels", 0
    )
    source_coverage_status = (
        "partial" if omitted_ldgsts or unbalanced_labeled_ldgsts else "complete"
    )
    return {
        "record_type": "header",
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "classification": "instruction-observed placement-neutral lane address trace",
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
                    "device commands",
                ],
            },
            "actual_memory_event_boundaries": [TARGET_BOUNDARY],
            "target_boundary_coverage": source_coverage_status,
            "source_operation_coverage": {
                "status": source_coverage_status,
                "ambiguous_opcode_family": (
                    "LDGSTS"
                    if omitted_ldgsts or unbalanced_labeled_ldgsts
                    else None
                ),
                "omitted_shared_destination_records": omitted_ldgsts,
                "labeled_ldgsts_operand_census": {
                    "scope": "per-kernel",
                    "equality_metrics": [
                        "records",
                        "active_lanes",
                        "width_bytes",
                    ],
                    "unbalanced_kernels": unbalanced_labeled_ldgsts,
                },
                **(
                    {
                        "explicitly_labeled_ldgsts_records": labeled_ldgsts,
                        "mref_metadata_format": "MEM_META_V1",
                        "effective_mref_operation": {
                            "applicability": (
                                "explicit v7 GLOBAL_TO_SHARED/LDGSTS operands only"
                            ),
                            "mapping": {
                                "GLOBAL": "R",
                                "SHARED": "W",
                            },
                            "evidence_level": "derived",
                            "basis": [
                                "explicit MEM_META_V1 mref memory space",
                                "GLOBAL_TO_SHARED/LDGSTS ISA operand semantics",
                                "cp.async synthetic address/index validation",
                            ],
                            "raw_instruction_flags_preserved": {
                                "isLoad": True,
                                "isStore": True,
                            },
                            "other_dual_flag_instructions": "fail-closed",
                        },
                    }
                    if first_pass["totals"].get("v7_mref_metadata_records")
                    else {}
                ),
                "reason": (
                    "tracer emits untagged LDGSTS operand records; shared-memory "
                    "destination offsets are omitted and global-source VAs are "
                    "identified by address domain"
                    if omitted_ldgsts
                    else "v7 LDGSTS operand labels are present, but global-source "
                    "and shared-destination records do not conserve records, "
                    "active lanes, and width-bytes within every kernel"
                    if unbalanced_labeled_ldgsts
                    else "v7 records explicitly label instruction and mref spaces; "
                    "GLOBAL_TO_SHARED/LDGSTS operand operations are derived from "
                    "the explicit mref role while raw dual direction flags are preserved"
                    if labeled_ldgsts
                    else "all captured global-memory opcodes expose a usable GPU VA"
                ),
            },
            "placement_neutral": True,
            "evidence_levels": ["observed", "derived", "hypothetical"],
            "event_id_scheme": "lane_{sequence_index:012d}",
            "issue_time": {
                "status": "unknown",
                "reason": "raw tracer v5 records order but no per-access timestamp",
            },
        },
        "address_contract": {
            "raw_address_space": "same-run process-local GPU virtual address",
            "logical_address_space": "portable object-offset-preserving virtual address",
            "logical_mapping": {
                "scheme": "known objects then touched anonymous spans, 4-KiB aligned",
                "alignment_bytes": PAGE_BYTES,
                "mapping_table_sha256": mapping_sha256,
                "population_bytes": population_bytes,
            },
            **(
                {
                    "transient_tensor_ownership": {
                        "status": "spatial_candidate_only",
                        "evidence": "same-run observed PyTorch tensor storage extents",
                        "allocation_generation_to_kernel_temporal_join": "unavailable",
                        "library_workspace_identity_claimed": False,
                    }
                }
                if probe_manifest.get("tensor_ownership") is not None
                else {}
            ),
        },
        "source": {
            "kind": "accelsim-nvbit-v5-raw-lane",
            "probe_manifest": str(probe_manifest_path.resolve()),
            "probe_manifest_sha256": sha256_file(probe_manifest_path),
            "probe_schema": probe_manifest.get("schema"),
            "kernel_order": order_evidence,
            "kernel_files": kernel_rows,
            "capture_census": first_pass["totals"],
        },
        "not_claimed": [
            "production issue timestamps or cross-kernel overlap",
            *(
                [
                    "fully tagged LDGSTS operand identity when source-operation coverage is partial"
                ]
                if source_coverage_status == "partial"
                else []
            ),
            "GPU L1/L2 misses",
            "memory-controller requests",
            "HBM/HBF address routing",
            "HBF device commands",
        ],
    }


def _open_output(path: Path, xz_preset: int) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xz":
        return lzma.open(path, "wb", preset=xz_preset)
    return path.open("wb")


def _write_record(stream: BinaryIO, record: dict[str, Any]) -> None:
    stream.write(canonical_json(record) + b"\n")


def _source_event_hash(
    kernel_sha256: str,
    record: NVBitV5Record,
    lane_id: int,
    rmw_operation: str | None = None,
) -> str:
    evidence = {
        "trace_file_sha256": kernel_sha256,
        "raw_record_index": record.raw_record_index,
        "raw_line_sha256": record.raw_line_sha256,
        "lane_id": lane_id,
    }
    if rmw_operation is not None:
        evidence["rmw_expanded_operation"] = rmw_operation
    return sha256_value(evidence)


def _memory_event(
    *,
    sequence_index: int,
    kernel: dict[str, Any],
    record: NVBitV5Record,
    instruction: NVBitV5Instruction,
    lane_ordinal: int,
    lane_id: int,
    address: int,
    operation: str,
    source_range: dict[str, Any],
) -> dict[str, Any]:
    obj = source_range["object_record"]
    object_offset = address - source_range["begin"]
    attribution = source_range["attribution"]
    if "ownership_status" in source_range["source_descriptor"]:
        attribution_source = (
            "full-range containment in a same-run observed PyTorch tensor-storage "
            "VA span; allocation-generation-to-kernel temporal join is unavailable, "
            "so library-workspace identity is not claimed"
        )
    elif attribution == "known":
        attribution_source = "full-range containment in a same-run registered allocation"
    else:
        attribution_source = (
            "full-range containment in a page-rounded unmatched VA span"
        )
    mref = instruction.memory_reference_metadata
    mref_metadata = (
        {
            "format": "MEM_META_V1",
            "instruction_memory_space": mref.instruction_memory_space,
            "instruction_memory_space_name": mref.instruction_memory_space_name,
            "mref_memory_space": mref.mref_memory_space,
            "mref_memory_space_name": mref.mref_memory_space_name,
            "num_mref": mref.num_mref,
            "instrumentation_mref_index": mref.instrumentation_mref_index,
            "address_mref_index": mref.address_mref_index,
            "nvbit_is_load": mref.is_load,
            "nvbit_is_store": mref.is_store,
        }
        if mref is not None
        else None
    )
    is_compound_ldgsts_operand = (
        mref is not None
        and instruction.opcode.upper().startswith("LDGSTS")
        and mref.instruction_memory_space_name == "GLOBAL_TO_SHARED"
    )
    is_expanded_rmw = is_rmw_instruction(instruction)
    if is_compound_ldgsts_operand:
        mref_metadata["effective_mref_operation"] = operation
        mref_metadata["effective_mref_operation_evidence"] = {
            "level": "derived",
            "source": (
                "explicit v7 mref space plus GLOBAL_TO_SHARED/LDGSTS ISA "
                "operand semantics and cp.async synthetic validation"
            ),
        }
        operation_level = "derived"
        operation_source = (
            "explicit v7 mref space plus GLOBAL_TO_SHARED/LDGSTS ISA operand "
            "semantics and cp.async synthetic validation; raw instruction "
            "isLoad/isStore flags remain preserved"
        )
    else:
        operation_level = "observed"
        operation_source = (
            "NVBit v7 isLoad/isStore flags"
            if mref is not None
            else "executed SASS opcode classified as load or store"
        )
    return {
        "record_type": "event",
        "sequence_index": sequence_index,
        "event_id": f"lane_{sequence_index:012d}",
        "event_type": "memory_access",
        "kind": obj["kind"],
        "access_boundary": TARGET_BOUNDARY,
        "op": operation,
        "raw_gpu_virtual_address": address,
        "raw_extent_id": "raw00",
        "logical_address": obj["logical_address"] + object_offset,
        "bytes": instruction.memory_width,
        "object_id": obj["object_id"],
        "object_offset": object_offset,
        "issue_ns": None,
        "duration_ns": 0,
        "dependencies": [],
        "metadata": {
            "source_format": "accelsim_tracer_v5",
            "kernel_ordinal": kernel["ordinal"],
            "kernel_id": kernel["kernel_id"],
            "kernel_file": kernel["file"],
            "raw_record_index": record.raw_record_index,
            "source_line_number": record.line_number,
            "instruction_group_id": (
                f"kernel_{kernel['ordinal']:06d}_record_{record.raw_record_index:012d}"
                + (f":rmw_{'read' if operation == 'R' else 'write'}" if is_expanded_rmw else "")
            ),
            "cta": list(instruction.cta),
            "warp_in_cta": instruction.warp_in_cta,
            "cluster": list(instruction.cluster),
            "cluster_cta": list(instruction.cluster_cta),
            "cluster_rank": instruction.cluster_rank,
            "pc": f"0x{instruction.pc:x}",
            "active_mask": f"0x{instruction.active_mask:08x}",
            "lane_ordinal_within_active_mask": lane_ordinal,
            "lane_id": lane_id,
            "opcode": instruction.opcode,
            "address_compression_mode": instruction.address_format,
            "raw_line_sha256": record.raw_line_sha256,
            "issue_time_status": "unknown",
            **(
                {"memory_reference_metadata": mref_metadata}
                if mref_metadata is not None
                else {}
            ),
            **(
                {
                    "rmw_lowering": "split-read-write",
                    "rmw_phase": "read" if operation == "R" else "write",
                }
                if is_expanded_rmw
                else {}
            ),
        },
        "field_evidence": {
            "access_boundary": field_evidence(
                "observed", "executed Accel-Sim/NVBit v5 global-memory record"
            ),
            "op": field_evidence(operation_level, operation_source),
            "raw_gpu_virtual_address": field_evidence(
                "observed", "active-lane address emitted by the raw tracer"
            ),
            "logical_address": field_evidence(
                "derived", "normalized object base plus validated raw object offset"
            ),
            "bytes": field_evidence(
                "observed", "per-lane memory width emitted by the raw tracer"
            ),
            "issue_ns": {
                "level": "derived",
                "status": "unknown",
                "source": "raw tracer provides order but no per-access timestamp",
            },
            "dependencies": field_evidence(
                "derived", "raw order is sequence_index; no dependency DAG is encoded"
            ),
            "object_attribution": field_evidence("derived", attribution_source),
        },
        "source_event_sha256": _source_event_hash(
            kernel["sha256"],
            record,
            lane_id,
            operation if is_expanded_rmw else None,
        ),
    }


def verify_import_artifact(
    output_path: Path, output_manifest_path: Path
) -> dict[str, Any]:
    """Verify the compressed bytes, streaming schema, and stored conservation."""
    manifest = _load_json(output_manifest_path)
    require(
        manifest.get("schema")
        == {"name": "hbfsim.nvbit_raw_lane_import", "version": 1},
        "wrong raw-lane import manifest schema",
    )
    require(
        sha256_file(output_path) == manifest.get("output_sha256"),
        "output SHA-256 mismatch",
    )
    summary = validate_trace_path(output_path)
    require(
        summary == manifest.get("streaming_validation"),
        "streaming validation summary disagrees with manifest",
    )
    second = (manifest.get("second_pass") or {}).get("totals") or {}
    require(
        summary["memory_events"] == second.get("global_lane_accesses"),
        "validated event count disagrees with raw-lane total",
    )
    require(
        summary["memory_bytes"] == second.get("global_lane_bytes"),
        "validated bytes disagree with raw-lane total",
    )
    return summary


def import_raw_trace(
    *,
    probe_manifest_path: Path,
    trace_root: Path,
    output_path: Path,
    output_manifest_path: Path | None = None,
    kernel_list: Path | None = None,
    kernel_ids: set[int] | None = None,
    xz_preset: int = 1,
    rmw_policy: str = "reject",
) -> dict[str, Any]:
    require(0 <= xz_preset <= 9, "XZ preset must be in [0, 9]")
    require(rmw_policy in RMW_POLICIES, f"unsupported RMW policy {rmw_policy!r}")
    probe_manifest_path = probe_manifest_path.resolve()
    trace_root = trace_root.resolve()
    output_path = output_path.resolve()
    output_manifest_path = (
        output_manifest_path.resolve()
        if output_manifest_path is not None
        else Path(f"{output_path}.manifest.json")
    )
    probe_manifest = _load_json(probe_manifest_path)
    known = load_known_ranges(probe_manifest)
    known_index = RangeIndex(known)
    try:
        trace_files, order_evidence = discover_trace_files(trace_root, kernel_list)
    except NVBitV5TraceError as error:
        raise RawLaneImportError(str(error)) from error
    if kernel_ids is not None:
        require(kernel_ids, "kernel-id selection cannot be empty")
        selected = [
            path for path in trace_files if kernel_id_from_filename(path) in kernel_ids
        ]
        observed_ids = {kernel_id_from_filename(path) for path in selected}
        require(
            observed_ids == kernel_ids,
            f"requested kernel ids not present: {sorted(kernel_ids - observed_ids)}",
        )
        trace_files = selected
        order_evidence = {
            **order_evidence,
            "selection": "requested kernel ids, preserving formal kernel-list order",
            "selected_kernel_ids": sorted(kernel_ids),
        }
    kernels = _kernel_descriptors(trace_files)

    touched_unknown_pages: set[int] = set()

    def collect_unknown(
        kernel: dict[str, Any],
        record: NVBitV5Record,
        instruction: NVBitV5Instruction,
        lane_ordinal: int,
        lane_id: int,
        address: int,
        operation: str,
        known_range: dict[str, Any] | None,
    ) -> None:
        del kernel, record, lane_ordinal, lane_id, operation
        if known_range is None:
            _add_touched_pages(
                touched_unknown_pages, address, instruction.memory_width
            )

    first_pass = scan_raw(
        kernels, known_index, lane_sink=collect_unknown, rmw_policy=rmw_policy
    )
    anonymous = anonymous_ranges(touched_unknown_pages)
    anonymous_index = RangeIndex(anonymous)
    objects = normalized_objects(known, anonymous)
    header = _mapping_header(
        probe_manifest_path,
        probe_manifest,
        kernels,
        order_evidence,
        objects,
        first_pass,
    )

    sequence_index = 0
    with _open_output(output_path, xz_preset) as stream:
        _write_record(stream, header)
        for obj in objects:
            _write_record(stream, obj)

        def emit_lane(
            kernel: dict[str, Any],
            record: NVBitV5Record,
            instruction: NVBitV5Instruction,
            lane_ordinal: int,
            lane_id: int,
            address: int,
            operation: str,
            known_range: dict[str, Any] | None,
        ) -> None:
            nonlocal sequence_index
            source_range = known_range
            if source_range is None:
                source_range = anonymous_index.match(address, instruction.memory_width)
            require(
                source_range is not None,
                f"second pass cannot map address 0x{address:x}",
            )
            event = _memory_event(
                sequence_index=sequence_index,
                kernel=kernel,
                record=record,
                instruction=instruction,
                lane_ordinal=lane_ordinal,
                lane_id=lane_id,
                address=address,
                operation=operation,
                source_range=source_range,
            )
            _write_record(stream, event)
            sequence_index += 1

        second_pass = scan_raw(
            kernels, known_index, lane_sink=emit_lane, rmw_policy=rmw_policy
        )

    require(
        first_pass == second_pass,
        "first/second raw scans disagree; source changed or importer is nondeterministic",
    )
    require(
        sequence_index == second_pass["totals"]["global_lane_accesses"],
        "emitted event count does not conserve raw lane accesses",
    )
    streaming_validation = validate_trace_path(output_path)
    require(
        streaming_validation["memory_events"] == sequence_index,
        "streaming validator event count disagrees with raw input",
    )
    require(
        streaming_validation["memory_bytes"]
        == second_pass["totals"]["global_lane_bytes"],
        "streaming validator bytes disagree with raw input",
    )

    manifest = {
        "schema": {"name": "hbfsim.nvbit_raw_lane_import", "version": 1},
        "status": "pass",
        "probe_manifest": str(probe_manifest_path),
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "trace_root": str(trace_root),
        "kernel_order": order_evidence,
        "kernel_files": [
            {
                key: kernel[key]
                for key in ("ordinal", "file", "sha256", "kernel_id")
            }
            for kernel in kernels
        ],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_compression": "xz" if output_path.suffix == ".xz" else "none",
        "objects": {
            "known": len(known),
            "anonymous": len(anonymous),
            "anonymous_touched_pages": len(touched_unknown_pages),
            "mapping_table_sha256": header["address_contract"]["logical_mapping"][
                "mapping_table_sha256"
            ],
        },
        "first_pass": first_pass,
        "second_pass": second_pass,
        "rmw_policy": rmw_policy,
        "conservation": {
            "first_equals_second": True,
            "emitted_events_equal_global_lane_accesses": True,
            "validated_bytes_equal_global_lane_bytes": True,
            "omitted_ldgsts_preserved_as_separate_census": True,
            "labeled_ldgsts_pair_census_scope": "per-kernel",
            "labeled_ldgsts_pair_census_metrics": [
                "records",
                "active_lanes",
                "width_bytes",
            ],
            "labeled_ldgsts_pair_census_balanced": (
                first_pass["totals"].get(
                    "unbalanced_labeled_ldgsts_kernels", 0
                )
                == 0
            ),
        },
        "streaming_validation": streaming_validation,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--kernel-list", type=Path)
    parser.add_argument("--kernel-id", type=int, action="append")
    parser.add_argument("--xz-preset", type=int, default=1)
    parser.add_argument(
        "--rmw-policy",
        choices=RMW_POLICIES,
        default="reject",
        help="reject atomics (default) or lower each RMW lane to ordered R then W events",
    )
    args = parser.parse_args()
    manifest = import_raw_trace(
        probe_manifest_path=args.probe_manifest,
        trace_root=args.trace_root,
        output_path=args.output,
        output_manifest_path=args.output_manifest,
        kernel_list=args.kernel_list,
        kernel_ids=set(args.kernel_id) if args.kernel_id is not None else None,
        xz_preset=args.xz_preset,
        rmw_policy=args.rmw_policy,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": manifest["output"],
                "output_sha256": manifest["output_sha256"],
                "objects": manifest["objects"],
                "totals": manifest["second_pass"]["totals"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

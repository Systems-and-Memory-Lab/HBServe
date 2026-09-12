#!/usr/bin/env python3
"""Stream a lazy trace plan as fixed-width target-object request records.

The plan remains the primary artifact.  This command materializes or pipes a
bounded stream only when a consumer requires it.  Each output record uses the
same 12-byte layout as a compact template, but ``object_index`` and
``kernel_ordinal`` are rebound to the full-model plan.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import stat
import sys
import time
from typing import Any, BinaryIO

from hbserve.traces.artifacts import resolve_input

from hbserve.traces._reference.compact_request_template import (
    KNOWN_REQUEST_FLAGS,
    OPERATION_NAME,
    RECORD_BYTES,
    RECORD_STRUCT,
    known_request_flags,
    load_json,
    require,
    sha256_file,
)
from hbserve.traces._reference.full_model_trace_plan import PLAN_SCHEMA
from hbserve.traces._reference.materialize_order_skeleton_template import (
    COMPILED_ORDER_ONLY_SCHEMA,
    SKELETON_RECORD_BYTES,
    SKELETON_SCHEMA,
    _bundle_ranges,
)
from hbserve.traces._reference.phase_aware_cta_generator import (
    bundle_order as phase_bundle_order,
    causal_is_implemented,
    causal_source_programs,
    generated_bundle_records,
    policy_rule_lookup,
    prepare_segment_generator,
    rule_coordinate,
)


SCHEMA = {"name": "hbfsim.streamed_lazy_trace_plan", "version": 1}


def source_object_kinds(manifest: dict[str, Any]) -> list[str]:
    objects = manifest.get("objects")
    require(isinstance(objects, list) and objects, "template has no object table")
    require(
        [int(item["template_object_index"]) for item in objects]
        == list(range(len(objects))),
        "template object indices are not dense",
    )
    return [str(item["kind"]) for item in objects]


def segment_template(
    plan: dict[str, Any], segment: dict[str, Any], shared_binary: Path | None
) -> tuple[dict[str, Any], Path]:
    template = segment.get("template") or plan.get("template")
    require(isinstance(template, dict), f"segment {segment['segment_id']} has no template")
    manifest_path = resolve_input(str(template["manifest"]))
    if segment.get("template") is None and shared_binary is not None:
        binary_path = shared_binary.resolve()
    else:
        binary_path = resolve_input(str(template["binary"]))
    manifest = load_json(manifest_path)
    require(binary_path.is_file(), f"missing template binary {binary_path}")
    require(
        binary_path.stat().st_size == int(manifest["requests"]) * RECORD_BYTES,
        f"template binary size disagrees for {segment['segment_id']}",
    )
    return manifest, binary_path


def segment_order_skeleton(
    *,
    segment: dict[str, Any],
    template_manifest_path: Path,
    repeat_count: int,
) -> tuple[dict[str, Any], Path] | None:
    descriptor = segment.get("order_skeleton")
    if descriptor is None:
        return None
    require(isinstance(descriptor, dict), "order-skeleton descriptor is malformed")
    manifest_path = resolve_input(str(descriptor.get("manifest", "")))
    binary_path = resolve_input(str(descriptor.get("binary", "")))
    require(manifest_path.is_file(), f"missing order-skeleton manifest {manifest_path}")
    require(
        descriptor.get("manifest_sha256") == sha256_file(manifest_path),
        "order-skeleton manifest digest differs from its plan binding",
    )
    manifest = load_json(manifest_path)
    require(
        manifest.get("schema") in (SKELETON_SCHEMA, COMPILED_ORDER_ONLY_SCHEMA),
        "unsupported order-skeleton schema",
    )
    require(manifest.get("status") == "PASS", "order skeleton has not passed")
    require(binary_path.is_file(), f"missing order skeleton {binary_path}")
    require(
        int(descriptor.get("record_bytes", -1)) == SKELETON_RECORD_BYTES,
        "order-skeleton binding has an unsupported record width",
    )
    require(
        int(descriptor.get("events", -1)) == int(manifest.get("events", -2)),
        "order-skeleton event count differs from its plan binding",
    )
    require(
        binary_path.stat().st_size
        == int(manifest.get("events", -1)) * SKELETON_RECORD_BYTES,
        "order-skeleton binary size disagrees",
    )
    require(
        manifest.get("binary_sha256") == sha256_file(binary_path),
        "order-skeleton binary digest differs",
    )
    require(
        descriptor.get("binary_sha256") == manifest.get("binary_sha256"),
        "order-skeleton binary digest differs from its plan binding",
    )
    source = manifest.get("source") or {}
    require(
        source.get("template_manifest_sha256")
        == sha256_file(template_manifest_path),
        "order skeleton refers to another address program",
    )
    require(
        int((manifest.get("geometry") or {}).get("repeat_count", -1))
        == repeat_count,
        "order skeleton repeat geometry differs from the segment",
    )
    return manifest, binary_path


def selected_segments(
    plan: dict[str, Any], segment_ids: set[str] | None
) -> list[dict[str, Any]]:
    segments = list(plan.get("segments") or [])
    require(segments, "plan has no request-bearing segments")
    if segment_ids is not None:
        segments = [item for item in segments if item["segment_id"] in segment_ids]
        observed = {item["segment_id"] for item in segments}
        require(observed == segment_ids,
                f"unknown requested segment ids: {sorted(segment_ids - observed)}")
    if all("sequence_ordinal_begin" in item for item in segments):
        segments.sort(key=lambda item: int(item["sequence_ordinal_begin"]))
    elif all("observed_kernel_begin" in item for item in segments):
        segments.sort(key=lambda item: int(item["observed_kernel_begin"]))
    return segments


def binding_vectors(
    *,
    segment: dict[str, Any],
    source_kinds: list[str],
    include_kinds: set[str] | None,
    runtime_parameters: dict[str, int],
    target_extents_by_index: list[int],
) -> tuple[list[int], list[bool], list[int], list[int]]:
    bindings = segment.get("object_bindings")
    require(isinstance(bindings, list), f"segment {segment['segment_id']} has no bindings")
    by_source = {
        int(item["source_template_object_index"]): item for item in bindings
    }
    require(len(by_source) == len(source_kinds), "segment bindings are incomplete")
    targets = []
    selected = []
    biases = []
    target_extents = []
    for source_index, kind in enumerate(source_kinds):
        binding = by_source[source_index]
        keep = include_kinds is None or kind in include_kinds
        target = binding.get("target_object_index")
        if keep:
            require(target is not None,
                    f"selected {kind} object {source_index} has no numeric binding")
            require(0 <= int(target) <= 0xFFFF,
                    "target object index exceeds compact u16")
        targets.append(int(target) if target is not None else 0)
        selected.append(keep)
        target_extents.append(
            target_extents_by_index[int(target)] if target is not None else 0
        )
        bias = int(binding.get("object_offset_bias_bytes", 0))
        parameter = binding.get("object_offset_parameter")
        if parameter is not None:
            require(isinstance(parameter, dict), "object offset parameter is malformed")
            name = str(parameter.get("name", ""))
            scale = int(parameter.get("scale_bytes", 0))
            bound_value = int(parameter.get("bound_value", -1))
            require(name and scale > 0 and bound_value >= 0,
                    "object offset parameter descriptor is invalid")
            value = int(runtime_parameters.get(name, bound_value))
            require(value >= 0, f"runtime parameter {name} must be nonnegative")
            bias += (value - bound_value) * scale
        biases.append(bias)
    return targets, selected, biases, target_extents


def declared_runtime_parameters(plan: dict[str, Any]) -> dict[str, dict[str, int]]:
    declared: dict[str, dict[str, int]] = {}
    for segment in plan.get("segments") or []:
        for binding in segment.get("object_bindings") or []:
            parameter = binding.get("object_offset_parameter")
            if parameter is None:
                continue
            value = {
                "scale_bytes": int(parameter["scale_bytes"]),
                "template_value": int(parameter["template_value"]),
                "bound_value": int(parameter["bound_value"]),
            }
            name = str(parameter["name"])
            current = declared.get(name)
            require(current is None or current == value,
                    f"runtime parameter {name} has inconsistent descriptors")
            declared[name] = value
    return declared


def segment_repetition(
    segment: dict[str, Any], object_count: int
) -> tuple[int, list[int]]:
    repeat = segment.get("repeat")
    if repeat is None:
        return 1, [0] * object_count
    require(isinstance(repeat, dict), "segment repeat descriptor is malformed")
    count = int(repeat.get("count", 0))
    require(count > 0, "segment repeat count must be positive")
    require(
        repeat.get("ordering", "instance-major") == "instance-major",
        "only instance-major repeat ordering is currently supported",
    )
    strides = repeat.get("object_offset_strides")
    require(
        isinstance(strides, list) and len(strides) == object_count,
        "segment repeat stride vector disagrees with its object table",
    )
    values = [int(value) for value in strides]
    require(all(value >= 0 for value in values), "repeat strides must be nonnegative")
    return count, values


def segment_phase_generator(
    *,
    segment: dict[str, Any],
    template: dict[str, Any],
    template_path: Path,
    cache: dict[str, tuple[Any, list[tuple[int, int, int]], dict[str, Any]]] | None = None,
) -> tuple[Any, list[tuple[int, int, int]], dict[str, Any]] | None:
    descriptor = segment.get("generator")
    if descriptor is None:
        return None
    require(isinstance(descriptor, dict), "segment generator descriptor is malformed")
    cache_key = hashlib.sha256(
        (
            str(template_path.resolve())
            + "\0"
            + json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        ).encode()
    ).hexdigest()
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    prepared = prepare_segment_generator(
        descriptor=descriptor,
        template_manifest=template,
        template_binary_path=template_path,
    )
    require(
        len(prepared.generator.target_extents) == len(template.get("objects") or []),
        "phase-aware target extent vector differs from the source object table",
    )
    result = prepared.generator, prepared.ranges, prepared.provenance
    if cache is not None:
        cache[cache_key] = result
    return result


def stream_phase_generator_segment(
    *,
    prepared: Any,
    ranges: list[tuple[int, int, int]],
    source_path: Path,
    target: BinaryIO,
    kernel_base: int,
    target_objects: list[int],
    selected_objects: list[bool],
    object_biases: list[int],
    target_object_extents: list[int],
    chunk_records: int,
    digest: Any,
    totals: Counter[str],
) -> dict[str, int]:
    """Generate CTA-x bodies and immediately rebind them into one plan segment."""

    output = bytearray()
    local: Counter[str] = Counter()

    def flush() -> None:
        if output:
            target.write(output)
            digest.update(output)
            output.clear()

    with source_path.open("rb") as source:
        for kernel_ordinal, begin, end in ranges:
            for target_cta_x in range(begin, end):
                local["target_ctas"] += 1
                for _bundle, records in generated_bundle_records(
                    prepared=prepared,
                    source=source,
                    kernel_ordinal=kernel_ordinal,
                    target_cta_x=target_cta_x,
                ):
                    local["bundles"] += 1
                    for record in records:
                        (
                            object_index,
                            object_offset,
                            source_kernel,
                            byte_count,
                            operation,
                            flags,
                        ) = record
                        require(source_kernel == kernel_ordinal,
                                "generated record changed its kernel ordinal")
                        if not selected_objects[object_index]:
                            totals["filtered_requests"] += 1
                            totals["filtered_bytes"] += byte_count
                            continue
                        rebound_offset = object_offset + object_biases[object_index]
                        require(0 <= rebound_offset <= 0xFFFFFFFF,
                                "expanded object offset exceeds compact u32")
                        require(
                            rebound_offset + byte_count
                            <= target_object_extents[object_index],
                            "expanded request escapes its target object extent",
                        )
                        full_kernel = kernel_base + kernel_ordinal
                        require(full_kernel <= 0xFFFF,
                                "full-model kernel ordinal exceeds u16")
                        output.extend(
                            RECORD_STRUCT.pack(
                                target_objects[object_index],
                                rebound_offset,
                                full_kernel,
                                byte_count,
                                operation,
                                flags,
                            )
                        )
                        totals["requests"] += 1
                        totals["bytes"] += byte_count
                        local["requests"] += 1
                        local["bytes"] += byte_count
                        if len(output) >= chunk_records * RECORD_BYTES:
                            flush()
    flush()
    return dict(sorted(local.items()))


def stream_numpy_tiled_phase_generator_segment(
    *,
    prepared: Any,
    ranges: list[tuple[int, int, int]],
    source_path: Path,
    target: BinaryIO,
    kernel_base: int,
    target_objects: list[int],
    selected_objects: list[bool],
    object_biases: list[int],
    target_object_extents: list[int],
    chunk_records: int,
    digest: Any,
    totals: Counter[str],
) -> dict[str, int] | None:
    """Vectorize validated tiled-swizzle generation without changing order.

    Returning ``None`` means that at least one policy is not tiled-swizzle and
    the caller should use the existing Python path.  This decision is made
    before writing output, so mixed-policy segments cannot be partially
    emitted and then silently retried.
    """

    if any(
        str(prepared.policies[kernel_ordinal]["kind"]) != "tiled_swizzle"
        for kernel_ordinal, _begin, _end in ranges
    ):
        return None

    import numpy as np

    dtype = np.dtype(
        [
            ("object_index", "<u2"),
            ("object_offset", "<u4"),
            ("kernel_ordinal", "<u2"),
            ("bytes", "<u2"),
            ("operation", "u1"),
            ("flags", "u1"),
        ],
        align=False,
    )
    require(dtype.itemsize == RECORD_BYTES, "NumPy compact dtype is not 12 bytes")
    source = np.memmap(source_path, mode="r", dtype=dtype)
    target_map = np.asarray(target_objects, dtype=np.uint16)
    selection_map = np.asarray(selected_objects, dtype=np.bool_)
    bias_map = np.asarray(object_biases, dtype=np.int64)
    extent_map = np.asarray(target_object_extents, dtype=np.int64)
    local: Counter[str] = Counter()

    for kernel_ordinal, begin, end in ranges:
        policy = prepared.policies[kernel_ordinal]
        source_x = int(policy["source_cta_x"])
        bundles = sorted(
            prepared.by_kernel_x.get((kernel_ordinal, source_x), []),
            key=phase_bundle_order,
        )
        require(bundles, f"kernel {kernel_ordinal}: source CTA has no captured bundles")
        source_count = sum(
            int(row["request_ordinal_end_exclusive"])
            - int(row["request_ordinal_begin"])
            for row in bundles
        )
        source_indices = np.empty(source_count, dtype=np.int64)
        source_y = np.empty(source_count, dtype=np.int32)
        source_z = np.empty(source_count, dtype=np.int32)
        cursor = 0
        for row in bundles:
            request_begin = int(row["request_ordinal_begin"])
            request_end = int(row["request_ordinal_end_exclusive"])
            count = request_end - request_begin
            source_indices[cursor : cursor + count] = np.arange(
                request_begin, request_end, dtype=np.int64
            )
            source_y[cursor : cursor + count] = int(row["cta_y"])
            source_z[cursor : cursor + count] = int(row["cta_z"])
            cursor += count
        require(cursor == source_count, "tiled source-index census differs")
        original = np.array(source[source_indices], copy=True)
        source_objects = original["object_index"].astype(np.int64)
        require(not bool((source_objects >= len(selected_objects)).any()),
                "tiled source request references an unknown object")
        require(not bool((source_z != 0).any()),
                "tiled-swizzle v1 source contains nonzero CTA z")
        require(not bool((original["kernel_ordinal"] != kernel_ordinal).any()),
                "tiled source request uses another kernel ordinal")
        require(not bool((original["operation"] > 1).any()),
                "tiled source request has an unsupported operation")
        unknown = original["flags"] & (~KNOWN_REQUEST_FLAGS & 0xFF)
        require(not bool((unknown != 0).any()), "unsupported compact flags")

        rules = list(policy["rules"])
        grid_y = int(policy["grid_cta_y"])
        rule_lookup = np.full(
            (grid_y, len(selected_objects), 2), -1, dtype=np.int32
        )
        for rule_id, rule in enumerate(rules):
            y = int(rule["cta_y"])
            z = int(rule["cta_z"])
            object_index = int(rule["object_index"])
            require(z == 0 and 0 <= y < grid_y,
                    "tiled rule escapes the supported CTA-y/z grid")
            operation = rule.get("operation")
            if operation is None:
                operation_indices = (0, 1)
            else:
                require(operation in {"R", "W"},
                        "tiled rule operation must be R or W")
                operation_indices = (0 if operation == "R" else 1,)
            for operation_index in operation_indices:
                require(rule_lookup[y, object_index, operation_index] == -1,
                        "duplicate or ambiguous tiled rule lookup entry")
                rule_lookup[y, object_index, operation_index] = rule_id
        source_operations = original["operation"].astype(np.int64)
        rule_ids = rule_lookup[source_y, source_objects, source_operations]
        require(not bool((rule_ids < 0).any()),
                "tiled source record has no validated object/y/operation rule")
        active_table = np.zeros((int(policy["period"]), grid_y), dtype=np.bool_)
        for phase, active_y in enumerate(policy["active_cta_y_by_phase"]):
            active_table[phase, np.asarray(active_y, dtype=np.int64)] = True
        bundle_y = np.asarray([int(row["cta_y"]) for row in bundles], dtype=np.int64)

        for target_cta_x in range(begin, end):
            phase = target_cta_x % int(policy["period"])
            topology_mask = active_table[phase, source_y]
            selected_mask = topology_mask & selection_map[source_objects]
            rejected_mask = topology_mask & ~selection_map[source_objects]
            if bool(rejected_mask.any()):
                rejected = original[rejected_mask]
                totals["filtered_requests"] += int(rejected.size)
                totals["filtered_bytes"] += int(rejected["bytes"].sum(dtype=np.uint64))
            positions = np.flatnonzero(selected_mask)
            delta_by_rule = np.asarray(
                [
                    rule_coordinate(rule, target_cta_x)
                    - rule_coordinate(rule, source_x)
                    for rule in rules
                ],
                dtype=np.int64,
            )
            local["target_ctas"] += 1
            local["bundles"] += int(active_table[phase, bundle_y].sum())
            for chunk_begin in range(0, int(positions.size), chunk_records):
                chunk_positions = positions[chunk_begin : chunk_begin + chunk_records]
                output = np.array(original[chunk_positions], copy=True)
                objects = source_objects[chunk_positions]
                full_offsets = (
                    output["object_offset"].astype(np.int64)
                    + delta_by_rule[rule_ids[chunk_positions]]
                    + bias_map[objects]
                )
                full_kernels = output["kernel_ordinal"].astype(np.uint32) + kernel_base
                require(not bool((full_kernels > 0xFFFF).any()),
                        "full-model kernel ordinal exceeds u16")
                require(
                    not bool(((full_offsets < 0) | (full_offsets > 0xFFFFFFFF)).any()),
                    "expanded object offset exceeds compact u32",
                )
                full_ends = full_offsets + output["bytes"].astype(np.int64)
                require(not bool((full_ends > extent_map[objects]).any()),
                        "expanded request escapes its target object extent")
                output["object_index"] = target_map[objects]
                output["object_offset"] = full_offsets.astype(np.uint32)
                output["kernel_ordinal"] = full_kernels.astype(np.uint16)
                payload = output.tobytes(order="C")
                target.write(payload)
                digest.update(payload)
                emitted = int(output.size)
                emitted_bytes = int(output["bytes"].sum(dtype=np.uint64))
                totals["requests"] += emitted
                totals["bytes"] += emitted_bytes
                local["requests"] += emitted
                local["bytes"] += emitted_bytes
    return dict(sorted(local.items()))


def stream_numpy_phase_rules_segment(
    *,
    prepared: Any,
    ranges: list[tuple[int, int, int]],
    source_path: Path,
    target: BinaryIO,
    kernel_base: int,
    target_objects: list[int],
    selected_objects: list[bool],
    object_biases: list[int],
    target_object_extents: list[int],
    chunk_records: int,
    digest: Any,
    totals: Counter[str],
) -> dict[str, int] | None:
    """Batch ordinary phase-rule CTAs while preserving the reference order.

    The original path calls the Python generator once per CTA and packs every
    request separately. Large prefill grids contain hundreds of thousands of
    CTAs even though each CTA reuses the same compact request body. This path
    resolves each source record to its validated rule once, then broadcasts
    those rules over a bounded CTA chunk with NumPy. It emits byte-for-byte
    the same CTA-major/bundle-major stream as ``generated_bundle_records``.

    Returning ``None`` is fail-closed: a mixed, causal, exact-anchor, or tiled
    policy is left to its existing implementation before output is written.
    """

    if any(
        str(prepared.policies[kernel_ordinal]["kind"]) != "phase_rules"
        for kernel_ordinal, _begin, _end in ranges
    ):
        return None

    import numpy as np

    dtype = np.dtype(
        [
            ("object_index", "<u2"),
            ("object_offset", "<u4"),
            ("kernel_ordinal", "<u2"),
            ("bytes", "<u2"),
            ("operation", "u1"),
            ("flags", "u1"),
        ],
        align=False,
    )
    require(dtype.itemsize == RECORD_BYTES, "NumPy compact dtype is not 12 bytes")
    source = np.memmap(source_path, mode="r", dtype=dtype)
    target_map = np.asarray(target_objects, dtype=np.uint16)
    selection_map = np.asarray(selected_objects, dtype=np.bool_)
    bias_map = np.asarray(object_biases, dtype=np.int64)
    extent_map = np.asarray(target_object_extents, dtype=np.int64)
    local: Counter[str] = Counter()

    for kernel_ordinal, begin, end in ranges:
        policy = prepared.policies[kernel_ordinal]
        source_x = int(policy["source_cta_x"])
        bundles = sorted(
            prepared.by_kernel_x.get((kernel_ordinal, source_x), []),
            key=phase_bundle_order,
        )
        require(bundles, f"kernel {kernel_ordinal}: source CTA has no captured bundles")

        source_count = sum(
            int(row["request_ordinal_end_exclusive"])
            - int(row["request_ordinal_begin"])
            for row in bundles
        )
        source_indices = np.empty(source_count, dtype=np.int64)
        bundle_for_record: list[dict[str, int]] = []
        cursor = 0
        for bundle in bundles:
            request_begin = int(bundle["request_ordinal_begin"])
            request_end = int(bundle["request_ordinal_end_exclusive"])
            count = request_end - request_begin
            source_indices[cursor : cursor + count] = np.arange(
                request_begin, request_end, dtype=np.int64
            )
            bundle_for_record.extend([bundle] * count)
            cursor += count
        require(cursor == source_count, "phase-rule source-index census differs")

        original = np.array(source[source_indices], copy=True)
        source_objects = original["object_index"].astype(np.int64)
        require(
            not bool((source_objects >= len(selected_objects)).any()),
            "phase-rule source request references an unknown object",
        )
        require(
            not bool((original["kernel_ordinal"] != kernel_ordinal).any()),
            "phase-rule source request uses another kernel ordinal",
        )
        require(
            not bool((original["operation"] > 1).any()),
            "phase-rule source request has an unsupported operation",
        )
        unknown = original["flags"] & (~KNOWN_REQUEST_FLAGS & 0xFF)
        require(not bool((unknown != 0).any()), "unsupported compact flags")

        lookup = policy_rule_lookup(policy, label="phase")
        rules = list(policy.get("rules") or [])
        rule_ids_by_identity = {id(rule): index for index, rule in enumerate(rules)}
        require(
            len(rule_ids_by_identity) == len(rules),
            "phase-rule policy repeats one rule object",
        )
        source_rule_ids = np.empty(source_count, dtype=np.int32)
        for index, (record, bundle) in enumerate(
            zip(original, bundle_for_record, strict=True)
        ):
            object_index = int(record["object_index"])
            operation_name = OPERATION_NAME[int(record["operation"])]
            base_key = object_index, int(bundle["cta_y"]), int(bundle["cta_z"])
            selector = int(bundle["warp_in_cta"]), int(bundle["warp_program_ordinal"])
            rule = lookup.get((*base_key, operation_name, *selector))
            if rule is None:
                rule = lookup.get((*base_key, None, *selector))
            if rule is None:
                rule = lookup.get((*base_key, operation_name, None, None))
            if rule is None:
                rule = lookup.get((*base_key, None, None, None))
            require(
                rule is not None,
                f"kernel {kernel_ordinal}: no validated phase rule for "
                f"object/yz/operation {(*base_key, operation_name)}",
            )
            source_rule_ids[index] = rule_ids_by_identity[id(rule)]

        selected_mask = selection_map[source_objects]
        rejected = original[~selected_mask]
        target_cta_count = end - begin
        if rejected.size:
            totals["filtered_requests"] += int(rejected.size) * target_cta_count
            totals["filtered_bytes"] += (
                int(rejected["bytes"].sum(dtype=np.uint64)) * target_cta_count
            )
        original = original[selected_mask]
        source_objects = source_objects[selected_mask]
        source_rule_ids = source_rule_ids[selected_mask]
        selected_count = int(original.size)
        local["target_ctas"] += target_cta_count
        local["bundles"] += len(bundles) * target_cta_count
        if selected_count == 0:
            continue

        full_kernel = kernel_base + kernel_ordinal
        require(full_kernel <= 0xFFFF, "full-model kernel ordinal exceeds u16")
        source_rule_coordinates = np.asarray(
            [rule_coordinate(rule, source_x) for rule in rules], dtype=np.int64
        )
        per_cta_bytes = int(original["bytes"].sum(dtype=np.uint64))
        ctas_per_chunk = max(1, chunk_records // selected_count)
        for cta_begin in range(begin, end, ctas_per_chunk):
            cta_end = min(end, cta_begin + ctas_per_chunk)
            delta_by_cta_rule = np.asarray(
                [
                    [rule_coordinate(rule, target_cta_x) for rule in rules]
                    for target_cta_x in range(cta_begin, cta_end)
                ],
                dtype=np.int64,
            )
            delta_by_cta_rule -= source_rule_coordinates
            repeats = cta_end - cta_begin
            output = np.tile(original, repeats)
            objects = np.tile(source_objects, repeats)
            rule_deltas = delta_by_cta_rule[:, source_rule_ids].reshape(-1)
            full_offsets = (
                output["object_offset"].astype(np.int64)
                + rule_deltas
                + bias_map[objects]
            )
            require(
                not bool(((full_offsets < 0) | (full_offsets > 0xFFFFFFFF)).any()),
                "expanded object offset exceeds compact u32",
            )
            full_ends = full_offsets + output["bytes"].astype(np.int64)
            require(
                not bool((full_ends > extent_map[objects]).any()),
                "expanded request escapes its target object extent",
            )
            output["object_index"] = target_map[objects]
            output["object_offset"] = full_offsets.astype(np.uint32)
            output["kernel_ordinal"] = np.uint16(full_kernel)
            payload = output.tobytes(order="C")
            target.write(payload)
            digest.update(payload)
            emitted = selected_count * repeats
            emitted_bytes = per_cta_bytes * repeats
            totals["requests"] += emitted
            totals["bytes"] += emitted_bytes
            local["requests"] += emitted
            local["bytes"] += emitted_bytes

    local["numpy_batched_phase_rules_v1"] = 1
    return dict(sorted(local.items()))


def stream_numpy_causal_phase_generator_segment(
    *,
    prepared: Any,
    ranges: list[tuple[int, int, int]],
    source_path: Path,
    target: BinaryIO,
    kernel_base: int,
    target_objects: list[int],
    selected_objects: list[bool],
    object_biases: list[int],
    target_object_extents: list[int],
    chunk_records: int,
    digest: Any,
    totals: Counter[str],
) -> dict[str, int] | None:
    """Vectorize validated causal-tile loop bodies without changing order."""

    if any(
        str(prepared.policies[kernel_ordinal]["kind"]) != "causal"
        or not causal_is_implemented(prepared.policies[kernel_ordinal])
        for kernel_ordinal, _begin, _end in ranges
    ):
        return None

    import numpy as np

    dtype = np.dtype(
        [
            ("object_index", "<u2"),
            ("object_offset", "<u4"),
            ("kernel_ordinal", "<u2"),
            ("bytes", "<u2"),
            ("operation", "u1"),
            ("flags", "u1"),
        ],
        align=False,
    )
    require(dtype.itemsize == RECORD_BYTES, "NumPy compact dtype is not 12 bytes")
    target_map = np.asarray(target_objects, dtype=np.uint16)
    selection_map = np.asarray(selected_objects, dtype=np.bool_)
    bias_map = np.asarray(object_biases, dtype=np.int64)
    extent_map = np.asarray(target_object_extents, dtype=np.int64)
    local: Counter[str] = Counter()

    with source_path.open("rb") as source:
        for kernel_ordinal, begin, end in ranges:
            policy, lookup, groups = causal_source_programs(
                prepared=prepared,
                source=source,
                kernel_ordinal=kernel_ordinal,
            )
            rules = list(policy.get("rules") or [])
            rule_ids_by_identity = {
                id(rule): index for index, rule in enumerate(rules)
            }
            require(
                len(rule_ids_by_identity) == len(rules),
                "causal policy repeats one rule object",
            )
            source_x = int(policy["source_cta_x"])
            source_rule_coordinates = np.asarray(
                [rule_coordinate(rule, source_x) for rule in rules],
                dtype=np.int64,
            )
            full_kernel = kernel_base + kernel_ordinal
            require(full_kernel <= 0xFFFF, "full-model kernel ordinal exceeds u16")

            def prepare_region(
                rows: list[tuple[dict[str, int], list[tuple[int, ...]], bool]],
            ) -> tuple[Any, Any, Any, int]:
                flat_records: list[tuple[int, ...]] = []
                flat_bundles: list[dict[str, int]] = []
                for bundle, records, _loop in rows:
                    flat_records.extend(records)
                    flat_bundles.extend([bundle] * len(records))
                if not flat_records:
                    return (
                        np.empty(0, dtype=dtype),
                        np.empty(0, dtype=np.int64),
                        np.empty(0, dtype=np.int32),
                        len(rows),
                    )
                original = np.asarray(flat_records, dtype=dtype)
                source_objects = original["object_index"].astype(np.int64)
                require(
                    not bool((source_objects >= len(selected_objects)).any()),
                    "causal source request references an unknown object",
                )
                require(
                    not bool((original["kernel_ordinal"] != kernel_ordinal).any()),
                    "causal source request uses another kernel ordinal",
                )
                require(
                    not bool((original["operation"] > 1).any()),
                    "causal source request has an unsupported operation",
                )
                unknown = original["flags"] & (~KNOWN_REQUEST_FLAGS & 0xFF)
                require(not bool((unknown != 0).any()), "unsupported compact flags")
                rule_ids = np.empty(len(flat_records), dtype=np.int32)
                for index, (record, bundle) in enumerate(
                    zip(original, flat_bundles, strict=True)
                ):
                    object_index = int(record["object_index"])
                    operation_name = OPERATION_NAME[int(record["operation"])]
                    base_key = (
                        object_index,
                        int(bundle["cta_y"]),
                        int(bundle["cta_z"]),
                    )
                    selector = (
                        int(bundle["warp_in_cta"]),
                        int(bundle["warp_program_ordinal"]),
                    )
                    rule = lookup.get((*base_key, operation_name, *selector))
                    if rule is None:
                        rule = lookup.get((*base_key, None, *selector))
                    if rule is None:
                        rule = lookup.get((*base_key, operation_name, None, None))
                    if rule is None:
                        rule = lookup.get((*base_key, None, None, None))
                    require(
                        rule is not None,
                        f"kernel {kernel_ordinal}: no validated causal rule for "
                        f"object/yz/operation {(*base_key, operation_name)}",
                    )
                    rule_ids[index] = rule_ids_by_identity[id(rule)]
                return original, source_objects, rule_ids, len(rows)

            prepared_groups = []
            for group in groups:
                loop_positions = [
                    index
                    for index, (_bundle, _records, is_loop) in enumerate(group)
                    if is_loop
                ]
                require(loop_positions, "causal group has no loop body")
                loop_begin = loop_positions[0]
                loop_end = loop_positions[-1] + 1
                prepared_groups.append(
                    (
                        prepare_region(group[:loop_begin]),
                        prepare_region(group[loop_begin:loop_end]),
                        prepare_region(group[loop_end:]),
                    )
                )

            def emit_region(
                region: tuple[Any, Any, Any, int], coordinates: list[int]
            ) -> None:
                original, source_objects, source_rule_ids, bundle_count = region
                repeats_total = len(coordinates)
                if repeats_total == 0:
                    return
                local["bundles"] += bundle_count * repeats_total
                selected_mask = selection_map[source_objects]
                rejected = original[~selected_mask]
                if rejected.size:
                    totals["filtered_requests"] += int(rejected.size) * repeats_total
                    totals["filtered_bytes"] += (
                        int(rejected["bytes"].sum(dtype=np.uint64)) * repeats_total
                    )
                selected = original[selected_mask]
                objects_once = source_objects[selected_mask]
                rule_ids_once = source_rule_ids[selected_mask]
                selected_count = int(selected.size)
                if selected_count == 0:
                    return
                per_repeat_bytes = int(selected["bytes"].sum(dtype=np.uint64))
                repeats_per_chunk = max(1, chunk_records // selected_count)
                for coordinate_begin in range(0, repeats_total, repeats_per_chunk):
                    coordinate_chunk = coordinates[
                        coordinate_begin : coordinate_begin + repeats_per_chunk
                    ]
                    delta_by_coordinate_rule = np.asarray(
                        [
                            [rule_coordinate(rule, coordinate) for rule in rules]
                            for coordinate in coordinate_chunk
                        ],
                        dtype=np.int64,
                    )
                    delta_by_coordinate_rule -= source_rule_coordinates
                    repeats = len(coordinate_chunk)
                    output = np.tile(selected, repeats)
                    objects = np.tile(objects_once, repeats)
                    rule_deltas = delta_by_coordinate_rule[
                        :, rule_ids_once
                    ].reshape(-1)
                    full_offsets = (
                        output["object_offset"].astype(np.int64)
                        + rule_deltas
                        + bias_map[objects]
                    )
                    require(
                        not bool(
                            ((full_offsets < 0) | (full_offsets > 0xFFFFFFFF)).any()
                        ),
                        "expanded object offset exceeds compact u32",
                    )
                    full_ends = full_offsets + output["bytes"].astype(np.int64)
                    require(
                        not bool((full_ends > extent_map[objects]).any()),
                        "expanded request escapes its target object extent",
                    )
                    output["object_index"] = target_map[objects]
                    output["object_offset"] = full_offsets.astype(np.uint32)
                    output["kernel_ordinal"] = np.uint16(full_kernel)
                    payload = output.tobytes(order="C")
                    target.write(payload)
                    digest.update(payload)
                    emitted = selected_count * repeats
                    emitted_bytes = per_repeat_bytes * repeats
                    totals["requests"] += emitted
                    totals["bytes"] += emitted_bytes
                    local["requests"] += emitted
                    local["bytes"] += emitted_bytes

            for target_cta_x in range(begin, end):
                local["target_ctas"] += 1
                for prefix, loop, suffix in prepared_groups:
                    emit_region(prefix, [target_cta_x])
                    emit_region(loop, list(range(target_cta_x, -1, -1)))
                    emit_region(suffix, [target_cta_x])

    local["numpy_batched_causal_v1"] = 1
    return dict(sorted(local.items()))


def stream_python_segment(
    *,
    source: BinaryIO,
    target: BinaryIO,
    request_begin: int,
    request_end: int,
    kernel_base: int,
    target_objects: list[int],
    selected_objects: list[bool],
    object_offset_deltas: list[int],
    target_object_extents: list[int],
    chunk_records: int,
    digest: Any,
    totals: Counter[str],
) -> None:
    source.seek(request_begin * RECORD_BYTES)
    remaining = request_end - request_begin
    while remaining:
        take = min(remaining, chunk_records)
        payload = source.read(take * RECORD_BYTES)
        require(len(payload) == take * RECORD_BYTES, "template binary is truncated")
        output = bytearray()
        for record in RECORD_STRUCT.iter_unpack(payload):
            object_index, object_offset, kernel_ordinal, byte_count, operation, flags = record
            require(known_request_flags(flags), "unsupported compact flags")
            if not selected_objects[object_index]:
                totals["filtered_requests"] += 1
                totals["filtered_bytes"] += byte_count
                continue
            rebound_offset = object_offset + object_offset_deltas[object_index]
            require(0 <= rebound_offset <= 0xFFFFFFFF,
                    "expanded object offset exceeds compact u32")
            require(
                rebound_offset + byte_count <= target_object_extents[object_index],
                "expanded request escapes its target object extent",
            )
            full_kernel = kernel_base + kernel_ordinal
            require(full_kernel <= 0xFFFF, "full-model kernel ordinal exceeds u16")
            output.extend(
                RECORD_STRUCT.pack(
                    target_objects[object_index],
                    rebound_offset,
                    full_kernel,
                    byte_count,
                    operation,
                    flags,
                )
            )
            totals["requests"] += 1
            totals["bytes"] += byte_count
        target.write(output)
        digest.update(output)
        remaining -= take


def stream_numpy_segment(
    *,
    source_path: Path,
    target: BinaryIO,
    request_begin: int,
    request_end: int,
    kernel_base: int,
    target_objects: list[int],
    selected_objects: list[bool],
    object_offset_deltas: list[int],
    target_object_extents: list[int],
    chunk_records: int,
    digest: Any,
    totals: Counter[str],
) -> None:
    import numpy as np

    dtype = np.dtype(
        [
            ("object_index", "<u2"),
            ("object_offset", "<u4"),
            ("kernel_ordinal", "<u2"),
            ("bytes", "<u2"),
            ("operation", "u1"),
            ("flags", "u1"),
        ],
        align=False,
    )
    require(dtype.itemsize == RECORD_BYTES, "NumPy compact dtype is not 12 bytes")
    source = np.memmap(source_path, mode="r", dtype=dtype)
    target_map = np.asarray(target_objects, dtype=np.uint16)
    selection_map = np.asarray(selected_objects, dtype=np.bool_)
    delta_map = np.asarray(object_offset_deltas, dtype=np.int64)
    extent_map = np.asarray(target_object_extents, dtype=np.int64)
    for begin in range(request_begin, request_end, chunk_records):
        end = min(request_end, begin + chunk_records)
        original = source[begin:end]
        mask = selection_map[original["object_index"]]
        if not bool(mask.all()):
            rejected = original[~mask]
            totals["filtered_requests"] += int(rejected.size)
            totals["filtered_bytes"] += int(rejected["bytes"].sum(dtype=np.uint64))
            original = original[mask]
        output = np.array(original, copy=True)
        unknown = output["flags"] & (~KNOWN_REQUEST_FLAGS & 0xFF)
        require(not bool((unknown != 0).any()), "unsupported compact flags")
        full_kernels = output["kernel_ordinal"].astype(np.uint32) + kernel_base
        full_offsets = (
            output["object_offset"].astype(np.int64)
            + delta_map[output["object_index"]]
        )
        require(not bool((full_kernels > 0xFFFF).any()),
                "full-model kernel ordinal exceeds u16")
        require(
            not bool(((full_offsets < 0) | (full_offsets > 0xFFFFFFFF)).any()),
            "expanded object offset exceeds compact u32",
        )
        full_ends = full_offsets + output["bytes"].astype(np.int64)
        require(
            not bool(
                (full_ends > extent_map[output["object_index"]]).any()
            ),
            "expanded request escapes its target object extent",
        )
        output["object_index"] = target_map[output["object_index"]]
        output["object_offset"] = full_offsets.astype(np.uint32)
        output["kernel_ordinal"] = full_kernels.astype(np.uint16)
        payload = output.tobytes(order="C")
        target.write(payload)
        digest.update(payload)
        totals["requests"] += int(output.size)
        totals["bytes"] += int(output["bytes"].sum(dtype=np.uint64))


def stream_numpy_order_skeleton_segment(
    *,
    template: dict[str, Any],
    template_path: Path,
    skeleton_manifest: dict[str, Any],
    skeleton_path: Path,
    target: BinaryIO,
    kernel_base: int,
    target_objects: list[int],
    selected_objects: list[bool],
    object_biases: list[int],
    object_strides: list[int],
    target_object_extents: list[int],
    chunk_events: int,
    digest: Any,
    totals: Counter[str],
) -> None:
    """Expand one order skeleton and rebind it directly into a full plan."""
    import numpy as np

    request_dtype = np.dtype(
        [
            ("object_index", "<u2"),
            ("object_offset", "<u4"),
            ("kernel_ordinal", "<u2"),
            ("bytes", "<u2"),
            ("operation", "u1"),
            ("flags", "u1"),
        ],
        align=False,
    )
    skeleton_dtype = np.dtype(
        [("bundle_index", "<u2"), ("group_instance", "<u2")],
        align=False,
    )
    require(request_dtype.itemsize == RECORD_BYTES,
            "NumPy compact dtype is not 12 bytes")
    require(skeleton_dtype.itemsize == SKELETON_RECORD_BYTES,
            "NumPy skeleton dtype is not four bytes")
    template_requests = int(template["requests"])
    begins, lengths = _bundle_ranges(template, template_requests)
    records = np.memmap(template_path, mode="r", dtype=request_dtype)
    skeleton = np.memmap(skeleton_path, mode="r", dtype=skeleton_dtype)
    require(int(skeleton.size) == int(skeleton_manifest["events"]),
            "order-skeleton event count differs")
    unknown = records["flags"] & (~KNOWN_REQUEST_FLAGS & 0xFF)
    require(not bool((unknown != 0).any()), "unsupported compact flags")
    target_map = np.asarray(target_objects, dtype=np.uint16)
    selection_map = np.asarray(selected_objects, dtype=np.bool_)
    bias_map = np.asarray(object_biases, dtype=np.int64)
    stride_map = np.asarray(object_strides, dtype=np.int64)
    extent_map = np.asarray(target_object_extents, dtype=np.int64)
    emitted_before = totals["requests"]
    filtered_before = totals["filtered_requests"]
    expanded_before_filter = 0

    for begin_event in range(0, int(skeleton.size), chunk_events):
        events = skeleton[begin_event : begin_event + chunk_events]
        bundle_indices = events["bundle_index"].astype(np.int64)
        require(not bool((bundle_indices >= begins.size).any()),
                "order skeleton bundle index escapes template")
        counts = lengths[bundle_indices].astype(np.int64)
        output_count = int(counts.sum())
        require(output_count > 0, "order-skeleton chunk emits no requests")
        expanded_before_filter += output_count
        repeated_events = np.repeat(np.arange(events.size, dtype=np.int64), counts)
        output_starts = np.cumsum(counts, dtype=np.int64) - counts
        within_bundle = np.arange(output_count, dtype=np.int64) - np.repeat(
            output_starts, counts
        )
        request_indices = begins[bundle_indices[repeated_events]].astype(np.int64)
        request_indices += within_bundle
        output = np.array(records[request_indices], copy=True)
        groups = events["group_instance"][repeated_events].astype(np.int64)
        original_objects = output["object_index"].astype(np.int64)
        mask = selection_map[original_objects]
        if not bool(mask.all()):
            rejected = output[~mask]
            totals["filtered_requests"] += int(rejected.size)
            totals["filtered_bytes"] += int(rejected["bytes"].sum(dtype=np.uint64))
            output = output[mask]
            groups = groups[mask]
            original_objects = original_objects[mask]
        full_offsets = (
            output["object_offset"].astype(np.int64)
            + bias_map[original_objects]
            + stride_map[original_objects] * groups
        )
        full_kernels = output["kernel_ordinal"].astype(np.uint32) + kernel_base
        require(not bool((full_kernels > 0xFFFF).any()),
                "full-model kernel ordinal exceeds u16")
        require(
            not bool(((full_offsets < 0) | (full_offsets > 0xFFFFFFFF)).any()),
            "expanded object offset exceeds compact u32",
        )
        full_ends = full_offsets + output["bytes"].astype(np.int64)
        require(not bool((full_ends > extent_map[original_objects]).any()),
                "expanded request escapes its target object extent")
        output["object_index"] = target_map[original_objects]
        output["object_offset"] = full_offsets.astype(np.uint32)
        output["kernel_ordinal"] = full_kernels.astype(np.uint16)
        payload = output.tobytes(order="C")
        target.write(payload)
        digest.update(payload)
        totals["requests"] += int(output.size)
        totals["bytes"] += int(output["bytes"].sum(dtype=np.uint64))

    repeat_count = int((skeleton_manifest.get("geometry") or {})["repeat_count"])
    require(
        expanded_before_filter == template_requests * repeat_count,
        "order skeleton does not conserve expanded template requests",
    )
    require(
        totals["requests"] - emitted_before
        == expanded_before_filter
        - (totals["filtered_requests"] - filtered_before),
        "order-skeleton filtered request census differs",
    )


def stream_plan(
    *,
    plan_path: Path,
    output_path: Path | None,
    output_manifest_path: Path,
    output_stream: BinaryIO | None = None,
    shared_template_binary: Path | None = None,
    include_kinds: set[str] | None = None,
    segment_ids: set[str] | None = None,
    chunk_records: int = 1_000_000,
    backend: str = "auto",
    runtime_parameters: dict[str, int] | None = None,
) -> dict[str, Any]:
    require(chunk_records > 0, "chunk record count must be positive")
    require(backend in {"auto", "python", "numpy"}, "unsupported backend")
    plan_path = plan_path.resolve()
    require(
        (output_path is None) != (output_stream is None),
        "provide exactly one output path or output stream",
    )
    if output_path is not None:
        output_path = output_path.resolve()
    output_manifest_path = output_manifest_path.resolve()
    plan = load_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported lazy-plan schema")
    segments = selected_segments(plan, segment_ids)
    runtime_parameters = {
        str(name): int(value) for name, value in (runtime_parameters or {}).items()
    }
    declared_parameters = declared_runtime_parameters(plan)
    require(
        set(runtime_parameters) <= set(declared_parameters),
        f"unknown runtime parameters: "
        f"{sorted(set(runtime_parameters) - set(declared_parameters))}",
    )
    if backend == "auto":
        try:
            import numpy  # noqa: F401
        except ImportError:
            backend = "python"
        else:
            backend = "numpy"

    plan_objects = plan.get("objects")
    require(isinstance(plan_objects, list), "plan object table is malformed")
    require(
        [int(item["target_object_index"]) for item in plan_objects]
        == list(range(len(plan_objects))),
        "plan target object indices are not dense",
    )
    target_extents_by_index = [int(item["bytes"]) for item in plan_objects]

    started = time.perf_counter()
    digest = hashlib.sha256()
    totals: Counter[str] = Counter()
    segment_results = []
    phase_generator_cache: dict[
        str, tuple[Any, list[tuple[int, int, int]], dict[str, Any]]
    ] = {}
    template_digest_cache: dict[Path, str] = {}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target_context = output_path.open("wb")
        output_label = str(output_path)
    else:
        assert output_stream is not None
        target_context = nullcontext(output_stream)
        output_label = "stream"
    with target_context as target:
        for segment in segments:
            template, binary_path = segment_template(
                plan, segment, shared_template_binary
            )
            template_descriptor = segment.get("template") or plan.get("template")
            assert isinstance(template_descriptor, dict)
            template_manifest_path = resolve_input(
                str(template_descriptor["manifest"])
            )
            source_kinds = source_object_kinds(template)
            (
                target_objects,
                selected_objects,
                object_biases,
                target_object_extents,
            ) = binding_vectors(
                segment=segment,
                source_kinds=source_kinds,
                include_kinds=include_kinds,
                runtime_parameters=runtime_parameters,
                target_extents_by_index=target_extents_by_index,
            )
            repeat_count, object_strides = segment_repetition(
                segment, len(source_kinds)
            )
            request_begin = int(segment.get("template_request_begin", 0))
            request_end = int(
                segment.get("template_request_end_exclusive", template["requests"])
            )
            require(0 <= request_begin <= request_end <= int(template["requests"]),
                    "segment request interval escapes its template")
            kernel_base = int(segment["expanded_kernel_ordinal_begin"])
            before = Counter(totals)
            phase_generator = segment_phase_generator(
                segment=segment,
                template=template,
                template_path=binary_path,
                cache=phase_generator_cache,
            )
            order_skeleton = segment_order_skeleton(
                segment=segment,
                template_manifest_path=template_manifest_path,
                repeat_count=repeat_count,
            )
            require(
                phase_generator is None or order_skeleton is None,
                "phase-aware generation and order-skeleton replay cannot share a segment",
            )
            generator_result = None
            generator_provenance = None
            if phase_generator is not None:
                require(repeat_count == 1 and not any(object_strides),
                        "phase-aware generated segments cannot also use repeat strides")
                require(request_begin == 0 and request_end == int(template["requests"]),
                        "phase-aware generated segments cannot slice their source template")
                prepared, ranges, generator_provenance = phase_generator
                if backend == "numpy":
                    generator_result = stream_numpy_tiled_phase_generator_segment(
                        prepared=prepared,
                        ranges=ranges,
                        source_path=binary_path,
                        target=target,
                        kernel_base=kernel_base,
                        target_objects=target_objects,
                        selected_objects=selected_objects,
                        object_biases=object_biases,
                        target_object_extents=target_object_extents,
                        chunk_records=chunk_records,
                        digest=digest,
                        totals=totals,
                    )
                    if generator_result is None:
                        generator_result = stream_numpy_phase_rules_segment(
                            prepared=prepared,
                            ranges=ranges,
                            source_path=binary_path,
                            target=target,
                            kernel_base=kernel_base,
                            target_objects=target_objects,
                            selected_objects=selected_objects,
                            object_biases=object_biases,
                            target_object_extents=target_object_extents,
                            chunk_records=chunk_records,
                            digest=digest,
                            totals=totals,
                        )
                    if generator_result is None:
                        generator_result = stream_numpy_causal_phase_generator_segment(
                            prepared=prepared,
                            ranges=ranges,
                            source_path=binary_path,
                            target=target,
                            kernel_base=kernel_base,
                            target_objects=target_objects,
                            selected_objects=selected_objects,
                            object_biases=object_biases,
                            target_object_extents=target_object_extents,
                            chunk_records=chunk_records,
                            digest=digest,
                            totals=totals,
                        )
                if generator_result is None:
                    generator_result = stream_phase_generator_segment(
                        prepared=prepared,
                        ranges=ranges,
                        source_path=binary_path,
                        target=target,
                        kernel_base=kernel_base,
                        target_objects=target_objects,
                        selected_objects=selected_objects,
                        object_biases=object_biases,
                        target_object_extents=target_object_extents,
                        chunk_records=chunk_records,
                        digest=digest,
                        totals=totals,
                    )
            elif order_skeleton is not None:
                require(backend == "numpy",
                        "order-skeleton streaming requires the NumPy backend")
                require(request_begin == 0 and request_end == int(template["requests"]),
                        "order-skeleton segments cannot slice their address program")
                skeleton_manifest, skeleton_path = order_skeleton
                stream_numpy_order_skeleton_segment(
                    template=template,
                    template_path=binary_path,
                    skeleton_manifest=skeleton_manifest,
                    skeleton_path=skeleton_path,
                    target=target,
                    kernel_base=kernel_base,
                    target_objects=target_objects,
                    selected_objects=selected_objects,
                    object_biases=object_biases,
                    object_strides=object_strides,
                    target_object_extents=target_object_extents,
                    chunk_events=chunk_records,
                    digest=digest,
                    totals=totals,
                )
            else:
                for instance in range(repeat_count):
                    object_offset_deltas = [
                        bias + stride * instance
                        for bias, stride in zip(object_biases, object_strides)
                    ]
                    if backend == "numpy":
                        stream_numpy_segment(
                            source_path=binary_path,
                            target=target,
                            request_begin=request_begin,
                            request_end=request_end,
                            kernel_base=kernel_base,
                            target_objects=target_objects,
                            selected_objects=selected_objects,
                            object_offset_deltas=object_offset_deltas,
                            target_object_extents=target_object_extents,
                            chunk_records=chunk_records,
                            digest=digest,
                            totals=totals,
                        )
                    else:
                        with binary_path.open("rb") as source:
                            stream_python_segment(
                                source=source,
                                target=target,
                                request_begin=request_begin,
                                request_end=request_end,
                                kernel_base=kernel_base,
                                target_objects=target_objects,
                                selected_objects=selected_objects,
                                object_offset_deltas=object_offset_deltas,
                                target_object_extents=target_object_extents,
                                chunk_records=chunk_records,
                                digest=digest,
                                totals=totals,
                            )
            emitted_requests = totals["requests"] - before["requests"]
            emitted_bytes = totals["bytes"] - before["bytes"]
            expected_census = segment.get("expected_census")
            if expected_census is not None and include_kinds is None:
                require(
                    emitted_requests == int(expected_census["requests"]),
                    f"segment {segment['segment_id']} request census differs",
                )
                require(
                    emitted_bytes == int(expected_census["bytes"]),
                    f"segment {segment['segment_id']} byte census differs",
                )
            segment_results.append(
                {
                    "segment_id": segment["segment_id"],
                    "requests": emitted_requests,
                    "bytes": emitted_bytes,
                    "filtered_requests": (
                        totals["filtered_requests"] - before["filtered_requests"]
                    ),
                    "repeat_count": repeat_count,
                    "template_binary": str(binary_path),
                    "template_binary_sha256": (
                        template_digest_cache[binary_path]
                        if binary_path in template_digest_cache
                        else template_digest_cache.setdefault(
                            binary_path, sha256_file(binary_path)
                        )
                    ),
                    "order_skeleton": (
                        str(order_skeleton[1]) if order_skeleton is not None else None
                    ),
                    "generator": generator_provenance,
                    "generator_totals": generator_result,
                    "expected_census": expected_census,
                    "expected_census_matches": (
                        expected_census is not None and include_kinds is None
                    ),
                }
            )
    all_segments_selected = segment_ids is None
    plan_expected_census = plan.get("expected_census")
    if (
        plan_expected_census is not None
        and all_segments_selected
        and include_kinds is None
    ):
        require(
            totals["requests"] == int(plan_expected_census["requests"]),
            "complete plan request census differs",
        )
        require(
            totals["bytes"] == int(plan_expected_census["bytes"]),
            "complete plan byte census differs",
        )
    emitted_binary_bytes = totals["requests"] * RECORD_BYTES
    output_materialized = False
    if output_path is not None:
        output_stat = output_path.stat()
        output_materialized = stat.S_ISREG(output_stat.st_mode)
        if output_materialized:
            require(output_stat.st_size == emitted_binary_bytes,
                    "streamed binary size does not conserve emitted requests")
    elapsed = time.perf_counter() - started
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "streamed full-plan target-object compact requests",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "output": output_label,
        "output_materialized": output_materialized,
        "output_sha256": digest.hexdigest(),
        "output_binary_bytes": emitted_binary_bytes,
        "record_bytes": RECORD_BYTES,
        "backend": backend,
        "include_kinds": sorted(include_kinds) if include_kinds is not None else None,
        "runtime_parameters": runtime_parameters,
        "declared_runtime_parameters": declared_parameters,
        "expected_census": plan_expected_census,
        "expected_census_matches": (
            plan_expected_census is not None
            and all_segments_selected
            and include_kinds is None
        ),
        "segments": segment_results,
        "totals": dict(sorted(totals.items())),
        "elapsed_seconds": elapsed,
        "records_per_second": totals["requests"] / elapsed if elapsed else None,
        "not_claimed": [
            "per-request production timestamps",
            "post-cache or HBF physical traffic",
            "segments absent from the input plan",
        ],
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="compact binary path, or '-' to stream records to stdout",
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--shared-template-binary", type=Path)
    parser.add_argument("--include-kind", action="append")
    parser.add_argument("--segment-id", action="append")
    parser.add_argument("--chunk-records", type=int, default=1_000_000)
    parser.add_argument("--backend", choices=("auto", "python", "numpy"), default="auto")
    parser.add_argument("--runtime-parameter", action="append", default=[])
    args = parser.parse_args()
    runtime_parameters: dict[str, int] = {}
    for raw in args.runtime_parameter:
        name, separator, value = raw.rpartition("=")
        require(bool(separator and name and value),
                f"runtime parameter must be NAME=INTEGER: {raw}")
        require(name not in runtime_parameters, f"duplicate runtime parameter {name}")
        runtime_parameters[name] = int(value)
    output_path = None if args.output == "-" else Path(args.output)
    result = stream_plan(
        plan_path=args.plan,
        output_path=output_path,
        output_manifest_path=args.output_manifest,
        output_stream=sys.stdout.buffer if output_path is None else None,
        shared_template_binary=args.shared_template_binary,
        include_kinds=set(args.include_kind) if args.include_kind else None,
        segment_ids=set(args.segment_id) if args.segment_id else None,
        chunk_records=args.chunk_records,
        backend=args.backend,
        runtime_parameters=runtime_parameters,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "backend": result["backend"],
                "requests": result["totals"]["requests"],
                "bytes": result["totals"]["bytes"],
                "elapsed_seconds": result["elapsed_seconds"],
                "records_per_second": result["records_per_second"],
            },
            sort_keys=True,
        ),
        file=sys.stderr if output_path is None else sys.stdout,
    )


if __name__ == "__main__":
    main()

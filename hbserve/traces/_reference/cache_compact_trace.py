#!/usr/bin/env python3
"""Apply the named reference GPU cache to a compact request stream.

Input and output use the 12-byte compact request record.  The command accepts
stdin/stdout, so a lazy full-model plan can feed cache/HBF evaluation without
ever materializing the expanded pre-cache trace.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, BinaryIO

from hbserve.traces._reference.compact_request_template import (
    OPERATION_CODE,
    OPERATION_NAME,
    RECORD_BYTES,
    RECORD_STRUCT,
    REQUEST_FLAG_EVICT_FIRST,
    known_request_flags,
    load_json,
    require,
    sha256_file,
)
from hbserve.traces._reference.full_model_trace_plan import PLAN_SCHEMA
from hbserve.traces._reference.gpu_request_transform import ReferenceLRUCache, SectorRequest


SCHEMA = {"name": "hbfsim.compact_cache_transform", "version": 1}


def open_input(path: Path | None, stream: BinaryIO | None):
    require((path is None) != (stream is None), "provide exactly one input path or stream")
    return path.open("rb") if path is not None else nullcontext(stream)


def open_output(path: Path | None, stream: BinaryIO | None):
    require((path is None) != (stream is None), "provide exactly one output path or stream")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("wb")
    return nullcontext(stream)


def cache_request(
    *, object_row: dict[str, Any], object_offset: int, byte_count: int,
    operation: int, kernel_ordinal: int, sequence: int, flags: int
) -> SectorRequest:
    base = int(object_row["logical_address"])
    object_id = str(object_row["object_id"])
    return SectorRequest(
        instruction_group_id=f"compact-{sequence}",
        object_id=object_id,
        raw_extent_id=object_id,
        kind=str(object_row["kind"]),
        op=OPERATION_NAME[operation],
        raw_address=base + object_offset,
        logical_address=base + object_offset,
        object_offset=object_offset,
        byte_count=byte_count,
        issue_ns=None,
        dependencies=(),
        source_lane_event_ids=(),
        source_lane_bytes=byte_count,
        metadata={
            "kernel_ordinal": kernel_ordinal,
            "opcode": (
                "LDG.E.EF"
                if flags & REQUEST_FLAG_EVICT_FIRST
                else "compact-no-cache-hint"
            ),
        },
    )


def transform_compact(
    *,
    plan_path: Path,
    input_path: Path | None,
    input_stream: BinaryIO | None,
    output_path: Path | None,
    output_stream: BinaryIO | None,
    manifest_path: Path,
    cache_mode: str,
    capacity_bytes: int,
    line_bytes: int,
    sector_bytes: int,
    associativity: int,
    write_policy: str,
    write_allocate: bool,
    write_miss_fetch: bool,
    final_drain: bool,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = load_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported lazy-plan schema")
    objects = plan.get("objects")
    require(isinstance(objects, list) and objects, "plan has no object table")
    require(
        [int(item["target_object_index"]) for item in objects]
        == list(range(len(objects))),
        "plan target object indices are not dense",
    )
    object_index_by_id = {
        str(item["object_id"]): int(item["target_object_index"])
        for item in objects
    }
    require(cache_mode in {"bypass", "reference-lru"}, "unsupported cache mode")
    cache = None
    if cache_mode == "reference-lru":
        cache = ReferenceLRUCache(
            capacity_bytes=capacity_bytes,
            line_bytes=line_bytes,
            sector_bytes=sector_bytes,
            associativity=associativity,
            write_policy=write_policy,
            write_allocate=write_allocate,
            read_admission="sass-ef-lru",
            address_space="logical",
            write_miss_fetch=write_miss_fetch,
        )

    counts: Counter[str] = Counter()
    input_digest = hashlib.sha256()
    digest = hashlib.sha256()
    started = time.perf_counter()
    remainder = b""
    last_kernel = 0

    def emit(target: BinaryIO, request: SectorRequest, operation: str, kernel: int, reason: str) -> None:
        object_index = object_index_by_id[request.object_id]
        payload = RECORD_STRUCT.pack(
            object_index,
            int(request.object_offset),
            kernel,
            int(request.byte_count),
            OPERATION_CODE[operation],
            0,
        )
        target.write(payload)
        digest.update(payload)
        counts["output_requests"] += 1
        counts["output_bytes"] += int(request.byte_count)
        counts[f"output_{operation.lower()}_requests"] += 1
        counts[f"output_{operation.lower()}_bytes"] += int(request.byte_count)
        counts[f"reason_{reason}_requests"] += 1

    with open_input(input_path, input_stream) as source, open_output(
        output_path, output_stream
    ) as target:
        assert source is not None and target is not None
        while payload := source.read(RECORD_BYTES * 65536):
            payload = remainder + payload
            usable = len(payload) // RECORD_BYTES * RECORD_BYTES
            remainder = payload[usable:]
            input_digest.update(payload[:usable])
            for record in RECORD_STRUCT.iter_unpack(payload[:usable]):
                object_index, object_offset, kernel, byte_count, operation, flags = record
                require(known_request_flags(flags), "unsupported compact flags")
                require(
                    not (flags & REQUEST_FLAG_EVICT_FIRST) or operation == 0,
                    "evict-first flag is only valid for reads",
                )
                require(object_index < len(objects), "compact request escapes object table")
                obj = objects[object_index]
                require(
                    object_offset + byte_count <= int(obj["bytes"]),
                    f"request escapes object {obj['object_id']}",
                )
                require(operation in OPERATION_NAME, "unsupported compact operation")
                counts["input_requests"] += 1
                counts["input_bytes"] += byte_count
                counts[f"input_{OPERATION_NAME[operation].lower()}_bytes"] += byte_count
                last_kernel = kernel
                request = cache_request(
                    object_row=obj,
                    object_offset=object_offset,
                    byte_count=byte_count,
                    operation=operation,
                    kernel_ordinal=kernel,
                    sequence=counts["input_requests"] - 1,
                    flags=flags,
                )
                if cache is None:
                    emit(target, request, request.op, kernel, "bypass")
                    continue
                for reason, downstream in cache.access(request):
                    downstream_op = downstream.op
                    if reason == "dirty-eviction":
                        downstream_op = "W"
                    elif reason == "write-allocate-fill":
                        downstream_op = "R"
                    emit(target, downstream, downstream_op, kernel, reason)
        require(not remainder, "truncated compact input record")
        if cache is not None and final_drain:
            for request in cache.drain():
                emit(target, request, "W", last_kernel, "final-dirty-drain")

    elapsed = time.perf_counter() - started
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "streaming named-reference-cache compact request transform",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "input": str(input_path) if input_path is not None else "stream",
        "input_sha256": input_digest.hexdigest(),
        "output": str(output_path) if output_path is not None else "stream",
        "output_sha256": digest.hexdigest(),
        "record_bytes": RECORD_BYTES,
        "counts": dict(sorted(counts.items())),
        "cache": {
            "mode": cache_mode,
            "capacity_bytes": capacity_bytes if cache is not None else 0,
            "line_bytes": line_bytes if cache is not None else 0,
            "sector_bytes": sector_bytes if cache is not None else 0,
            "associativity": associativity if cache is not None else 0,
            "write_policy": write_policy if cache is not None else "none",
            "read_admission": "sass-ef-lru" if cache is not None else "none",
            "write_allocate": write_allocate if cache is not None else False,
            "write_miss_fetch": write_miss_fetch if cache is not None else False,
            "final_drain": final_drain if cache is not None else False,
            "stats": dict(sorted(cache.stats.items())) if cache is not None else {},
            "final_state": cache.state_summary() if cache is not None else {},
        },
        "elapsed_seconds": elapsed,
        "input_records_per_second": counts["input_requests"] / elapsed if elapsed else None,
        "not_claimed": [
            "cycle-accurate NVIDIA cache behavior",
            "production request issue timestamps",
            "physical HBF timing",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", required=True, help="compact input path or '-'")
    parser.add_argument("--output", required=True, help="compact output path or '-'")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-mode", choices=("bypass", "reference-lru"), default="reference-lru")
    parser.add_argument("--capacity-bytes", type=int, default=40 * 1024 * 1024)
    parser.add_argument("--line-bytes", type=int, default=128)
    parser.add_argument("--sector-bytes", type=int, default=32)
    parser.add_argument("--associativity", type=int, default=16)
    parser.add_argument("--write-policy", choices=("write-through", "write-back"), default="write-back")
    parser.add_argument("--no-write-allocate", action="store_true")
    parser.add_argument("--no-write-miss-fetch", action="store_true")
    parser.add_argument("--final-drain", action="store_true")
    args = parser.parse_args()
    input_path = None if args.input == "-" else Path(args.input)
    output_path = None if args.output == "-" else Path(args.output)
    result = transform_compact(
        plan_path=args.plan,
        input_path=input_path,
        input_stream=sys.stdin.buffer if input_path is None else None,
        output_path=output_path,
        output_stream=sys.stdout.buffer if output_path is None else None,
        manifest_path=args.manifest,
        cache_mode=args.cache_mode,
        capacity_bytes=args.capacity_bytes,
        line_bytes=args.line_bytes,
        sector_bytes=args.sector_bytes,
        associativity=args.associativity,
        write_policy=args.write_policy,
        write_allocate=not args.no_write_allocate,
        write_miss_fetch=not args.no_write_miss_fetch,
        final_drain=args.final_drain,
    )
    print(
        json.dumps(
            {"status": result["status"], **result["counts"],
             "elapsed_seconds": result["elapsed_seconds"]},
            sort_keys=True,
        ),
        file=sys.stderr if output_path is None else sys.stdout,
    )


if __name__ == "__main__":
    main()

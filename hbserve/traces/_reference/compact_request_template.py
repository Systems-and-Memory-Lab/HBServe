#!/usr/bin/env python3
"""Compile a routed text trace into a placement-neutral compact request template.

The existing HBFSim frontend keeps a JSON object for every 32-byte request.
That representation is useful as an auditable interchange format, but it is
too large to repeat for every transformer layer.  This module reverses the
named routing map, resolves each request to ``(object, object offset)``, and
stores one fixed-width record:

``object_index:u16, object_offset:u32, kernel_ordinal:u16, bytes:u16,
operation:u8, flags:u8``.

The binary is still one complete representative-layer request stream.  A
separate full-model plan can bind the object-relative records to every layer
without copying the template bytes.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import time
from typing import Any, BinaryIO, Iterable


SCHEMA = {"name": "hbfsim.compact_gpu_request_template", "version": 1}
SASS_FLAG_SCHEMA = {"name": "hbfsim.compact_gpu_request_template", "version": 2}
SUPPORTED_SCHEMAS = (SCHEMA, SASS_FLAG_SCHEMA)
RECORD_STRUCT = struct.Struct("<HIHHBB")
RECORD_BYTES = RECORD_STRUCT.size
OPERATION_CODE = {"R": 0, "W": 1}
OPERATION_NAME = {value: key for key, value in OPERATION_CODE.items()}
REQUEST_FLAG_EVICT_FIRST = 1 << 0
KNOWN_REQUEST_FLAGS = REQUEST_FLAG_EVICT_FIRST
LINE_RE = re.compile(
    rb"^TX id=device_(\d+) target=([A-Z_]+) op=([RW]) "
    rb"addr=(\d+) bytes=(\d+) issue_ns=([^ ]+) duration_ns=([^ ]+) "
    rb"deps=([^ ]+) stack=([^\r\n]+)\r?\n?$"
)


class CompactTemplateError(ValueError):
    """Raised when compilation would guess or lose a request."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise CompactTemplateError(message)


def request_flags(operation: str, opcode: str) -> int:
    """Encode cache-relevant SASS suffixes without changing the 12-B layout."""

    require(operation in OPERATION_CODE, "unsupported compact operation")
    tokens = str(opcode).upper().split(".")
    return (
        REQUEST_FLAG_EVICT_FIRST
        if operation == "R" and "EF" in tokens
        else 0
    )


def known_request_flags(flags: int) -> bool:
    return 0 <= flags <= 0xFF and flags & ~KNOWN_REQUEST_FLAGS == 0


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompactTemplateError(f"cannot read JSON object {path}") from error
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def normalized_objects(summary: dict[str, Any]) -> list[dict[str, Any]]:
    raw = summary.get("objects")
    require(isinstance(raw, list) and raw, "object summary has no objects")
    objects = []
    seen_ids: set[str] = set()
    previous_end = 0
    for index, source in enumerate(
        sorted(raw, key=lambda item: int(item["logical_address"]))
    ):
        require(isinstance(source, dict), "malformed object summary row")
        object_id = source.get("object_id")
        kind = source.get("kind")
        name = source.get("source_name")
        base = source.get("logical_address")
        byte_count = source.get("object_bytes")
        require(
            isinstance(object_id, str)
            and object_id
            and object_id not in seen_ids,
            f"duplicate or malformed object id {object_id!r}",
        )
        require(kind in {"weight", "kv_cache", "activation", "anonymous"},
                f"unsupported object kind {kind!r}")
        require(isinstance(name, str) and name, f"{object_id}: missing source name")
        require(
            isinstance(base, int)
            and not isinstance(base, bool)
            and isinstance(byte_count, int)
            and not isinstance(byte_count, bool)
            and base >= previous_end
            and byte_count > 0,
            f"{object_id}: invalid or overlapping logical extent",
        )
        seen_ids.add(object_id)
        previous_end = base + byte_count
        objects.append(
            {
                "template_object_index": index,
                "object_id": object_id,
                "kind": kind,
                "source_name": name,
                "logical_address": base,
                "bytes": byte_count,
                "expected_requests": int(source.get("requests", 0)),
                "expected_request_bytes": int(source.get("request_bytes", 0)),
            }
        )
    require(len(objects) <= 0xFFFF, "u16 object index is insufficient")
    return objects


class ObjectIndex:
    def __init__(self, objects: list[dict[str, Any]]) -> None:
        self.objects = objects
        self.begins = [int(item["logical_address"]) for item in objects]

    def match(self, address: int, byte_count: int) -> tuple[int, int]:
        cursor = bisect_right(self.begins, address) - 1
        require(cursor >= 0, f"request address {address} precedes every object")
        item = self.objects[cursor]
        base = int(item["logical_address"])
        size = int(item["bytes"])
        require(
            base <= address and address + byte_count <= base + size,
            f"request [{address},{address + byte_count}) escapes object {item['object_id']}",
        )
        offset = address - base
        require(offset <= 0xFFFFFFFF, "u32 object offset is insufficient")
        return cursor, offset


def normalized_hbf_mapping(route_manifest: dict[str, Any]) -> list[dict[str, int | str]]:
    mapping = route_manifest.get("hbf_device_logical_mapping") or {}
    raw = mapping.get("objects") or []
    require(isinstance(raw, list), "malformed HBF logical mapping")
    result = []
    previous_end = 0
    for item in sorted(raw, key=lambda row: int(row["target_hbf_logical_address"])):
        require(isinstance(item, dict), "malformed HBF mapping row")
        target = int(item["target_hbf_logical_address"])
        source = int(item["source_logical_address"])
        byte_count = int(item["bytes"])
        require(
            target >= previous_end and source >= 0 and byte_count > 0,
            "invalid or overlapping HBF mapping",
        )
        previous_end = target + byte_count
        result.append(
            {
                "object_id": str(item["object_id"]),
                "target_begin": target,
                "target_end": target + byte_count,
                "source_begin": source,
            }
        )
    return result


def reverse_hbf_address(mapping: list[dict[str, int | str]], address: int,
                        byte_count: int) -> int:
    for item in mapping:
        target_begin = int(item["target_begin"])
        target_end = int(item["target_end"])
        if target_begin <= address and address + byte_count <= target_end:
            return int(item["source_begin"]) + address - target_begin
    raise CompactTemplateError(
        f"HBF request [{address},{address + byte_count}) has no reverse mapping"
    )


def normalized_kernel_ranges(value: dict[str, Any]) -> list[dict[str, Any]]:
    raw = value.get("kernels")
    require(isinstance(raw, list) and raw, "kernel-range artifact has no kernels")
    result = []
    previous_end = 0
    for expected_ordinal, item in enumerate(raw):
        require(isinstance(item, dict), "malformed kernel-range row")
        ordinal = int(item["ordinal"])
        begin = int(item["request_ordinal_begin"])
        end = int(item["request_ordinal_end_exclusive"])
        require(
            ordinal == expected_ordinal and begin == previous_end and end > begin,
            f"kernel {ordinal} request range is not contiguous",
        )
        require(ordinal <= 0xFFFF, "u16 kernel ordinal is insufficient")
        previous_end = end
        result.append(
            {
                "ordinal": ordinal,
                "kernel_id": int(item["kernel_id"]),
                "request_ordinal_begin": begin,
                "request_ordinal_end_exclusive": end,
                "request_count": end - begin,
                "request_bytes": int(item["request_bytes"]),
            }
        )
    require(
        int(value.get("total_device_requests", previous_end)) == previous_end,
        "kernel ranges do not conserve the total request count",
    )
    return result


def expected_object_totals(objects: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    return {
        int(item["template_object_index"]): (
            int(item["expected_requests"]),
            int(item["expected_request_bytes"]),
        )
        for item in objects
    }


def _flush_buffer(stream: BinaryIO, buffer: bytearray, digest: Any) -> None:
    if not buffer:
        return
    stream.write(buffer)
    digest.update(buffer)
    buffer.clear()


def compile_template(
    *,
    transactions_path: Path,
    route_manifest_path: Path,
    object_summary_path: Path,
    kernel_ranges_path: Path,
    output_path: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    transactions_path = transactions_path.resolve()
    route_manifest_path = route_manifest_path.resolve()
    object_summary_path = object_summary_path.resolve()
    kernel_ranges_path = kernel_ranges_path.resolve()
    output_path = output_path.resolve()
    output_manifest_path = output_manifest_path.resolve()
    for path in (
        transactions_path,
        route_manifest_path,
        object_summary_path,
        kernel_ranges_path,
    ):
        require(path.is_file(), f"missing input: {path}")
    require(not output_path.exists(), f"refusing to overwrite {output_path}")
    require(
        not output_manifest_path.exists(),
        f"refusing to overwrite {output_manifest_path}",
    )

    route = load_json(route_manifest_path)
    summary = load_json(object_summary_path)
    ranges_value = load_json(kernel_ranges_path)
    objects = normalized_objects(summary)
    object_index = ObjectIndex(objects)
    hbf_mapping = normalized_hbf_mapping(route)
    kernels = normalized_kernel_ranges(ranges_value)
    expected_requests = int(route["transaction_count"])
    expected_bytes = int(route["transaction_bytes"])
    require(
        kernels[-1]["request_ordinal_end_exclusive"] == expected_requests,
        "kernel ranges disagree with routed transaction count",
    )
    require(
        int((summary.get("totals") or {}).get("requests", expected_requests))
        == expected_requests,
        "object summary disagrees with routed transaction count",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    by_object: dict[int, Counter[str]] = {
        int(item["template_object_index"]): Counter() for item in objects
    }
    by_kernel: dict[int, Counter[str]] = {
        int(item["ordinal"]): Counter() for item in kernels
    }
    kernel_cursor = 0
    buffer = bytearray()
    with transactions_path.open("rb") as source, output_path.open("xb") as target:
        for sequence_index, line in enumerate(source):
            match = LINE_RE.fullmatch(line)
            require(match is not None, f"malformed transaction line {sequence_index + 1}")
            event_index = int(match.group(1))
            require(
                event_index == sequence_index,
                f"event id {event_index} is not contiguous at {sequence_index}",
            )
            target_name = match.group(2).decode("ascii")
            operation = match.group(3).decode("ascii")
            address = int(match.group(4))
            byte_count = int(match.group(5))
            require(0 < byte_count <= 0xFFFF, "u16 request width is insufficient")
            if target_name == "HBM":
                source_address = address
            elif target_name == "HBF_LOGICAL":
                source_address = reverse_hbf_address(
                    hbf_mapping, address, byte_count
                )
            else:
                raise CompactTemplateError(f"unsupported route target {target_name!r}")

            while sequence_index >= kernels[kernel_cursor]["request_ordinal_end_exclusive"]:
                kernel_cursor += 1
                require(kernel_cursor < len(kernels), "request escaped kernel ranges")
            kernel = kernels[kernel_cursor]
            require(
                sequence_index >= kernel["request_ordinal_begin"],
                "request precedes current kernel range",
            )
            object_ordinal, object_offset = object_index.match(
                source_address, byte_count
            )
            buffer.extend(
                RECORD_STRUCT.pack(
                    object_ordinal,
                    object_offset,
                    int(kernel["ordinal"]),
                    byte_count,
                    OPERATION_CODE[operation],
                    0,
                )
            )
            if len(buffer) >= 8 * 1024 * 1024:
                _flush_buffer(target, buffer, digest)

            counts["requests"] += 1
            counts["bytes"] += byte_count
            counts[f"{operation.lower()}_requests"] += 1
            counts[f"{operation.lower()}_bytes"] += byte_count
            counts[f"target_{target_name.lower()}_requests"] += 1
            object_counter = by_object[object_ordinal]
            object_counter["requests"] += 1
            object_counter["bytes"] += byte_count
            object_counter[f"{operation.lower()}_requests"] += 1
            object_counter[f"{operation.lower()}_bytes"] += byte_count
            kernel_counter = by_kernel[int(kernel["ordinal"])]
            kernel_counter["requests"] += 1
            kernel_counter["bytes"] += byte_count
        _flush_buffer(target, buffer, digest)

    require(counts["requests"] == expected_requests, "request count did not conserve")
    require(counts["bytes"] == expected_bytes, "request bytes did not conserve")
    require(
        output_path.stat().st_size == expected_requests * RECORD_BYTES,
        "binary size does not match fixed-width record count",
    )
    for object_ordinal, expected in expected_object_totals(objects).items():
        actual = by_object[object_ordinal]
        require(
            (actual["requests"], actual["bytes"]) == expected,
            f"object {objects[object_ordinal]['object_id']} totals disagree: "
            f"{(actual['requests'], actual['bytes'])} != {expected}",
        )
    for kernel in kernels:
        actual = by_kernel[int(kernel["ordinal"])]
        require(
            actual["requests"] == int(kernel["request_count"])
            and actual["bytes"] == int(kernel["request_bytes"]),
            f"kernel {kernel['ordinal']} totals disagree",
        )

    elapsed = time.perf_counter() - started
    public_objects = []
    for item in objects:
        ordinal = int(item["template_object_index"])
        public_objects.append(
            {
                **item,
                "compiled": dict(sorted(by_object[ordinal].items())),
            }
        )
    public_kernels = []
    for item in kernels:
        ordinal = int(item["ordinal"])
        public_kernels.append(
            {**item, "compiled": dict(sorted(by_kernel[ordinal].items()))}
        )
    manifest = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": (
            "placement-neutral object-relative complete representative-layer "
            "32-byte request template"
        ),
        "binary_format": {
            "endianness": "little",
            "struct": RECORD_STRUCT.format,
            "record_bytes": RECORD_BYTES,
            "fields": [
                "template_object_index:u16",
                "object_offset:u32",
                "kernel_ordinal:u16",
                "bytes:u16",
                "operation:u8 (R=0,W=1)",
                "flags:u8 (zero in v1)",
            ],
        },
        "requests": expected_requests,
        "request_bytes": expected_bytes,
        "binary_bytes": output_path.stat().st_size,
        "binary_sha256": digest.hexdigest(),
        "object_table_sha256": sha256_value(public_objects),
        "objects": public_objects,
        "kernels": public_kernels,
        "totals": dict(sorted(counts.items())),
        "source": {
            "transactions": str(transactions_path),
            "transaction_trace_sha256": route.get("transaction_trace_sha256"),
            "device_trace_sha256": route.get("device_trace_sha256"),
            "route_manifest": str(route_manifest_path),
            "route_manifest_sha256": sha256_file(route_manifest_path),
            "object_summary": str(object_summary_path),
            "object_summary_sha256": sha256_file(object_summary_path),
            "kernel_ranges": str(kernel_ranges_path),
            "kernel_ranges_sha256": sha256_file(kernel_ranges_path),
            "routing_reversed": True,
        },
        "provenance": {
            "address": "derived source logical object base plus exact object offset",
            "operation_and_bytes": "preserved from routed transaction stream",
            "kernel_ordinal": "derived from digest-bound contiguous request ranges",
            "issue_time": "implicit request ordinal only; no production timestamp",
        },
        "not_claimed": [
            "issued GPU lane addresses before coalescing",
            "exact activation identity across layers",
            "production issue timestamps",
            "post-cache hardware equivalence beyond the source transform",
        ],
        "compilation": {
            "elapsed_seconds": elapsed,
            "records_per_second": expected_requests / elapsed if elapsed else None,
        },
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def iter_records(stream: BinaryIO) -> Iterable[tuple[int, int, int, int, int, int]]:
    remainder = b""
    while True:
        block = stream.read(RECORD_BYTES * 65536)
        if not block:
            break
        payload = remainder + block
        usable = len(payload) // RECORD_BYTES * RECORD_BYTES
        yield from RECORD_STRUCT.iter_unpack(payload[:usable])
        remainder = payload[usable:]
    require(not remainder, "truncated compact template record")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", required=True, type=Path)
    parser.add_argument("--route-manifest", required=True, type=Path)
    parser.add_argument("--object-summary", required=True, type=Path)
    parser.add_argument("--kernel-ranges", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    result = compile_template(
        transactions_path=args.transactions,
        route_manifest_path=args.route_manifest,
        object_summary_path=args.object_summary,
        kernel_ranges_path=args.kernel_ranges,
        output_path=args.output,
        output_manifest_path=args.output_manifest,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "requests": result["requests"],
                "binary_bytes": result["binary_bytes"],
                "binary_sha256": result["binary_sha256"],
                "elapsed_seconds": result["compilation"]["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

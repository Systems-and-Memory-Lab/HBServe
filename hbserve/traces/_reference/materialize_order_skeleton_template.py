#!/usr/bin/env python3
"""Reconstruct compact addresses in one captured order-skeleton stream."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from hbserve.traces._reference.compact_request_template import (
    KNOWN_REQUEST_FLAGS,
    RECORD_BYTES,
    load_json,
    require,
    sha256_file,
)
from hbserve.traces._reference.extract_nvbit_order_skeleton import RECORD_BYTES as SKELETON_RECORD_BYTES, SCHEMA as SKELETON_SCHEMA
from hbserve.traces._reference.parameterized_template_common import object_table, repeat_descriptor


SCHEMA = {"name": "hbfsim.order_skeleton_materialized_template", "version": 1}
COMPILED_ORDER_ONLY_SCHEMA = {
    "name": "hbfsim.compiled_order_only_skeleton",
    "version": 1,
}


def _bundle_ranges(manifest: dict[str, Any], template_requests: int) -> tuple[Any, Any]:
    import numpy as np

    program = manifest.get("instruction_program")
    require(isinstance(program, dict), "template has no instruction program")
    bundles = program.get("bundles")
    require(isinstance(bundles, list) and bundles, "instruction program has no bundles")
    begins = np.empty(len(bundles), dtype=np.uint32)
    lengths = np.empty(len(bundles), dtype=np.uint16)
    covered = np.zeros(template_requests, dtype=np.bool_)
    for index, row in enumerate(bundles):
        begin = int(row["request_ordinal_begin"])
        end = int(row["request_ordinal_end_exclusive"])
        require(0 <= begin < end <= template_requests, "bundle range escapes template")
        require(end - begin <= 0xFFFF, "bundle request count exceeds u16")
        require(not bool(covered[begin:end].any()), "instruction bundles overlap")
        covered[begin:end] = True
        begins[index] = begin
        lengths[index] = end - begin
    require(bool(covered.all()), "instruction bundles do not cover the template")
    return begins, lengths


def materialize_order_skeleton(
    *,
    template_manifest_path: Path,
    template_binary_path: Path,
    skeleton_manifest_path: Path,
    skeleton_binary_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    chunk_events: int = 1_000_000,
) -> dict[str, Any]:
    import numpy as np

    require(chunk_events > 0, "chunk event count must be positive")
    template_manifest_path = template_manifest_path.resolve()
    template_binary_path = template_binary_path.resolve()
    skeleton_manifest_path = skeleton_manifest_path.resolve()
    skeleton_binary_path = skeleton_binary_path.resolve()
    output_path = output_path.resolve()
    output_manifest_path = output_manifest_path.resolve()
    require(not output_path.exists(), f"refusing to overwrite {output_path}")
    require(not output_manifest_path.exists(), f"refusing to overwrite {output_manifest_path}")
    template = load_json(template_manifest_path)
    skeleton_manifest = load_json(skeleton_manifest_path)
    require(template.get("status") == "PASS", "template has not passed")
    skeleton_schema = skeleton_manifest.get("schema")
    require(
        skeleton_schema == SKELETON_SCHEMA
        or skeleton_schema == COMPILED_ORDER_ONLY_SCHEMA,
        "unsupported order skeleton",
    )
    require(skeleton_manifest.get("status") == "PASS", "order skeleton has not passed")
    source = skeleton_manifest.get("source") or {}
    require(source.get("template_manifest_sha256") == sha256_file(template_manifest_path),
            "order skeleton refers to another template")
    require(skeleton_manifest.get("binary_sha256") == sha256_file(skeleton_binary_path),
            "order skeleton binary digest differs")
    objects = object_table(template)
    repeat_count, strides = repeat_descriptor(template, objects)
    require(int((skeleton_manifest.get("geometry") or {}).get("repeat_count", -1))
            == repeat_count, "order skeleton repeat geometry differs")
    template_requests = int(template.get("requests", 0))
    require(template_binary_path.stat().st_size == template_requests * RECORD_BYTES,
            "template binary size disagrees")
    begins, lengths = _bundle_ranges(template, template_requests)

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
        [("bundle_index", "<u2"), ("group_instance", "<u2")], align=False
    )
    require(request_dtype.itemsize == RECORD_BYTES, "compact request dtype is not 12 bytes")
    require(skeleton_dtype.itemsize == SKELETON_RECORD_BYTES,
            "order skeleton dtype is not 4 bytes")
    template_records = np.memmap(template_binary_path, mode="r", dtype=request_dtype)
    skeleton = np.memmap(skeleton_binary_path, mode="r", dtype=skeleton_dtype)
    require(int(skeleton.size) == int(skeleton_manifest["events"]),
            "order skeleton event count differs")
    unknown = template_records["flags"] & (~KNOWN_REQUEST_FLAGS & 0xFF)
    require(not bool((unknown != 0).any()), "unsupported compact flags")
    stride_map = np.asarray(strides, dtype=np.uint64)
    extent_map = np.asarray([int(item["bytes"]) for item in objects], dtype=np.uint64)

    digest = hashlib.sha256()
    totals: Counter[str] = Counter()
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as target:
        for begin_event in range(0, int(skeleton.size), chunk_events):
            events = skeleton[begin_event : begin_event + chunk_events]
            bundle_indices = events["bundle_index"].astype(np.int64)
            require(not bool((bundle_indices >= begins.size).any()),
                    "skeleton bundle index escapes template program")
            counts = lengths[bundle_indices].astype(np.int64)
            output_count = int(counts.sum())
            require(output_count > 0, "skeleton chunk emits no compact requests")
            repeated_events = np.repeat(np.arange(events.size, dtype=np.int64), counts)
            output_starts = np.cumsum(counts, dtype=np.int64) - counts
            within_bundle = np.arange(output_count, dtype=np.int64) - np.repeat(
                output_starts, counts
            )
            request_indices = begins[bundle_indices[repeated_events]].astype(np.int64)
            request_indices += within_bundle
            output = np.array(template_records[request_indices], copy=True)
            object_indices = output["object_index"].astype(np.int64)
            groups = events["group_instance"][repeated_events].astype(np.uint64)
            offsets = output["object_offset"].astype(np.uint64) + (
                stride_map[object_indices] * groups
            )
            ends = offsets + output["bytes"].astype(np.uint64)
            require(not bool((ends > extent_map[object_indices]).any()),
                    "skeleton-expanded request escapes its object")
            require(not bool((offsets > 0xFFFFFFFF).any()),
                    "skeleton-expanded request offset exceeds u32")
            output["object_offset"] = offsets.astype(np.uint32)
            payload = output.tobytes(order="C")
            target.write(payload)
            digest.update(payload)
            totals["requests"] += int(output.size)
            totals["bytes"] += int(output["bytes"].sum(dtype=np.uint64))
            totals["r_requests"] += int((output["operation"] == 0).sum())
            totals["w_requests"] += int((output["operation"] == 1).sum())
            totals["r_bytes"] += int(
                output["bytes"][output["operation"] == 0].sum(dtype=np.uint64)
            )
            totals["w_bytes"] += int(
                output["bytes"][output["operation"] == 1].sum(dtype=np.uint64)
            )
    elapsed = time.perf_counter() - started
    expected_requests = template_requests * repeat_count
    require(totals["requests"] == expected_requests,
            "skeleton expansion request count disagrees")
    require(output_path.stat().st_size == expected_requests * RECORD_BYTES,
            "skeleton-expanded binary size disagrees")
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": (
            "small address program expanded in a captured instruction-order skeleton"
        ),
        "objects": copy.deepcopy(objects),
        "requests": int(totals["requests"]),
        "request_bytes": int(totals["bytes"]),
        "totals": dict(sorted(totals.items())),
        "binary_bytes": output_path.stat().st_size,
        "binary_sha256": digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "records_per_second": totals["requests"] / elapsed if elapsed else None,
        "parameterization": copy.deepcopy(template["parameterization"]),
        "source": {
            "template_manifest": str(template_manifest_path),
            "template_manifest_sha256": sha256_file(template_manifest_path),
            "template_binary": str(template_binary_path),
            "template_binary_sha256": sha256_file(template_binary_path),
            "skeleton_manifest": str(skeleton_manifest_path),
            "skeleton_manifest_sha256": sha256_file(skeleton_manifest_path),
            "skeleton_binary": str(skeleton_binary_path),
            "skeleton_binary_sha256": sha256_file(skeleton_binary_path),
        },
        "output": str(output_path),
        "not_claimed": [
            "production issue timestamps",
            "uninstrumented production cross-warp order",
            "reuse after changing kernel implementation or launch geometry",
        ],
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--template-binary", type=Path, required=True)
    parser.add_argument("--skeleton-manifest", type=Path, required=True)
    parser.add_argument("--skeleton-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--chunk-events", type=int, default=1_000_000)
    args = parser.parse_args()
    result = materialize_order_skeleton(
        template_manifest_path=args.template_manifest,
        template_binary_path=args.template_binary,
        skeleton_manifest_path=args.skeleton_manifest,
        skeleton_binary_path=args.skeleton_binary,
        output_path=args.output,
        output_manifest_path=args.output_manifest,
        chunk_events=args.chunk_events,
    )
    print(json.dumps({
        "status": result["status"],
        "requests": result["requests"],
        "request_bytes": result["request_bytes"],
        "elapsed_seconds": result["elapsed_seconds"],
        "records_per_second": result["records_per_second"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

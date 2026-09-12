#!/usr/bin/env python3
"""Test whether one large GEMV trace is a repeated CTA address template.

The analysis is deliberately exact at the issued global-lane boundary.  It
normalizes the weight and output addresses by the rows assigned to each CTA,
keeps shared input addresses object-relative, and preserves instruction order
inside each warp.  Cross-warp record interleaving is reported separately: it
is an instrumentation/scheduling observation, not part of a CTA's reusable
address program.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any

from hbserve.traces._reference.load_store_trace_contract import sha256_file
from hbserve.traces._reference.nvbit_raw_to_address_trace import (
    RangeIndex,
    _tensor_ownership_descriptors,
    load_known_ranges,
)
from hbserve.traces._reference.nvbit_v5_trace import iter_instruction_records, iter_instructions, read_trace_header


SCHEMA = {"name": "hbfsim.cta_translation_audit", "version": 1}
GRID_RE = re.compile(r"^\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_grid(value: Any) -> tuple[int, int, int]:
    require(isinstance(value, str), f"missing grid dimension: {value!r}")
    match = GRID_RE.fullmatch(value)
    require(match is not None, f"unsupported grid dimension: {value!r}")
    assert match is not None
    return tuple(int(match.group(index)) for index in range(1, 4))


def descriptor_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    objects = manifest.get("objects") or {}
    if name in objects:
        value = objects[name]
        require(isinstance(value, dict), f"object {name} is not a descriptor")
        return value
    weights = objects.get("weights")
    require(isinstance(weights, list), "probe manifest has no weights")
    matches = [item for item in weights if item.get("name") == name]
    if len(matches) == 1:
        return matches[0]

    # Intermediate GEMV outputs are captured by the same-run ownership hooks,
    # but are intentionally not promoted to semantic probe objects.  A
    # translation proof still needs their observed tensor shape.  Accept an
    # ownership span only when every full-span observation agrees on shape and
    # logical byte count; ambiguous/reused spans remain fail closed.
    ownership_matches = [
        descriptor
        for candidate_name, _is_weight, descriptor in
        _tensor_ownership_descriptors(manifest)
        if candidate_name == name
    ]
    require(
        len(ownership_matches) == 1,
        f"expected one descriptor named {name}, got {len(matches)}",
    )
    ownership = ownership_matches[0]
    full_span_views = []
    for allocation in ownership["allocation_candidates"]:
        for view in allocation["tensor_views"]:
            if (
                int(view["address_begin"]) == int(ownership["address_begin"])
                and int(view["address_end_exclusive"])
                == int(ownership["address_end_exclusive"])
            ):
                full_span_views.append(view)
    shapes = {tuple(int(value) for value in view["shape"]) for view in full_span_views}
    logical_bytes = {int(view["logical_bytes"]) for view in full_span_views}
    require(
        len(shapes) == 1 and len(logical_bytes) == 1,
        f"ownership descriptor {name} has no unambiguous full-span tensor shape",
    )
    return {
        **ownership,
        "shape": list(next(iter(shapes))),
        "logical_bytes": next(iter(logical_bytes)),
    }


def derive_translation(
    *,
    manifest: dict[str, Any],
    grid_x: int,
    weight_name: str,
    output_name: str,
) -> dict[str, int]:
    weight = descriptor_by_name(manifest, weight_name)
    output = descriptor_by_name(manifest, output_name)
    weight_shape = [int(value) for value in weight["shape"]]
    output_shape = [int(value) for value in output["shape"]]
    require(len(weight_shape) == 2, "weight must be a matrix")
    vocab, hidden = weight_shape
    require(output_shape and output_shape[-1] == vocab,
            "output vocabulary extent disagrees with the weight")
    require(vocab % grid_x == 0,
            "vocabulary rows are not evenly partitioned across CTA x")
    rows_per_cta = vocab // grid_x
    require(int(weight["logical_bytes"]) % vocab == 0,
            "weight row size is not integral")
    require(int(output["logical_bytes"]) % vocab == 0,
            "output element size is not integral")
    weight_row_bytes = int(weight["logical_bytes"]) // vocab
    output_element_bytes = int(output["logical_bytes"]) // vocab
    return {
        "vocab": vocab,
        "hidden": hidden,
        "rows_per_cta": rows_per_cta,
        "weight_row_bytes": weight_row_bytes,
        "output_element_bytes": output_element_bytes,
        "weight_cta_stride_bytes": rows_per_cta * weight_row_bytes,
        "output_cta_stride_bytes": rows_per_cta * output_element_bytes,
    }


def normalized_offset(
    *,
    source_name: str,
    object_begin: int,
    address: int,
    cta_x: int,
    weight_name: str,
    output_name: str,
    weight_cta_stride_bytes: int,
    output_cta_stride_bytes: int,
) -> int:
    offset = address - object_begin
    if source_name == weight_name:
        return offset - cta_x * weight_cta_stride_bytes
    if source_name == output_name:
        return offset - cta_x * output_cta_stride_bytes
    return offset


def canonical_cta_digest(warp_hashers: dict[int, Any]) -> str:
    """Combine per-warp programs without depending on cross-warp interleaving."""
    digest = hashlib.sha256()
    for warp, hasher in sorted(warp_hashers.items()):
        digest.update(struct.pack("<I", warp))
        digest.update(hasher.digest())
    return digest.hexdigest()


def audit_cta_translation(
    *,
    probe_manifest_path: Path,
    trace_path: Path,
    weight_name: str,
    output_name: str,
    coverage_audit_path: Path | None = None,
    expected_template_audit_path: Path | None = None,
    proof_only: bool = False,
) -> dict[str, Any]:
    probe_manifest_path = probe_manifest_path.resolve()
    trace_path = trace_path.resolve()
    manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
    header = read_trace_header(trace_path)
    grid = parse_grid(header["grid_dim"])
    require(grid[1:] == (1, 1), "only one-dimensional CTA translation is supported")
    filter_declared = bool(header.get("trace_cta_x_filter_declared", False))
    capture_begin = int(header.get("trace_cta_x_begin", 0))
    raw_capture_end = int(header.get("trace_cta_x_end_exclusive", -1))
    capture_end = grid[0] if raw_capture_end == -1 else raw_capture_end
    require(
        0 <= capture_begin < capture_end <= grid[0],
        f"declared CTA-x trace range [{capture_begin},{raw_capture_end}) "
        f"escapes grid x={grid[0]}",
    )
    captured_ctas = tuple(range(capture_begin, capture_end))
    full_grid_capture = capture_begin == 0 and capture_end == grid[0]
    translation = derive_translation(
        manifest=manifest,
        grid_x=grid[0],
        weight_name=weight_name,
        output_name=output_name,
    )
    expected_template_evidence = None
    expected_template_digest = None
    allowed_source_names = None
    if expected_template_audit_path is not None:
        expected_template_audit_path = expected_template_audit_path.resolve()
        expected_template = json.loads(
            expected_template_audit_path.read_text(encoding="utf-8")
        )
        require(expected_template.get("schema") == SCHEMA,
                "unsupported expected-template CTA audit")
        expected_kernel = expected_template.get("kernel") or {}
        require(expected_kernel.get("kernel_name") == header["kernel_name"],
                "expected-template audit refers to another kernel")
        require(expected_kernel.get("grid") == list(grid),
                "expected-template audit grid differs")
        require(expected_template.get("translation") == {
            "weight_name": weight_name,
            "output_name": output_name,
            **translation,
        }, "expected-template audit translation differs")
        expected_result = expected_template.get("result") or {}
        require(bool(expected_result.get("exact_single_translation_template")),
                "expected-template audit has multiple templates")
        expected_template_digest = str(
            expected_result.get("dominant_template_sha256", "")
        )
        require(bool(expected_template_digest),
                "expected-template audit has no canonical digest")
        expected_sources = expected_template.get("by_source")
        require(isinstance(expected_sources, dict) and expected_sources,
                "expected-template audit has no source census")
        require("anonymous" not in expected_sources,
                "expected-template audit contains anonymous addresses")
        allowed_source_names = set(expected_sources)
        expected_template_evidence = {
            "path": str(expected_template_audit_path),
            "sha256": sha256_file(expected_template_audit_path),
            "canonical_digest": expected_template_digest,
            "allowed_source_names": sorted(allowed_source_names),
        }

    ranges = load_known_ranges(manifest)
    if allowed_source_names is not None:
        ranges = [
            item for item in ranges
            if str(item["source_name"]) in allowed_source_names
        ]
        require(
            {str(item["source_name"]) for item in ranges}
            == allowed_source_names,
            "expected-template source filter is absent from the full manifest",
        )
    index = RangeIndex(ranges)
    strict_hashers = (
        None if proof_only
        else {cta: hashlib.sha256() for cta in captured_ctas}
    )
    warp_hashers: dict[int, dict[int, Any]] = {cta: {} for cta in captured_ctas}
    per_cta_records = {cta: 0 for cta in captured_ctas}
    per_cta_lanes = {cta: 0 for cta in captured_ctas}
    by_source: dict[str, Counter[str]] | None = None if proof_only else {}
    totals: Counter[str] = Counter()

    instructions = (
        iter_instructions(trace_path)
        if proof_only else
        (record.instruction for record in iter_instruction_records(trace_path))
    )
    for instruction in instructions:
        if instruction.memory_width <= 0:
            continue
        metadata = instruction.memory_reference_metadata
        require(metadata is not None, "CTA audit requires v7 memory-reference metadata")
        if metadata.mref_memory_space_name != "GLOBAL":
            continue
        cta_x, cta_y, cta_z = instruction.cta
        require(cta_y == 0 and cta_z == 0 and 0 <= cta_x < grid[0],
                f"CTA coordinate escapes grid: {instruction.cta}")
        require(
            cta_x in per_cta_records,
            f"trace contains CTA {cta_x} outside declared range "
            f"[{capture_begin},{capture_end})",
        )
        operation = (
            "RW" if metadata.is_load and metadata.is_store
            else "R" if metadata.is_load
            else "W" if metadata.is_store
            else "NONE"
        )
        require(operation != "NONE", "global memory reference has no operation")
        strict_hasher = None if strict_hashers is None else strict_hashers[cta_x]
        hasher = warp_hashers[cta_x].setdefault(
            instruction.warp_in_cta, hashlib.sha256()
        )
        record_prefix = struct.pack(
                "<IQIIII",
                instruction.active_mask,
                instruction.pc,
                instruction.memory_width,
                len(instruction.addresses),
                metadata.address_mref_index,
                len(instruction.opcode),
        )
        if strict_hasher is not None:
            strict_hasher.update(struct.pack("<I", instruction.warp_in_cta))
            strict_hasher.update(record_prefix)
            strict_hasher.update(instruction.opcode.encode("utf-8"))
            strict_hasher.update(operation.encode("ascii"))
        hasher.update(record_prefix)
        hasher.update(instruction.opcode.encode("utf-8"))
        hasher.update(operation.encode("ascii"))
        for lane, address in zip(instruction.active_lanes, instruction.addresses):
            known = index.match(address, instruction.memory_width)
            if known is None:
                source_name = "anonymous"
                begin = 0
                normalized = address
            else:
                source_name = str(known["source_name"])
                begin = int(known["begin"])
                normalized = normalized_offset(
                    source_name=source_name,
                    object_begin=begin,
                    address=address,
                    cta_x=cta_x,
                    weight_name=weight_name,
                    output_name=output_name,
                    weight_cta_stride_bytes=translation[
                        "weight_cta_stride_bytes"
                    ],
                    output_cta_stride_bytes=translation[
                        "output_cta_stride_bytes"
                    ],
                )
            require(normalized >= 0,
                    f"negative normalized offset for CTA {cta_x}: {normalized}")
            encoded_name = source_name.encode("utf-8")
            lane_payload = struct.pack("<IIQ", lane, len(encoded_name), normalized)
            if strict_hasher is not None:
                strict_hasher.update(lane_payload)
                strict_hasher.update(encoded_name)
            hasher.update(lane_payload)
            hasher.update(encoded_name)
            if by_source is not None:
                counters = by_source.setdefault(source_name, Counter())
                counters["lanes"] += 1
                counters["bytes"] += instruction.memory_width
                counters[f"{operation.lower()}_lanes"] += 1
                counters[f"{operation.lower()}_bytes"] += instruction.memory_width
        per_cta_records[cta_x] += 1
        per_cta_lanes[cta_x] += len(instruction.addresses)
        totals["global_records"] += 1
        totals["global_lanes"] += len(instruction.addresses)
        totals["global_lane_bytes"] += (
            len(instruction.addresses) * instruction.memory_width
        )

    require(
        all(count > 0 for count in per_cta_records.values()),
        "one or more declared CTAs were absent",
    )
    strict_digest_counts = None
    if strict_hashers is not None:
        strict_digests = [
            strict_hashers[cta].hexdigest() for cta in captured_ctas
        ]
        strict_digest_counts = Counter(strict_digests)
    digests = [canonical_cta_digest(warp_hashers[cta]) for cta in captured_ctas]
    digest_counts = Counter(digests)
    dominant_digest, dominant_count = digest_counts.most_common(1)[0]
    exact_single_template = len(digest_counts) == 1
    expected_template_match = (
        None if expected_template_digest is None
        else exact_single_template and dominant_digest == expected_template_digest
    )
    coverage_evidence = None
    coverage_match = None
    if coverage_audit_path is not None:
        coverage_audit_path = coverage_audit_path.resolve()
        coverage = json.loads(coverage_audit_path.read_text(encoding="utf-8"))
        require(coverage.get("schema") == SCHEMA, "unsupported coverage CTA audit")
        require(
            coverage.get("status") == "PASS_SINGLE_CTA_PER_WARP_TRANSLATION_TEMPLATE",
            "coverage CTA audit did not prove a full-grid single template",
        )
        coverage_kernel = coverage.get("kernel") or {}
        require(coverage_kernel.get("kernel_name") == header["kernel_name"],
                "coverage CTA audit refers to another kernel implementation")
        require(coverage_kernel.get("grid") == list(grid),
                "coverage CTA audit grid differs")
        require(coverage.get("translation") == {
            "weight_name": weight_name,
            "output_name": output_name,
            **translation,
        }, "coverage CTA audit translation differs")
        coverage_result = coverage.get("result") or {}
        require(
            int(coverage_result.get("ctas", -1)) == grid[0]
            and bool(coverage_result.get("exact_single_translation_template")),
            "coverage CTA audit is not a full-grid proof",
        )
        coverage_match = (
            exact_single_template
            and dominant_digest == coverage_result.get("dominant_template_sha256")
            and min(per_cta_records.values())
            == int(coverage_result.get("record_count_min", -1))
            and max(per_cta_records.values())
            == int(coverage_result.get("record_count_max", -1))
            and min(per_cta_lanes.values())
            == int(coverage_result.get("lane_count_min", -1))
            and max(per_cta_lanes.values())
            == int(coverage_result.get("lane_count_max", -1))
        )
        coverage_evidence = {
            "path": str(coverage_audit_path),
            "sha256": sha256_file(coverage_audit_path),
            "full_grid_ctas": grid[0],
            "sample_matches_full_grid_template": coverage_match,
        }

    if expected_template_match is False:
        status = "FAIL_FULL_GRID_DIFFERS_FROM_EXPECTED_TEMPLATE"
    elif not exact_single_template:
        status = "FAIL_MULTIPLE_NORMALIZED_CTA_TEMPLATES"
    elif full_grid_capture:
        status = "PASS_SINGLE_CTA_PER_WARP_TRANSLATION_TEMPLATE"
    elif coverage_match is True:
        status = "PASS_FILTERED_CTA_SAMPLE_MATCHES_FULL_AUDIT"
    elif coverage_match is False:
        status = "FAIL_FILTERED_CTA_SAMPLE_DIFFERS_FROM_FULL_AUDIT"
    else:
        status = "PASS_FILTERED_CTA_SAMPLE_SINGLE_TEMPLATE_ONLY"
    return {
        "schema": SCHEMA,
        "status": status,
        "classification": "exact issued-global-lane CTA translation audit",
        "inputs": {
            "probe_manifest": str(probe_manifest_path),
            "probe_manifest_sha256": sha256_file(probe_manifest_path),
            "trace": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
            "coverage_audit": coverage_evidence,
            "expected_template_audit": expected_template_evidence,
        },
        "kernel": {
            "kernel_id": header["kernel_id"],
            "kernel_name": header["kernel_name"],
            "grid": list(grid),
        },
        "translation": {
            "weight_name": weight_name,
            "output_name": output_name,
            **translation,
        },
        "result": {
            "ctas": len(captured_ctas),
            "grid_ctas": grid[0],
            "capture_cta_x_begin": capture_begin,
            "capture_cta_x_end_exclusive": capture_end,
            "capture_filter_declared": filter_declared,
            "full_grid_capture": full_grid_capture,
            "unique_normalized_cta_templates": len(digest_counts),
            "unique_strict_cross_warp_interleavings": (
                None if strict_digest_counts is None else len(strict_digest_counts)
            ),
            "strict_cross_warp_interleaving_exact": (
                None if strict_digest_counts is None
                else len(strict_digest_counts) == 1
            ),
            "matches_expected_template": expected_template_match,
            "proof_only": proof_only,
            "dominant_template_ctas": dominant_count,
            "dominant_template_fraction": dominant_count / len(captured_ctas),
            "dominant_template_sha256": dominant_digest,
            "exact_single_translation_template": exact_single_template,
            "record_count_min": min(per_cta_records.values()),
            "record_count_max": max(per_cta_records.values()),
            "lane_count_min": min(per_cta_lanes.values()),
            "lane_count_max": max(per_cta_lanes.values()),
            "warp_count_min": min(len(value) for value in warp_hashers.values()),
            "warp_count_max": max(len(value) for value in warp_hashers.values()),
            "ideal_raw_capture_reduction_factor": (
                grid[0] / len(captured_ctas)
                if exact_single_template
                else grid[0] / len(digest_counts)
            ),
        },
        "totals": dict(sorted(totals.items())),
        "by_source": (
            None if by_source is None else {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(by_source.items())
            }
        ),
        "template_histogram": [
            {"sha256": digest, "ctas": count}
            for digest, count in digest_counts.most_common()
        ],
        "not_claimed": [
            "CTA execution timestamps or inter-CTA scheduling order",
            "reuse after changing the kernel implementation or tile geometry",
            "post-cache or HBF-device traffic equivalence",
            *(
                ["full-grid per-source census and strict cross-warp interleaving distribution"]
                if proof_only else []
            ),
            *(
                ["full-grid homogeneity without the separately bound coverage audit"]
                if not full_grid_capture and coverage_match is not True
                else []
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--weight-name", default="lm_head.weight")
    parser.add_argument("--output-name", default="decode_output_logits")
    parser.add_argument("--coverage-audit", type=Path)
    parser.add_argument("--expected-template-audit", type=Path)
    parser.add_argument(
        "--proof-only",
        action="store_true",
        help=(
            "scan every lane for the canonical template proof while skipping "
            "unused raw-line, strict-interleaving, and source-census work"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_cta_translation(
        probe_manifest_path=args.probe_manifest,
        trace_path=args.trace,
        weight_name=args.weight_name,
        output_name=args.output_name,
        coverage_audit_path=args.coverage_audit,
        expected_template_audit_path=args.expected_template_audit,
        proof_only=args.proof_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "ctas": result["result"]["ctas"],
                "unique_templates": result["result"][
                    "unique_normalized_cta_templates"
                ],
                "dominant_fraction": result["result"][
                    "dominant_template_fraction"
                ],
                "capture_reduction_factor": result["result"][
                    "ideal_raw_capture_reduction_factor"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

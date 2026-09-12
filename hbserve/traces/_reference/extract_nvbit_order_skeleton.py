#!/usr/bin/env python3
"""Compile one audited full-grid NVBit order into a 4-byte event skeleton.

The full address trace is used only as a characterization oracle.  Once a
small per-warp address program has passed the full-grid translation audit,
each request-emitting global instruction needs only two values: the compact
program-bundle index and the CTA-group instance.  Addresses, lane masks,
opcodes, and widths remain in the small program template.

This fast scanner is deliberately bound by SHA-256 to a separately completed
strict audit.  It does not replace the strict NVBit parser.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import lzma
from pathlib import Path
import struct
import sys
import time
from typing import Any, BinaryIO

from hbserve.traces._reference.analyze_cta_translation import SCHEMA as AUDIT_SCHEMA, parse_grid
from hbserve.traces._reference.compact_request_template import load_json, require, sha256_file
from hbserve.traces._reference.nvbit_v5_trace import read_trace_header


SCHEMA = {"name": "hbfsim.nvbit_order_skeleton", "version": 1}
RECORD_STRUCT = struct.Struct("<HH")
RECORD_BYTES = RECORD_STRUCT.size
GLOBAL_MARKERS = (
    b"MEM_META_V1 3 GLOBAL 3 GLOBAL ",
    b"MEM_META_V1 6 GLOBAL_TO_SHARED 3 GLOBAL ",
)


def _open_binary_trace(path: Path) -> BinaryIO:
    if path.suffix == ".xz":
        return lzma.open(path, "rb")
    return path.open("rb")


def _bundle_lookup(
    template: dict[str, Any], ctas_per_group: int
) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int], int]]:
    program = template.get("instruction_program")
    require(isinstance(program, dict), "template has no instruction program")
    bundles = program.get("bundles")
    require(isinstance(bundles, list) and bundles, "instruction program has no bundles")
    lookup = {}
    counts: dict[tuple[int, int], int] = {}
    for bundle_index, row in enumerate(bundles):
        cta = int(row["cta_in_group"])
        warp = int(row["warp_in_cta"])
        ordinal = int(row["warp_program_ordinal"])
        require(0 <= cta < ctas_per_group and warp >= 0 and ordinal >= 0,
                "invalid instruction-bundle identity")
        key = (cta, warp, ordinal)
        require(key not in lookup, f"duplicate instruction bundle {key}")
        require(bundle_index <= 0xFFFF, "instruction-bundle index exceeds u16")
        lookup[key] = bundle_index
        counts[(cta, warp)] = max(counts.get((cta, warp), 0), ordinal + 1)
    for key, count in counts.items():
        require(
            all((key[0], key[1], ordinal) in lookup for ordinal in range(count)),
            f"instruction-bundle ordinals are not dense for {key}",
        )
    return lookup, counts


def extract_order_skeleton(
    *,
    template_manifest_path: Path,
    full_audit_path: Path,
    raw_trace_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    buffer_records: int = 1_000_000,
) -> dict[str, Any]:
    require(buffer_records > 0, "buffer record count must be positive")
    require(sys.byteorder == "little", "native u32 skeleton buffer requires little endian")
    template_manifest_path = template_manifest_path.resolve()
    full_audit_path = full_audit_path.resolve()
    raw_trace_path = raw_trace_path.resolve()
    output_path = output_path.resolve()
    output_manifest_path = output_manifest_path.resolve()
    require(not output_path.exists(), f"refusing to overwrite {output_path}")
    require(not output_manifest_path.exists(), f"refusing to overwrite {output_manifest_path}")
    template = load_json(template_manifest_path)
    audit = load_json(full_audit_path)
    require(template.get("status") == "PASS", "program template has not passed")
    require(audit.get("schema") == AUDIT_SCHEMA, "unsupported CTA audit")
    require(
        audit.get("status") == "PASS_SINGLE_CTA_PER_WARP_TRANSLATION_TEMPLATE",
        "order skeleton requires a passing full-grid CTA audit",
    )
    audit_input = audit.get("inputs") or {}
    require(audit_input.get("trace_sha256") == sha256_file(raw_trace_path),
            "full audit refers to another raw trace")
    header = read_trace_header(raw_trace_path)
    grid = parse_grid(header.get("grid_dim"))
    require(grid[1:] == (1, 1), "only one-dimensional CTA grids are supported")
    require((audit.get("kernel") or {}).get("grid") == list(grid),
            "full audit grid differs from raw trace")
    require((audit.get("kernel") or {}).get("kernel_name") == header["kernel_name"],
            "full audit kernel differs from raw trace")
    parameterization = template.get("parameterization") or {}
    ctas_per_group = int(parameterization.get("ctas_per_group", 0))
    repeat_count = int(parameterization.get("repeat_count", 0))
    require(ctas_per_group > 0 and repeat_count * ctas_per_group == grid[0],
            "template parameterization disagrees with full grid")
    require(repeat_count <= 0xFFFF, "CTA-group instance exceeds u16")
    lookup, expected_counts = _bundle_lookup(template, ctas_per_group)
    warp_ids = sorted({warp for _cta, warp in expected_counts})
    require(warp_ids == list(range(len(warp_ids))), "warp IDs are not dense")
    warps_per_cta = len(warp_ids)
    require(warps_per_cta > 0, "template has no warp programs")
    for cta in range(ctas_per_group):
        require(
            {(candidate_cta, warp) for candidate_cta, warp in expected_counts
             if candidate_cta == cta}
            == {(cta, warp) for warp in warp_ids},
            f"template CTA {cta} does not contain every warp program",
        )

    counters = array("H", [0]) * (grid[0] * warps_per_cta)
    digest = hashlib.sha256()
    buffer = array("I")
    events = 0
    raw_instruction_records = 0
    zero_active_global_records = 0
    started = time.perf_counter()
    execution_fields = 13 if bool(header.get("trace_core")) else 11
    active_mask_field = execution_fields + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_binary_trace(raw_trace_path) as source, output_path.open("xb") as target:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith((b"-", b"#")):
                continue
            raw_instruction_records += 1
            if not any(marker in stripped for marker in GLOBAL_MARKERS):
                continue
            fields = stripped.split(None, active_mask_field + 1)
            require(
                len(fields) > active_mask_field,
                f"raw trace line {line_number} has a truncated execution prefix",
            )
            try:
                cta_x = int(fields[0])
                cta_y = int(fields[1])
                cta_z = int(fields[2])
                warp = int(fields[3])
                active_mask = int(fields[active_mask_field], 16)
            except ValueError as error:
                raise ValueError(
                    f"raw trace line {line_number} has a malformed execution prefix"
                ) from error
            require(cta_y == 0 and cta_z == 0 and 0 <= cta_x < grid[0],
                    f"raw trace line {line_number} has an invalid CTA")
            require(0 <= warp < warps_per_cta,
                    f"raw trace line {line_number} has an invalid warp")
            if active_mask == 0:
                zero_active_global_records += 1
                continue
            counter_index = cta_x * warps_per_cta + warp
            ordinal = counters[counter_index]
            local_cta = cta_x % ctas_per_group
            bundle_index = lookup.get((local_cta, warp, ordinal))
            require(
                bundle_index is not None,
                f"raw order exceeds template program at CTA {cta_x}, warp {warp}, "
                f"ordinal {ordinal}",
            )
            group_instance = cta_x // ctas_per_group
            counters[counter_index] = ordinal + 1
            # On a little-endian host, one native u32 stores exactly the public
            # <bundle:u16, group:u16> record without a per-event struct call.
            buffer.append(int(bundle_index) | (group_instance << 16))
            events += 1
            if len(buffer) >= buffer_records:
                payload = buffer.tobytes()
                target.write(payload)
                digest.update(payload)
                buffer = array("I")
        if buffer:
            payload = buffer.tobytes()
            target.write(payload)
            digest.update(payload)

    for cta_x in range(grid[0]):
        local_cta = cta_x % ctas_per_group
        for warp in warp_ids:
            observed = counters[cta_x * warps_per_cta + warp]
            expected = expected_counts[(local_cta, warp)]
            require(observed == expected,
                    f"CTA {cta_x} warp {warp} emitted {observed}, expected {expected}")
    expected_events = sum(counters)
    require(events == expected_events, "skeleton event count disagrees with warp census")
    audit_global_records = int((audit.get("totals") or {}).get("global_records", -1))
    require(
        events + zero_active_global_records == audit_global_records,
        "request-emitting and zero-active records do not conserve audited globals",
    )
    require(output_path.stat().st_size == events * RECORD_BYTES,
            "skeleton binary size disagrees")
    elapsed = time.perf_counter() - started
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": (
            "full-grid tracer-observed request-emitting instruction order skeleton"
        ),
        "binary_format": {
            "endianness": "little",
            "struct": RECORD_STRUCT.format,
            "record_bytes": RECORD_BYTES,
            "fields": ["template_bundle_index", "cta_group_instance"],
        },
        "events": events,
        "binary_bytes": output_path.stat().st_size,
        "binary_sha256": digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "events_per_second": events / elapsed if elapsed else None,
        "geometry": {
            "grid_x": grid[0],
            "ctas_per_group": ctas_per_group,
            "repeat_count": repeat_count,
            "warps_per_cta": warps_per_cta,
        },
        "census": {
            "raw_instruction_records": raw_instruction_records,
            "audited_global_records": audit_global_records,
            "request_emitting_global_records": events,
            "zero_active_global_records": zero_active_global_records,
        },
        "reduction": {
            "full_raw_compressed_bytes": raw_trace_path.stat().st_size,
            "skeleton_to_full_raw_compressed_fraction": (
                output_path.stat().st_size / raw_trace_path.stat().st_size
            ),
        },
        "source": {
            "template_manifest": str(template_manifest_path),
            "template_manifest_sha256": sha256_file(template_manifest_path),
            "full_audit": str(full_audit_path),
            "full_audit_sha256": sha256_file(full_audit_path),
            "raw_trace": str(raw_trace_path),
            "raw_trace_sha256": sha256_file(raw_trace_path),
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
    parser.add_argument("--full-audit", type=Path, required=True)
    parser.add_argument("--raw-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--buffer-records", type=int, default=1_000_000)
    args = parser.parse_args()
    result = extract_order_skeleton(
        template_manifest_path=args.template_manifest,
        full_audit_path=args.full_audit,
        raw_trace_path=args.raw_trace,
        output_path=args.output,
        output_manifest_path=args.output_manifest,
        buffer_records=args.buffer_records,
    )
    print(json.dumps({
        "status": result["status"],
        "events": result["events"],
        "binary_bytes": result["binary_bytes"],
        "elapsed_seconds": result["elapsed_seconds"],
        "events_per_second": result["events_per_second"],
        "reduction": result["reduction"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

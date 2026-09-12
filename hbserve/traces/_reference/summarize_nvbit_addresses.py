#!/usr/bin/env python3
"""Map Accel-Sim/NVBit lane addresses back to live Qwen tensor ranges."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from hbserve.traces._reference.nvbit_v5_trace import open_trace, parse_instruction_record, read_trace_header


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def access_mode(opcode: str) -> str:
    upper = opcode.upper()
    if upper.startswith(("ATOM", "RED")):
        return "read_write"
    if upper.startswith(("STG", "ST.")):
        return "write"
    if upper.startswith(("LDG", "LD.")):
        return "read"
    return "other"


def address_space(opcode: str) -> str:
    upper = opcode.upper()
    if upper.startswith(("LDS", "STS", "LDSM")):
        return "shared"
    if upper.startswith(("LDL", "STL")):
        return "local"
    if upper.startswith(("LDG", "STG", "LDGSTS", "ATOM", "RED")):
        return "global"
    return "other"


def unmatched_kernel_category(kernel_name: str) -> str:
    """Classify unmatched traffic by kernel role, not by allocation identity.

    This is intentionally a name-based diagnostic.  It must not be described
    as an allocation census because the current probe did not intercept every
    CUDA allocation.
    """
    lower = kernel_name.lower()
    if "catarraybatchedcopy" in lower:
        return "dynamic_kv_concat_or_copy_temporary"
    if "flash_fwd" in lower or "attention" in lower:
        return "attention_activation_or_scratch"
    if "gemv" in lower or "gemm" in lower:
        return "projection_activation_or_output"
    if "elementwise" in lower or "vectorized" in lower:
        return "elementwise_or_rope_temporary"
    return "unclassified_kernel_role"


def parse_instruction(line: str) -> tuple[str, int, list[int]] | None:
    instruction = parse_instruction_record(line)
    if instruction is None or instruction.memory_width == 0:
        return None
    return (
        instruction.opcode,
        instruction.memory_width,
        list(instruction.addresses),
    )


def object_ranges(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    objects = manifest["objects"]
    result: list[dict[str, Any]] = []
    for weight in objects["weights"]:
        result.append(
            {
                "name": weight["name"],
                "kind": "weight",
                "begin": int(weight["address_begin"]),
                "end": int(weight["address_end_exclusive"]),
            }
        )
    for name, descriptor in objects.items():
        if name == "weights":
            continue
        result.append(
            {
                "name": name,
                "kind": "kv" if name.startswith("kv_") else "activation",
                "begin": int(descriptor["address_begin"]),
                "end": int(descriptor["address_end_exclusive"]),
            }
        )
    return sorted(result, key=lambda item: (item["begin"], item["end"]))


def matching_object(address: int, ranges: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [item for item in ranges if item["begin"] <= address < item["end"]]
    if not matches:
        return None
    return min(matches, key=lambda item: item["end"] - item["begin"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ranges = object_ranges(manifest)
    trace_files = sorted(args.trace_root.glob("kernel-*.trace*"))
    if not trace_files:
        raise ValueError(f"no trace files found under {args.trace_root}")

    opcodes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    spaces: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    per_object: dict[str, Counter[str]] = defaultdict(Counter)
    per_object_lines: dict[str, set[int]] = defaultdict(set)
    per_object_pages: dict[str, set[int]] = defaultdict(set)
    unmatched_by_kernel_category: dict[str, Counter[str]] = defaultdict(Counter)
    kernel_summaries = []

    for trace_path in trace_files:
        kernel: Counter[str] = Counter()
        kernel_unmatched_pages: set[int] = set()
        trace_header = read_trace_header(trace_path)
        kernel_name = str(trace_header["kernel_name"])
        shared_bytes = trace_header.get("shared_memory_bytes")
        shared_base = trace_header.get("shared_memory_base_address")
        with open_trace(trace_path) as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.startswith("-kernel name = "):
                    kernel_name = line.removeprefix("-kernel name = ").strip()
                    continue
                try:
                    instruction = parse_instruction_record(line)
                except (IndexError, ValueError) as error:
                    raise ValueError(
                        f"cannot parse {trace_path}:{line_number}: {line.strip()}"
                    ) from error
                if instruction is None or instruction.memory_width == 0:
                    continue
                opcode = instruction.opcode
                memory_width = instruction.memory_width
                addresses = list(instruction.addresses)
                if opcode.upper().startswith("LDGSTS"):
                    if not addresses:
                        mode = "other"
                        space = "ldgsts_inactive"
                    else:
                        if not isinstance(shared_bytes, int) or shared_bytes <= 0:
                            raise ValueError(
                                f"{trace_path}: LDGSTS requires a shmem header"
                            )

                        def is_shared_address(address: int) -> bool:
                            if 0 <= address < shared_bytes:
                                return True
                            return (
                                isinstance(shared_base, int)
                                and shared_base
                                <= address
                                < shared_base + shared_bytes
                            )

                        shared_flags = [is_shared_address(value) for value in addresses]
                        if all(shared_flags):
                            mode = "other"
                            space = "shared_destination_mirror"
                        elif any(shared_flags):
                            raise ValueError(
                                f"{trace_path}:{line_number}: mixed LDGSTS domains"
                            )
                        else:
                            mode = "read"
                            space = "global"
                else:
                    mode = access_mode(opcode)
                    space = address_space(opcode)
                opcodes[opcode] += 1
                modes[mode] += 1
                spaces[space] += 1
                totals["memory_instructions"] += 1
                totals["lane_accesses"] += len(addresses)
                totals["lane_bytes"] += len(addresses) * memory_width
                totals[f"{space}_memory_instructions"] += 1
                totals[f"{space}_lane_accesses"] += len(addresses)
                totals[f"{space}_lane_bytes"] += len(addresses) * memory_width
                kernel["memory_instructions"] += 1
                kernel["lane_accesses"] += len(addresses)
                kernel["lane_bytes"] += len(addresses) * memory_width
                if space != "global":
                    continue
                kernel["global_memory_instructions"] += 1
                kernel["global_lane_accesses"] += len(addresses)
                kernel["global_lane_bytes"] += len(addresses) * memory_width
                for address in addresses:
                    matched = matching_object(address, ranges)
                    if matched is None:
                        totals["unmatched_lane_accesses"] += 1
                        totals["unmatched_lane_bytes"] += memory_width
                        kernel["unmatched_lane_accesses"] += 1
                        kernel["unmatched_lane_bytes"] += memory_width
                        kernel_unmatched_pages.add(address // 4096)
                        continue
                    name = matched["name"]
                    totals["matched_lane_accesses"] += 1
                    totals["matched_lane_bytes"] += memory_width
                    kernel["matched_lane_accesses"] += 1
                    kernel["matched_lane_bytes"] += memory_width
                    per_object[name][f"{mode}_lane_accesses"] += 1
                    per_object[name][f"{mode}_lane_bytes"] += memory_width
                    per_object_lines[name].add(address // 128)
                    per_object_pages[name].add(address // 4096)
        category = unmatched_kernel_category(kernel_name)
        category_counter = unmatched_by_kernel_category[category]
        category_counter["kernels"] += 1
        category_counter["unmatched_lane_accesses"] += kernel.get(
            "unmatched_lane_accesses", 0
        )
        category_counter["unmatched_lane_bytes"] += kernel.get(
            "unmatched_lane_bytes", 0
        )
        kernel_summaries.append(
            {
                "file": trace_path.name,
                "sha256": sha256_file(trace_path),
                "kernel_name": kernel_name,
                "unmatched_kernel_role": category,
                "unmatched_unique_4k_pages": len(kernel_unmatched_pages),
                **dict(kernel),
            }
        )

    object_by_name = {item["name"]: item for item in ranges}
    object_summaries = []
    for name, counters in sorted(
        per_object.items(), key=lambda item: -sum(item[1].values())
    ):
        metadata = object_by_name[name]
        allocation_bytes = metadata["end"] - metadata["begin"]
        read_bytes = counters.get("read_lane_bytes", 0)
        write_bytes = counters.get("write_lane_bytes", 0)
        object_summaries.append(
            {
                "name": name,
                "kind": metadata["kind"],
                "allocation_bytes": allocation_bytes,
                **dict(counters),
                "read_to_allocation_ratio": (
                    read_bytes / allocation_bytes if allocation_bytes else 0.0
                ),
                "write_to_allocation_ratio": (
                    write_bytes / allocation_bytes if allocation_bytes else 0.0
                ),
                "unique_128b_lines": len(per_object_lines[name]),
                "unique_4k_pages": len(per_object_pages[name]),
            }
        )

    global_lane_bytes = totals["global_lane_bytes"]
    summary = {
        "schema": {"name": "hbfsim.nvbit_address_summary", "version": 1},
        "source_manifest": str(args.manifest.resolve()),
        "trace_root": str(args.trace_root.resolve()),
        "trace_files": len(trace_files),
        "totals": {
            **dict(totals),
            "known_object_byte_fraction": (
                totals["matched_lane_bytes"] / global_lane_bytes
                if global_lane_bytes
                else 0.0
            ),
        },
        "access_modes_by_instruction": dict(modes.most_common()),
        "address_spaces_by_instruction": dict(spaces.most_common()),
        "opcodes_by_instruction": dict(opcodes.most_common()),
        "objects": object_summaries,
        "kernels": kernel_summaries,
        "unmatched_by_kernel_role": {
            "classification": (
                "derived kernel-name heuristic; not a CUDA allocation census"
            ),
            "roles": {
                name: dict(sorted(counters.items()))
                for name, counters in sorted(unmatched_by_kernel_category.items())
            },
        },
        "interpretation": {
            "lane_bytes": "executed SASS lane width; not post-cache DRAM traffic",
            "known_object_fraction": "fraction of global lane bytes mapped to recorded weight/KV/input/output ranges; temporary tensors are intentionally unmatched",
            "unique_lines_pages": "virtual address footprint, not physical HBF transactions",
            "unmatched_kernel_role": "diagnostic grouping by kernel name; exact temporary-allocation identity is still unknown",
            "ldgsts": "v5 emits untagged global-source and shared-destination operand records; shared offsets are excluded from the global census",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["totals"], sort_keys=True))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

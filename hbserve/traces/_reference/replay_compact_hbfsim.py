#!/usr/bin/env python3
"""Stream a compact request trace into the HBFSim session protocol.

The protocol requires the transaction digest in BEGIN.  This frontend scans
the compact binary once to count per-kernel requests, once to hash the exact
transaction text, and once more while piping that text to HBFSim.  It never
stores the hundreds-of-megabytes transaction file.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator

from hbserve.traces._reference.compact_request_template import (
    OPERATION_NAME,
    RECORD_BYTES,
    RECORD_STRUCT,
    known_request_flags,
    load_json,
    require,
    sha256_file,
)
from hbserve.traces._reference.full_model_trace_plan import PLAN_SCHEMA


SCHEMA = {"name": "hbfsim.compact_stream_replay", "version": 1}
PAGE_BYTES = 4096
POLICIES = ("all-hbm", "all-hbf", "weight-kv-hbf", "weight-hbf", "kv-hbf")


@dataclass(frozen=True)
class SyntheticWriteEnvelope:
    """A bounded post-L2 traffic uncertainty, not a reconstructed cache trace.

    The historical class name is retained for artifact compatibility; the
    operation field permits the same digest-bound mechanism to carry reads.
    """

    target: str
    start_bytes: int
    uniform_bytes: int
    end_bytes: int
    request_bytes: int
    base_address: int
    operation: str = "W"

    @property
    def total_bytes(self) -> int:
        return self.start_bytes + self.uniform_bytes + self.end_bytes

    @property
    def start_requests(self) -> int:
        return self.start_bytes // self.request_bytes

    @property
    def uniform_requests(self) -> int:
        return self.uniform_bytes // self.request_bytes

    @property
    def end_requests(self) -> int:
        return self.end_bytes // self.request_bytes

    @property
    def total_requests(self) -> int:
        return self.total_bytes // self.request_bytes


def build_synthetic_envelope(
    *, target: str | None, start_bytes: int, uniform_bytes: int,
    request_bytes: int, base_address: int, operation: str = "W",
    end_bytes: int = 0,
) -> SyntheticWriteEnvelope | None:
    require(start_bytes >= 0 and uniform_bytes >= 0 and end_bytes >= 0,
            "synthetic write bytes must be nonnegative")
    require(request_bytes in {32, 128},
            "synthetic write request size must be 32 or 128 B")
    require(base_address >= 0, "synthetic write base address must be nonnegative")
    require(operation in {"R", "W"}, "synthetic operation must be R or W")
    total = start_bytes + uniform_bytes + end_bytes
    if total == 0:
        require(target is None,
                "synthetic write target requires nonzero synthetic bytes")
        return None
    require(target in {"HBM", "HBF_LOGICAL"},
            "nonzero synthetic writes require an HBM or HBF_LOGICAL target")
    require(start_bytes % request_bytes == 0,
            "synthetic start write bytes must contain whole requests")
    require(uniform_bytes % request_bytes == 0,
            "synthetic uniform write bytes must contain whole requests")
    require(end_bytes % request_bytes == 0,
            "synthetic end write bytes must contain whole requests")
    require(base_address % request_bytes == 0,
            "synthetic write base address must be request aligned")
    return SyntheticWriteEnvelope(
        target=str(target), start_bytes=start_bytes,
        uniform_bytes=uniform_bytes, end_bytes=end_bytes,
        request_bytes=request_bytes,
        base_address=base_address, operation=operation,
    )


def iter_records(path: Path) -> Iterator[tuple[int, int, int, int, int, int]]:
    remainder = b""
    with path.open("rb") as stream:
        while payload := stream.read(RECORD_BYTES * 65536):
            payload = remainder + payload
            usable = len(payload) // RECORD_BYTES * RECORD_BYTES
            yield from RECORD_STRUCT.iter_unpack(payload[:usable])
            remainder = payload[usable:]
    require(not remainder, "truncated compact input")


def target_for_kind(kind: str, policy: str) -> str:
    if policy == "all-hbm":
        return "HBM"
    if policy == "all-hbf":
        return "HBF_LOGICAL"
    if policy == "weight-kv-hbf":
        return "HBF_LOGICAL" if kind in {"weight", "kv_cache"} else "HBM"
    if policy == "weight-hbf":
        return "HBF_LOGICAL" if kind == "weight" else "HBM"
    if policy == "kv-hbf":
        return "HBF_LOGICAL" if kind == "kv_cache" else "HBM"
    raise ValueError(f"unsupported policy {policy}")


def load_cadence(path: Path, offset: int, kernel_count: int) -> list[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(offset >= 0 and offset + kernel_count <= len(rows),
            "cadence kernel interval escapes CSV")
    selected = rows[offset : offset + kernel_count]
    origin = int(selected[0]["Start (ns)"])
    result = []
    for row in selected:
        start = int(row["Start (ns)"]) - origin
        duration = int(row["Duration (ns)"])
        require(start >= 0 and duration >= 0, "invalid cadence row")
        result.append((start, duration))
    return result


def issue_ns(
    cadence: list[tuple[int, int]], kernel: int, rank: int, count: int
) -> float:
    require(0 <= kernel < len(cadence), "request kernel escapes cadence")
    start, duration = cadence[kernel]
    if count <= 1:
        return float(start)
    return start + duration * rank / (count - 1)


def transaction_line(
    *, sequence: int, target: str, operation: str, address: int,
    byte_count: int, issue: float, id_prefix: str = "e",
    dependencies: tuple[str, ...] = (),
) -> bytes:
    dependency_field = "-" if not dependencies else ",".join(dependencies)
    return (
        f"TX id={id_prefix}{sequence} target={target} op={operation} addr={address} "
        f"bytes={byte_count} issue_ns={format(issue, '.17g')} "
        f"duration_ns=0 deps={dependency_field} stack=-\n"
    ).encode("ascii")


def scan_counts(binary: Path, objects: list[dict[str, Any]]) -> tuple[Counter[int], Counter[str]]:
    by_kernel: Counter[int] = Counter()
    totals: Counter[str] = Counter()
    for object_index, offset, kernel, byte_count, operation, flags in iter_records(binary):
        require(known_request_flags(flags), "unsupported compact flags")
        require(object_index < len(objects), "request object index escapes plan")
        require(offset + byte_count <= int(objects[object_index]["bytes"]),
                "request escapes plan object")
        require(operation in OPERATION_NAME, "unsupported compact operation")
        by_kernel[kernel] += 1
        totals["transactions"] += 1
        totals["transaction_bytes"] += byte_count
    require(totals["transactions"] > 0, "compact trace is empty")
    return by_kernel, totals


def iter_transaction_lines(
    *, binary: Path, objects: list[dict[str, Any]], policy: str,
    cadence: list[tuple[int, int]], kernel_counts: Counter[int],
    synthetic: SyntheticWriteEnvelope | None = None,
) -> Iterator[tuple[bytes, str, int]]:
    ranks: Counter[int] = Counter()
    synthetic_sequence = 0

    if synthetic is not None:
        for index in range(synthetic.start_requests):
            yield (
                transaction_line(
                    sequence=synthetic_sequence, id_prefix="s",
                    target=synthetic.target, operation=synthetic.operation,
                    address=synthetic.base_address + index * synthetic.request_bytes,
                    byte_count=synthetic.request_bytes, issue=0.0,
                ),
                synthetic.target,
                synthetic.request_bytes,
            )
            synthetic_sequence += 1

    observed_kernels = sorted(kernel_counts)
    uniform_by_kernel: dict[int, int] = {}
    if synthetic is not None and synthetic.uniform_requests:
        quotient, remainder = divmod(
            synthetic.uniform_requests, len(observed_kernels)
        )
        uniform_by_kernel = {
            kernel: quotient + (1 if index < remainder else 0)
            for index, kernel in enumerate(observed_kernels)
        }

    uniform_emitted = 0

    def uniform_after_kernel(kernel: int) -> Iterator[tuple[bytes, str, int]]:
        nonlocal synthetic_sequence, uniform_emitted
        if synthetic is None:
            return
        start, duration = cadence[kernel]
        for unused in range(uniform_by_kernel.get(kernel, 0)):
            address = synthetic.base_address + (
                synthetic.start_requests + uniform_emitted
            ) * synthetic.request_bytes
            yield (
                transaction_line(
                    sequence=synthetic_sequence, id_prefix="s",
                    target=synthetic.target, operation=synthetic.operation, address=address,
                    byte_count=synthetic.request_bytes,
                    issue=float(start + duration),
                ),
                synthetic.target,
                synthetic.request_bytes,
            )
            synthetic_sequence += 1
            uniform_emitted += 1

    previous_kernel: int | None = None
    for sequence, record in enumerate(iter_records(binary)):
        object_index, offset, kernel, byte_count, operation, _flags = record
        if previous_kernel is not None and kernel != previous_kernel:
            require(kernel > previous_kernel,
                    "compact requests must have nondecreasing kernel ordinals")
            yield from uniform_after_kernel(previous_kernel)
        obj = objects[object_index]
        target = target_for_kind(str(obj["kind"]), policy)
        current_rank = ranks[kernel]
        ranks[kernel] += 1
        issue = issue_ns(cadence, kernel, current_rank, kernel_counts[kernel])
        yield (
            transaction_line(
                sequence=sequence,
                target=target,
                operation=OPERATION_NAME[operation],
                address=int(obj["logical_address"]) + offset,
                byte_count=byte_count,
                issue=issue,
            ),
            target,
            byte_count,
        )
        previous_kernel = kernel
    if previous_kernel is not None:
        yield from uniform_after_kernel(previous_kernel)
    if synthetic is not None:
        require(previous_kernel is not None,
                "synthetic end traffic requires an observed kernel")
        end_issue = float(sum(cadence[previous_kernel]))
        for unused in range(synthetic.end_requests):
            address = synthetic.base_address + synthetic_sequence * synthetic.request_bytes
            yield (
                transaction_line(
                    sequence=synthetic_sequence, id_prefix="s",
                    target=synthetic.target, operation=synthetic.operation,
                    address=address, byte_count=synthetic.request_bytes,
                    issue=end_issue,
                ),
                synthetic.target,
                synthetic.request_bytes,
            )
            synthetic_sequence += 1
        require(uniform_emitted == synthetic.uniform_requests,
                "synthetic uniform traffic count disagrees")
        require(synthetic_sequence == synthetic.total_requests,
                "synthetic total traffic count disagrees")


def run_replay(
    *, plan_path: Path, binary_path: Path, policy: str,
    cadence_csv_path: Path, cadence_kernel_offset: int,
    simulator_path: Path, system_config_path: Path, output_dir: Path,
    synthetic_write_target: str | None = None,
    synthetic_start_write_bytes: int = 0,
    synthetic_uniform_write_bytes: int = 0,
    synthetic_end_write_bytes: int = 0,
    synthetic_write_request_bytes: int = 32,
    synthetic_write_base_address: int | None = None,
    synthetic_operation: str = "W",
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    binary_path = binary_path.resolve()
    cadence_csv_path = cadence_csv_path.resolve()
    simulator_path = simulator_path.resolve()
    system_config_path = system_config_path.resolve()
    output_dir = output_dir.resolve()
    require(policy in POLICIES, "unsupported routing policy")
    plan = load_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported plan schema")
    objects = plan.get("objects")
    require(isinstance(objects, list) and objects, "plan has no objects")
    require([int(obj["target_object_index"]) for obj in objects]
            == list(range(len(objects))), "plan object indices are not dense")
    object_end = max(
        int(obj["logical_address"]) + int(obj["bytes"]) for obj in objects
    )
    default_synthetic_base = (object_end + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES
    synthetic = build_synthetic_envelope(
        target=synthetic_write_target,
        start_bytes=synthetic_start_write_bytes,
        uniform_bytes=synthetic_uniform_write_bytes,
        end_bytes=synthetic_end_write_bytes,
        request_bytes=synthetic_write_request_bytes,
        base_address=(
            default_synthetic_base
            if synthetic_write_base_address is None
            else synthetic_write_base_address
        ),
        operation=synthetic_operation,
    )
    kernel_counts, totals = scan_counts(binary_path, objects)
    source_totals = totals.copy()
    if synthetic is not None:
        totals["transactions"] += synthetic.total_requests
        totals["transaction_bytes"] += synthetic.total_bytes
        traffic_name = "read" if synthetic.operation == "R" else "write"
        totals[f"synthetic_{traffic_name}_transactions"] += synthetic.total_requests
        totals[f"synthetic_{traffic_name}_bytes"] += synthetic.total_bytes
    kernel_count = max(kernel_counts) + 1
    # A post-cache stream can legitimately contain no downstream request for
    # a fully hitting kernel.  Retain its ordinal gap and use the observed
    # cadence row of every kernel that does remain.
    cadence = load_cadence(cadence_csv_path, cadence_kernel_offset, kernel_count)

    transaction_digest = hashlib.sha256()
    by_target: dict[str, Counter[str]] = {}
    for line, target, byte_count in iter_transaction_lines(
        binary=binary_path, objects=objects, policy=policy,
        cadence=cadence, kernel_counts=kernel_counts, synthetic=synthetic,
    ):
        transaction_digest.update(line)
        counter = by_target.setdefault(target, Counter())
        counter["transactions"] += 1
        counter["bytes"] += byte_count
    transaction_sha = transaction_digest.hexdigest()
    binary_sha = sha256_file(binary_path)
    logical_sha = binary_sha
    if synthetic is not None:
        logical_sha = hashlib.sha256(
            binary_sha.encode("ascii")
            + json.dumps(asdict(synthetic), sort_keys=True).encode("ascii")
        ).hexdigest()

    hbf_ranges = [
        (int(obj["logical_address"]), int(obj["logical_address"]) + int(obj["bytes"]))
        for obj in objects
        if target_for_kind(str(obj["kind"]), policy) == "HBF_LOGICAL"
    ]
    if hbf_ranges:
        first_lpn = min(begin for begin, _end in hbf_ranges) // PAGE_BYTES
        last_lpn = (max(end for _begin, end in hbf_ranges) + PAGE_BYTES - 1) // PAGE_BYTES
    else:
        first_lpn = 0
        last_lpn = 0
    if synthetic is not None and synthetic.target == "HBF_LOGICAL":
        synthetic_first = synthetic.base_address // PAGE_BYTES
        synthetic_last = (
            synthetic.base_address + synthetic.total_bytes + PAGE_BYTES - 1
        ) // PAGE_BYTES
        if hbf_ranges:
            first_lpn = min(first_lpn, synthetic_first)
            last_lpn = max(last_lpn, synthetic_last)
        else:
            first_lpn = synthetic_first
            last_lpn = synthetic_last
    command = [
        str(simulator_path), "--system-config", str(system_config_path),
        "--enable-hbm", "true" if "HBM" in by_target else "false",
        "--enable-hbf", "true" if "HBF_LOGICAL" in by_target else "false",
        "--enable-external", "false",
        "--static-hbf-blocks-per-plane", "0",
        "--published-hbf-blocks-per-plane", "0",
        "--initial-hbf-logical-first-lpn", str(first_lpn),
        "--initial-hbf-logical-pages", str(last_lpn - first_lpn),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdin is not None
    process.stdin.write(f"BEGIN 0 {logical_sha} {transaction_sha}\n".encode("ascii"))
    streamed = Counter()
    try:
        for line, target, byte_count in iter_transaction_lines(
            binary=binary_path, objects=objects, policy=policy,
            cadence=cadence, kernel_counts=kernel_counts, synthetic=synthetic,
        ):
            process.stdin.write(line)
            streamed["transactions"] += 1
            streamed["transaction_bytes"] += byte_count
            streamed[f"{target}_transactions"] += 1
        process.stdin.write(b"END 0\nQUIT\n")
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise
    require(process.returncode == 0,
            f"HBFSim failed with exit {process.returncode}: {stderr.decode(errors='replace')}")
    receipts = [json.loads(line) for line in stdout.decode("utf-8").splitlines() if line.strip()]
    require(len(receipts) == 3, "HBFSim did not emit ready/pass/stopped receipts")
    require(receipts[0].get("result") == "ready"
            and receipts[1].get("result") == "pass"
            and receipts[2].get("result") == "stopped",
            "HBFSim receipt sequence failed")
    completion = receipts[1]
    require(streamed["transactions"] == totals["transactions"],
            "streamed transaction count disagrees")
    require(streamed["transaction_bytes"] == totals["transaction_bytes"],
            "streamed transaction bytes disagree")
    require(int(completion["transactions"]) == totals["transactions"],
            "HBFSim receipt transaction count disagrees")
    require(int(completion["transaction_bytes"]) == totals["transaction_bytes"],
            "HBFSim receipt bytes disagree")
    require(completion["transaction_trace_sha256"] == transaction_sha,
            "HBFSim receipt digest disagrees")

    receipt_path = output_dir / "receipts.jsonl"
    stderr_path = output_dir / "simulator.stderr"
    receipt_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "streamed compact GPU request replay through HBFSim core",
        "inputs": {
            "plan": str(plan_path), "plan_sha256": sha256_file(plan_path),
            "compact_binary": str(binary_path), "compact_binary_sha256": binary_sha,
            "cadence_csv": str(cadence_csv_path), "cadence_csv_sha256": sha256_file(cadence_csv_path),
            "cadence_kernel_offset": cadence_kernel_offset,
            "simulator": str(simulator_path), "simulator_sha256": sha256_file(simulator_path),
            "system_config": str(system_config_path), "system_config_sha256": sha256_file(system_config_path),
        },
        "policy": policy,
        "transactions": dict(totals),
        "source_transactions": dict(source_totals),
        "synthetic_traffic_envelope": asdict(synthetic) if synthetic is not None else None,
        "synthetic_write_envelope": (
            asdict(synthetic)
            if synthetic is not None and synthetic.operation == "W"
            else None
        ),
        "synthetic_read_envelope": (
            asdict(synthetic)
            if synthetic is not None and synthetic.operation == "R"
            else None
        ),
        "by_target": {key: dict(value) for key, value in sorted(by_target.items())},
        "transaction_trace_sha256": transaction_sha,
        "command": command,
        "completion": completion,
        "receipts": str(receipt_path),
        "stderr": str(stderr_path),
        "not_claimed": [
            "production per-request issue timing",
            "hardware-calibrated HBF latency",
            "physical NVIDIA cache behavior beyond the named upstream transform",
            "the synthetic traffic envelope as an exact NVIDIA cache address trace",
        ],
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--compact-binary", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--cadence-csv", type=Path, required=True)
    parser.add_argument("--cadence-kernel-offset", type=int, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--synthetic-write-target", "--synthetic-target",
        dest="synthetic_write_target", choices=("HBM", "HBF_LOGICAL")
    )
    parser.add_argument(
        "--synthetic-start-write-bytes", "--synthetic-start-bytes",
        dest="synthetic_start_write_bytes", type=int, default=0,
    )
    parser.add_argument(
        "--synthetic-uniform-write-bytes", "--synthetic-uniform-bytes",
        dest="synthetic_uniform_write_bytes", type=int, default=0,
    )
    parser.add_argument(
        "--synthetic-end-write-bytes", "--synthetic-end-bytes",
        dest="synthetic_end_write_bytes", type=int, default=0,
    )
    parser.add_argument(
        "--synthetic-write-request-bytes", type=int, choices=(32, 128), default=32
    )
    parser.add_argument("--synthetic-write-base-address", type=int)
    parser.add_argument("--synthetic-operation", choices=("R", "W"), default="W")
    args = parser.parse_args()
    result = run_replay(
        plan_path=args.plan, binary_path=args.compact_binary,
        policy=args.policy, cadence_csv_path=args.cadence_csv,
        cadence_kernel_offset=args.cadence_kernel_offset,
        simulator_path=args.simulator, system_config_path=args.system_config,
        output_dir=args.output_dir,
        synthetic_write_target=args.synthetic_write_target,
        synthetic_start_write_bytes=args.synthetic_start_write_bytes,
        synthetic_uniform_write_bytes=args.synthetic_uniform_write_bytes,
        synthetic_end_write_bytes=args.synthetic_end_write_bytes,
        synthetic_write_request_bytes=args.synthetic_write_request_bytes,
        synthetic_write_base_address=args.synthetic_write_base_address,
        synthetic_operation=args.synthetic_operation,
    )
    print(json.dumps({
        "status": result["status"], "policy": result["policy"],
        "transactions": result["transactions"]["transactions"],
        "transaction_bytes": result["transactions"]["transaction_bytes"],
        "finish_ns": result["completion"].get("finish_ns"),
        "elapsed_ns": result["completion"].get("elapsed_ns"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

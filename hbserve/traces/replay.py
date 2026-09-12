"""Opt-in phase-serial replay through the dependency-capable HBFSim client.

This is a memory schedule, not a recovered GPU instruction/compute schedule.
Neither input addresses nor request sizes are coalesced. All requests in a
phase are joined before the next phase, including writes on the slower tier.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable, Iterator, Sequence

from hbfsim_client import ResolvedSystemConfig, SimulationSession, Transaction, TransactionBatch

from .common import load_json, new_output, require, save, sha256_file
from ._reference.compact_request_template import RECORD_BYTES, RECORD_STRUCT, OPERATION_NAME
from ._reference.compact_hbfsim_routing import build_routing
from .plan_contract import validate_plan
from ._reference.replay_compact_hbfsim import POLICIES


@dataclass(frozen=True)
class Phase:
    id: str
    transactions: tuple[Transaction, ...]


def phase_batches(phases: Iterable[Phase], *, logical_digest: str,
                  routing_digest: str, completions: bool = False) -> Iterator[tuple[str, TransactionBatch]]:
    """Namespace ids and retain only the last completion/source boundary.

Reference kernels supply no intra-kernel dataflow edges. Explicit phases
may contain local dependencies and a preceding phase's terminal barrier.
Unknown edges are rejected, never silently removed. Empty phases also wait.
"""
    previous_complete = None
    previous_alias: dict[str, str] = {}
    for index, phase in enumerate(phases):
        local: dict[str, str] = {}
        mapped = []
        for tx in phase.transactions:
            tx.validate()
            require(tx.id not in local and tx.id not in previous_alias,
                    f"duplicate operation id: {tx.id}")
            require(all(dep in local or dep in previous_alias for dep in tx.dependencies),
                    f"unknown/forward dependency in {tx.id}")
            dependencies = [local.get(dep, previous_alias.get(dep)) for dep in tx.dependencies]
            if previous_complete:
                dependencies.append(previous_complete)
            mapped_id = f"replay/{index}/op/{tx.id}"
            mapped.append(replace(tx, id=mapped_id,
                                  dependencies=tuple(dict.fromkeys(dependencies))))
            local[tx.id] = mapped_id
        final_id = f"replay/{index}/complete"
        # ALL operations, not the final-listed request or one device's tail.
        waits = tuple(tx.id for tx in mapped) + ((previous_complete,) if previous_complete else ())
        mapped.append(Transaction(final_id, "BARRIER", None, 0, 0, 0.0, dependencies=waits))
        previous_alias = {}
        if phase.transactions and phase.transactions[-1].target == "BARRIER":
            source_tail = phase.transactions[-1].id
            previous_alias[source_tail] = local[source_tail]
        yield phase.id, TransactionBatch(
            batch_id=index, transactions=tuple(mapped), logical_trace_sha256=logical_digest,
            routing_sidecar_sha256=routing_digest, frontier=(final_id,),
            retain=(final_id, *previous_alias.values()), completions=completions,
        )
        previous_complete = final_id


def run_phases(session, phases: Iterable[Phase], *, logical_digest: str,
               routing_digest: str, on_phase: Callable[[dict], None] | None = None,
               completions: bool = False) -> dict:
    origin = session.completed_frontier_ns
    finish = origin
    counts: Counter = Counter(phases=0, memory_requests=0, R_bytes=0, W_bytes=0, dependency_edges=0)
    for name, batch in phase_batches(phases, logical_digest=logical_digest,
                                     routing_digest=routing_digest, completions=completions):
        receipt = session.submit(batch)
        finish = receipt["finish_ns"]
        blocking = receipt["blocking_finish_ns"]
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0 for v in (finish, blocking)),
                "non-finite completion time")
        require(math.isclose(finish, blocking, rel_tol=1e-12, abs_tol=1e-6),
                "completion barrier did not wait for the whole phase")
        require(math.isclose(finish, session.completed_frontier_ns, rel_tol=1e-12, abs_tol=1e-6),
                "session frontier does not match the all-device completion")
        for operations in receipt["transaction_latency_by_target"].values():
            for stats in operations.values():
                tail = stats.get("finish_ns")
                require(tail is None or tail <= finish + 1e-6,
                        "a device operation outlives the declared completion")
        counts["phases"] += 1
        for tx in batch.transactions:
            if tx.target != "BARRIER":
                counts["memory_requests"] += 1
                counts[f"{tx.op}_bytes"] += tx.bytes
            counts["dependency_edges"] += len(tx.dependencies)
        if on_phase:
            on_phase({"phase": name, "completion": receipt})
    return {"memory_finish_ns": finish, "memory_elapsed_ns": finish - origin,
            "batch_origin_ns": origin, "counts": dict(counts),
            "completion_contract": "last all-operation barrier; both tiers and both directions",
            "compute_timing_modeled": False}


class ReferenceInput:
    def __init__(self, root: Path, *, plan_path: Path, policy: str, page_bytes: int,
                 max_phase_records: int):
        self.limit = max_phase_records
        self.path = root / "post-cache.bin"
        self.plan, self.objects, plan_digest = validate_plan(plan_path)
        self.manifest = load_json(root / "post-cache.manifest.json")
        require(self.manifest.get("schema") == {"name": "hbfsim.compact_cache_transform", "version": 1}
                and self.manifest.get("status") == "PASS", "requires a successful reference cache manifest")
        require(self.manifest.get("plan_sha256") == plan_digest, "cache manifest belongs to another plan")
        require(self.manifest.get("record_bytes") == RECORD_BYTES, "compact record width mismatch")
        self.digest = self.manifest["output_sha256"]
        self.routing = build_routing(objects=self.objects, policy=policy, hbf_page_bytes=page_bytes)
        self.input_artifacts = {"plan_sha256": plan_digest,
                                "post-cache.manifest.json": sha256_file(root / "post-cache.manifest.json")}

    def phases(self) -> Iterator[Phase]:
        digest = hashlib.sha256()
        group = []
        last_kernel = None
        counts = Counter()
        with self.path.open("rb") as source:
            while payload := source.read(RECORD_BYTES * 65536):
                require(len(payload) % RECORD_BYTES == 0, "truncated compact input")
                digest.update(payload)
                for obj_index, offset, kernel, size, operation, flags in RECORD_STRUCT.iter_unpack(payload):
                    require(last_kernel is None or kernel >= last_kernel,
                            "kernel ordinals regress; serial-kernels cannot infer concurrent stream dependencies")
                    require(flags == 0, "post-cache replay does not accept cache-admission flags")
                    require(obj_index < len(self.objects) and operation in OPERATION_NAME and size > 0,
                            "malformed compact request")
                    require(offset + size <= self.objects[obj_index]["bytes"], "compact request escapes object")
                    if last_kernel is not None and kernel != last_kernel:
                        yield Phase(f"kernel/{last_kernel}", tuple(group))
                        group = []
                    last_kernel = kernel
                    target = self.routing["target_by_index"][obj_index]
                    address = self.routing["target_base_by_index"][obj_index] + offset
                    group.append(Transaction(f"request/{counts['requests']}", target,
                                             OPERATION_NAME[operation], address, size, 0.0))
                    require(len(group) <= self.limit, "kernel exceeds --max-phase-records; no silent chunk serialization")
                    counts["requests"] += 1
                    counts["bytes"] += size
        if group:
            yield Phase(f"kernel/{last_kernel}", tuple(group))
        require(digest.hexdigest() == self.digest, "post-cache digest mismatch")
        require(counts["requests"] == self.manifest["counts"]["output_requests"]
                and counts["bytes"] == self.manifest["counts"]["output_bytes"], "post-cache census mismatch")


def preflight(source) -> dict:
    counts = Counter(phases=0, memory_requests=0, maximum_phase_records=0)
    for _, batch in phase_batches(source.phases(), logical_digest=source.digest,
                                   routing_digest=source.routing["mapping_table_sha256"]):
        counts["phases"] += 1
        counts["memory_requests"] += sum(tx.target != "BARRIER" for tx in batch.transactions)
        counts["maximum_phase_records"] = max(counts["maximum_phase_records"], len(batch.transactions) - 1)
    return dict(counts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hbserve trace replay", description=__doc__)
    parser.add_argument("--source", choices=("reference",), default="reference")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="reference input's exact generating plan")
    parser.add_argument("--schedule", choices=("serial-kernels",), required=True)
    parser.add_argument("--placement", choices=POLICIES, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, action="append", required=True)
    parser.add_argument("--max-phase-records", type=int, default=1_000_000,
                        help="memory safety cap; never changes modeled concurrency")
    parser.add_argument("--read-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.max_phase_records > 0, "max-phase-records must be positive")
    require(args.schedule == "serial-kernels",
            "schedule does not match the input dialect")
    require(args.plan is not None, "--plan is required for reference")
    started = time.perf_counter()
    config = ResolvedSystemConfig.load(args.system_config)
    kwargs = dict(policy=args.placement, page_bytes=config.hbf_geometry.page_size_bytes,
                  max_phase_records=args.max_phase_records)
    source = ReferenceInput(args.input_root, plan_path=args.plan, **kwargs)
    checked = preflight(source)  # Verify hashes, order, bounds and edges BEFORE starting the engine.
    routing = source.routing
    for row in routing["target_objects"]:
        if row["target"] == "HBM":
            require(row["target_logical_address"] + row["bytes"] <= config.hbm_capacity_bytes,
                    "HBM placement exceeds configured capacity; adjust the declared population/placement")
    config = config.resolve(args.simulator, enable_hbf=routing["enable_hbf"])
    if routing["enable_hbf"]:
        require(routing["initial_hbf_logical_pages"] * config.hbf_geometry.page_size_bytes
                <= config.logical_hbf_capacity_bytes, "HBF placement exceeds logical capacity")
    preflight_seconds = time.perf_counter() - started
    output = new_output(args.output_root)
    setup_started = time.perf_counter()
    session = SimulationSession(simulator_path=args.simulator, system_config=config,
                                enable_hbm=routing["enable_hbm"], enable_hbf=routing["enable_hbf"],
                                initial_hbf_logical_pages=routing["initial_hbf_logical_pages"],
                                read_timeout_s=args.read_timeout_seconds)
    replay_started = time.perf_counter()
    try:
        with (output / "phases.partial.jsonl").open("x", encoding="utf-8") as stream:
            def record(row):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            result = run_phases(session, source.phases(), logical_digest=source.digest,
                                routing_digest=routing["mapping_table_sha256"], on_phase=record)
    except BaseException:
        try:
            session.close()
        except Exception:
            pass  # Preserve the original failure; no successful final artifact.
        raise
    session.close()
    result.update(schema={"name": "hbserve.dependency_trace_replay", "version": 1},
                  status="PASS_DEPENDENCY_REPLAY_NOT_HARDWARE_TIMING_VALIDATION",
                  source=args.source, schedule=args.schedule, input_sha256=source.digest,
                  input_artifacts=source.input_artifacts, preflight=checked,
                  preflight_host_seconds=preflight_seconds,
                  session_setup_host_seconds=replay_started - setup_started,
                  replay_and_close_host_seconds=time.perf_counter() - replay_started,
                  total_host_seconds_through_close=time.perf_counter() - started,
                  routing=routing, session=session.source_receipt(),
                  limitations=["serial phase/kernel memory schedule, not a true GPU compute DAG",
                               "reference has no intra-kernel instruction dependencies",
                               "no request coalescing or request-size change",
                               "memory completion excludes end-of-session HBF persistence drain",
                               "reference terminal GPU-cache drain stays with its recorded kernel",
                               "one phase is buffered; oversized phases fail before replay"])
    (output / "phases.partial.jsonl").rename(output / "phases.jsonl")
    save(output / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "memory_elapsed_ns", "counts")}, sort_keys=True))
    return 0

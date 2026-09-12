"""Bind modeled post-cache requests to native fixed-window transactions.

This provider replaces traffic, not placement or execution. It consumes one
verified reference export and explicit object-to-layout bindings. Native
remappers, sessions and measurement boundaries remain in control.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import hashlib
from pathlib import Path
from typing import Any, Mapping

from hbserve.traces.common import load_json, require, sha256_file
from hbserve.traces.plan_contract import uint, validate_plan
from hbserve.traces._reference.compact_request_template import RECORD_BYTES, RECORD_STRUCT, OPERATION_NAME
from hbserve.windows.memory_trace import CanonicalTraceBatch, LogicalTransaction, MemoryLayout, canonical_sha256
from hbserve.windows.window_contract import FixedFootprintTraceError
from hbserve.windows.window_emitters import FixedFootprintPhase

KIND = "reference_post_cache_memory_window"
BINDING_SCHEMA = {"name": "hbserve.reference_window_binding", "version": 1}
CLASSES = {"weight": "immutable_weight", "kv_cache": "kv", "activation": "metadata", "anonymous": "metadata"}
OBJECT_CLASSES = {"weight": "model_weights", "kv_cache": "kv_cache", "activation": "runtime_buffers", "anonymous": "runtime_buffers"}


def _path(base: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"missing {label}")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class ReferenceWindowTrace:
    layout: MemoryLayout
    population: Mapping[str, Any]
    workload: Mapping[str, Any]
    phases: tuple[FixedFootprintPhase, ...]
    source: Mapping[str, Any]
    traffic_by_kind: Mapping[str, Mapping[str, int]]
    window_shape: str = "reference_post_cache"

    @cached_property
    def read_bytes(self) -> int:
        return sum(phase.read_bytes for phase in self.phases)

    @cached_property
    def write_bytes(self) -> int:
        return sum(phase.write_bytes for phase in self.phases)

    @cached_property
    def digest(self) -> str:
        return canonical_sha256({"generator": "reference", "layout": self.layout.digest,
                                 "source": dict(self.source),
                                 "phases": [phase.canonical() for phase in self.phases]})

    def receipt(self) -> dict[str, Any]:
        total = self.read_bytes + self.write_bytes
        return {
            "schema": {"name": "hbserve.reference_window_trace", "version": 1},
            "generator": "reference", "trace_sha256": self.digest,
            "layout_sha256": self.layout.digest, "window_shape": self.window_shape,
            "measurement_window": {
                "start": self.workload["initial_state"],
                "stages": [self.source["workload"]["phase"], "native_final_drain"],
                "window_shape": self.window_shape,
                "scope": "exact bound source plan; missing source coverage is not filled by coarse traffic",
            },
            "phase_count": len(self.phases), "layer_count": self.layout.num_layers,
            "source": dict(self.source),
            "traffic": {"bytes": total,
                        "reads": {"bytes": self.read_bytes, "fraction": self.read_bytes / total},
                        "writes": {"bytes": self.write_bytes, "fraction": self.write_bytes / total}},
            "traffic_by_source_kind": dict(self.traffic_by_kind),
            "phases": [phase.canonical() for phase in self.phases],
            "invariants": {
                "request_sizes_counts_and_order_preserved": True,
                "address_mapping": "explicit object offsets into native canonical layout",
                "same_trace_for_every_topology": True,
                "native_remappers_and_session_used": True,
                "schedule": "serial-kernels; all requests in each kernel ready together",
                "kernel_completion_waits_for_all_reads_and_writes": True,
                "gpu_cache_recomputed_or_reset_by_replay": False,
                "native_coarse_traffic_added": False,
                "compute_time_modeled": False,
            },
            "not_claimed": ["hardware post-L2 capture", "true GPU issue/return timing",
                            "full model beyond declared source coverage", "arbitrary model/context support",
                            "dynamic serving integration", "accelerated fine execution"],
        }


def build_reference_trace(*, layout: MemoryLayout, population: Mapping[str, Any],
                          workload: Mapping[str, Any], base: Path) -> ReferenceWindowTrace:
    try:
        return _build(layout=layout, population=population, workload=workload, base=base)
    except (ValueError, KeyError, TypeError) as error:
        raise FixedFootprintTraceError(f"reference window: {error}") from error


def _build(*, layout: MemoryLayout, population: Mapping[str, Any],
           workload: Mapping[str, Any], base: Path) -> ReferenceWindowTrace:
    allowed = {"kind", "reference_source", "locality_seed", "compute_time", "initial_state",
               "same_trace_for_every_topology"}
    require(set(workload) == allowed, "reference workload fields differ; native workload knobs are not silently ignored")
    require(workload["kind"] == KIND and workload["compute_time"] == "not_modeled"
            and workload["same_trace_for_every_topology"] is True, "unsupported reference workload contract")
    uint(workload["locality_seed"], "locality seed")
    require(isinstance(workload["initial_state"], str) and workload["initial_state"], "initial state must be explicit")
    binding_path = _path(base, workload["reference_source"], "reference source")
    binding = load_json(binding_path)
    expected = {"schema", "plan", "post_cache_root", "plan_sha256", "post_cache_sha256",
                "layout_sha256", "source_workload", "schedule", "object_bindings", "max_records"}
    require(set(binding) == expected and binding["schema"] == BINDING_SCHEMA, "unsupported reference binding schema/fields")
    require(binding["schedule"] == "serial-kernels", "reference requires explicit serial-kernels")
    require(binding["layout_sha256"] == layout.digest, "binding belongs to a different native layout")
    require(population.get("layout_sha256") == layout.digest, "native population/layout mismatch")
    plan_path = _path(binding_path.parent, binding["plan"], "plan")
    plan, objects, plan_digest = validate_plan(plan_path)
    require(binding["plan_sha256"] == plan_digest, "reference plan digest mismatch")
    require(binding["source_workload"] == plan.get("workload"), "source model/context/phase scope mismatch")
    scope = plan["workload"]
    require(scope.get("phase") in {"prefill", "decode"}, "only explicit prefill or decode plans are supported")
    require(uint(scope.get("layers"), "source layers", minimum=1) == layout.num_layers,
            "source layer count differs from native model; no implicit layer extrapolation")
    root = _path(binding_path.parent, binding["post_cache_root"], "post-cache root")
    manifest_path, binary = root / "post-cache.manifest.json", root / "post-cache.bin"
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == {"name": "hbfsim.compact_cache_transform", "version": 1}
            and manifest.get("status") == "PASS", "requires successful reference cache output")
    require(manifest.get("plan_sha256") == plan_digest and manifest.get("record_bytes") == RECORD_BYTES,
            "cache manifest/plan/record-width mismatch")
    maximum = uint(binding["max_records"], "max records", minimum=1)
    count = uint(manifest["counts"]["output_requests"], "output requests", minimum=1)
    require(count <= maximum, "reference exceeds declared max_records; native window mode buffers the selected window")
    require(binary.stat().st_size == count * RECORD_BYTES, "post-cache size/census mismatch")
    binary_digest = sha256_file(binary)
    require(binary_digest == binding["post_cache_sha256"] == manifest.get("output_sha256"), "post-cache digest mismatch")

    rows = binding["object_bindings"]
    require(isinstance(rows, list) and len(rows) == len(objects), "bind every source object exactly once")
    bound = {}
    spans = []
    by_name = {row["object_id"]: row for row in objects}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"object_id", "region_id", "offset_bytes"}, "malformed object binding")
        name = row["object_id"]
        require(name in by_name and name not in bound, "unknown/duplicate bound object")
        obj = by_name[name]
        region = layout.region(row["region_id"])
        offset = uint(row["offset_bytes"], "region offset")
        require(region.placement_class == CLASSES[obj["kind"]], "binding changes the object's native placement class")
        require(offset + obj["bytes"] <= region.bytes, "bound object exceeds native region")
        address = region.begin + offset
        require(address % 4096 == obj["logical_address"] % 4096, "binding must preserve within-page offsets")
        bound[name] = (region, address)
        spans.append((address, address + obj["bytes"], name))
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        require(left[1] <= right[0], "bound objects overlap; implicit aliasing is unsupported")

    binding_digest = sha256_file(binding_path)
    source = {"binding_sha256": binding_digest, "plan_sha256": plan_digest,
              "post_cache_sha256": binary_digest, "manifest_sha256": sha256_file(manifest_path),
              "workload": scope, "coverage": plan.get("coverage"),
              "cache": manifest.get("cache"), "read_fill": manifest.get("read_fill"),
              "writeback": manifest.get("writeback"),
              "initial_state_declaration": workload["initial_state"],
              "terminal_drain_policy": "kept with the source's recorded kernel; not a measured writeback timestamp"}
    phases = []
    traffic = {kind: Counter(R_bytes=0, W_bytes=0, requests=0) for kind in CLASSES}
    records, last_kernel, transactions, routing, labels = 0, None, [], {}, {}

    def flush(kernel: int) -> None:
        identifier = f"reference/kernel-{kernel}"
        terminal = LogicalTransaction(id=f"{identifier}/complete", op=None, addr=0, bytes=0, issue_ns=0.0,
                                      dependencies=tuple(tx.id for tx in transactions))
        layers = {segment.get("layer") for segment in plan.get("segments", [])
                  if segment.get("expanded_kernel_ordinal_begin", -1) <= kernel
                  < segment.get("expanded_kernel_ordinal_end_exclusive", -1)}
        layer = next(iter(layers)) if len(layers) == 1 else None
        classes = {label["object_class"] for label in labels.values()}
        phases.append(FixedFootprintPhase(
            id=identifier, stage=scope["phase"], layer=layer,
            object_class=next(iter(classes)) if len(classes) == 1 else "mixed_reference",
            objects=tuple(sorted({route["region_id"] for route in routing.values()})),
            trace_group=CanonicalTraceBatch(batch_id=len(phases), transactions=(*transactions, terminal),
                                           layout=layout, routing=dict(routing), audit_labels=dict(labels),
                                           contract_sha256=binding_digest)))

    with binary.open("rb") as stream:
        digest = hashlib.sha256()
        while payload := stream.read(RECORD_BYTES * 65536):
            require(len(payload) % RECORD_BYTES == 0, "truncated reference stream")
            digest.update(payload)
            for index, offset, kernel, size, op, flags in RECORD_STRUCT.iter_unpack(payload):
                require(last_kernel is None or kernel >= last_kernel, "kernel ordinals regress; no timing is guessed")
                require(index < len(objects) and op in OPERATION_NAME and size > 0 and flags == 0, "invalid post-cache record")
                obj = objects[index]
                require(offset + size <= obj["bytes"], "source request escapes object")
                if last_kernel is not None and kernel != last_kernel:
                    flush(last_kernel)
                    transactions, routing, labels = [], {}, {}
                last_kernel = kernel
                region, address = bound[obj["object_id"]]
                identifier = f"reference/kernel-{kernel}/r{records}"
                operation = OPERATION_NAME[op]
                transactions.append(LogicalTransaction(id=identifier, op=operation, addr=address + offset, bytes=size, issue_ns=0.0))
                routing[identifier] = {"region_id": region.id, "group": region.group}
                labels[identifier] = {"stage": scope["phase"], "object_class": OBJECT_CLASSES[obj["kind"]],
                                      "object": region.id, "source_object": obj["object_id"], "source_kind": obj["kind"],
                                      "access_pattern": "reference_record_order", "traffic_boundary": "modeled-post-gpu-cache"}
                traffic[obj["kind"]][operation + "_bytes"] += size
                traffic[obj["kind"]]["requests"] += 1
                records += 1
    require(digest.hexdigest() == binary_digest, "reference changed during binding")
    if transactions:
        flush(last_kernel)
    require(records == count, "reference request census mismatch")
    total = sum(row["R_bytes"] + row["W_bytes"] for row in traffic.values())
    require(total == manifest["counts"]["output_bytes"], "reference byte census mismatch")
    for op in ("R", "W"):
        require(sum(row[op + "_bytes"] for row in traffic.values()) == manifest["counts"].get(f"output_{op.lower()}_bytes", 0),
                f"reference {op} census mismatch")
    return ReferenceWindowTrace(layout, population, dict(workload), tuple(phases), source,
                                {kind: dict(row) for kind, row in traffic.items()})

#!/usr/bin/env python3
"""Canonical address-level memory trace compiler for fixed memory windows.

The output of this module is deliberately below model semantics: address,
read/write, byte count, issue time, and dependency/barrier records. Model,
layer, request, and KV ownership information stays in a digest-bound sidecar
used by the independent remapper and never enters HBFSim's execution API.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hbserve.public_model import derive_public_model_capacity_inputs as derive_model_capacity_inputs


LOGICAL_TRACE_SCHEMA = {
    "name": "hbfsim.canonical_memory_trace",
    "version": 1,
}
ROUTING_SIDECAR_SCHEMA = {
    "name": "hbfsim.memory_remap_sidecar",
    "version": 1,
}
PAGE_SIZE_BYTES = 4096


class MemoryTraceError(ValueError):
    """A memory window cannot produce a canonical memory trace."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MemoryTraceError(
            f"{description} must be an integer >= {minimum}"
        )
    return value


def _finite(value: Any, description: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryTraceError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise MemoryTraceError(
            f"{description} must be finite and >= {minimum}"
        )
    return result


def _align_up(value: int, alignment: int = PAGE_SIZE_BYTES) -> int:
    if value < 0 or alignment <= 0:
        raise MemoryTraceError("alignment operands are invalid")
    return (value + alignment - 1) // alignment * alignment


def _safe_identifier(value: str, description: str) -> str:
    if not value or any(
        not (character.isalnum() or character in "_-.:/")
        for character in value
    ):
        raise MemoryTraceError(f"{description} is not protocol-safe: {value!r}")
    return value


@dataclass(frozen=True)
class LogicalTransaction:
    """One canonical memory operation or generic dependency timer."""

    id: str
    op: str | None
    addr: int
    bytes: int
    issue_ns: float
    duration_ns: float = 0.0
    dependencies: tuple[str, ...] = ()

    @property
    def is_barrier(self) -> bool:
        return self.op is None

    def validate(self) -> None:
        _safe_identifier(self.id, "logical transaction id")
        _finite(self.issue_ns, f"{self.id}.issue_ns")
        _finite(self.duration_ns, f"{self.id}.duration_ns")
        if self.is_barrier:
            if self.addr != 0 or self.bytes != 0:
                raise MemoryTraceError(
                    f"logical barrier {self.id} must have zero address and bytes"
                )
        elif self.op not in {"R", "W"} or self.addr < 0 or self.bytes <= 0:
            raise MemoryTraceError(
                f"logical memory transaction {self.id} is malformed"
            )
        elif self.duration_ns != 0.0:
            raise MemoryTraceError(
                f"logical memory transaction {self.id} has a duration"
            )
        if self.addr > 2**64 - 1 or self.bytes > 2**64 - self.addr:
            raise MemoryTraceError(
                f"logical transaction {self.id} address range overflows"
            )
        if len(set(self.dependencies)) != len(self.dependencies):
            raise MemoryTraceError(
                f"logical transaction {self.id} repeats a dependency"
            )
        for dependency in self.dependencies:
            _safe_identifier(dependency, f"{self.id} dependency")
            if dependency == self.id:
                raise MemoryTraceError(
                    f"logical transaction {self.id} depends on itself"
                )

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "record": "barrier" if self.is_barrier else "memory",
            "op": self.op,
            "addr": self.addr,
            "bytes": self.bytes,
            "issue_ns": self.issue_ns,
            "duration_ns": self.duration_ns,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True)
class MemoryRegion:
    id: str
    begin: int
    bytes: int
    placement_class: str
    group: int | None = None

    @property
    def end(self) -> int:
        return self.begin + self.bytes

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "begin": self.begin,
            "bytes": self.bytes,
            "placement_class": self.placement_class,
            "group": self.group,
        }


@dataclass(frozen=True)
class MemoryLayout:
    model_descriptor_sha256: str
    model_name: str
    vocab_size: int
    alignment_bytes: int
    regions: tuple[MemoryRegion, ...]
    address_space_bytes: int
    num_layers: int
    block_size_tokens: int
    bytes_per_token_per_layer: int
    kv_block_stride_bytes: int
    num_logical_kv_blocks: int
    kv_region_id: str
    metadata_region_id: str
    kv_storage_order: str = "block_major"

    def __post_init__(self) -> None:
        if len(self.model_descriptor_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.model_descriptor_sha256
        ):
            raise MemoryTraceError("model descriptor digest is malformed")
        if not self.model_name:
            raise MemoryTraceError("memory layout model name is empty")
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("alignment_bytes", self.alignment_bytes),
            ("address_space_bytes", self.address_space_bytes),
            ("num_layers", self.num_layers),
            ("block_size_tokens", self.block_size_tokens),
            ("bytes_per_token_per_layer", self.bytes_per_token_per_layer),
            ("kv_block_stride_bytes", self.kv_block_stride_bytes),
            ("num_logical_kv_blocks", self.num_logical_kv_blocks),
        ):
            _integer(value, f"memory layout {name}", minimum=1)
        if (
            self.kv_block_stride_bytes
            != self.block_size_tokens
            * self.bytes_per_token_per_layer
            * self.num_layers
        ):
            raise MemoryTraceError("memory layout KV stride is inconsistent")
        if self.kv_storage_order not in {"block_major", "layer_major"}:
            raise MemoryTraceError(
                "memory layout KV storage order must be block_major or "
                "layer_major"
            )
        if not self.regions:
            raise MemoryTraceError("memory layout has no regions")
        seen: set[str] = set()
        prior_end = 0
        for region in self.regions:
            _safe_identifier(region.id, "memory region id")
            if region.id in seen:
                raise MemoryTraceError(f"duplicate memory region: {region.id}")
            seen.add(region.id)
            if (
                region.begin < prior_end
                or region.begin % self.alignment_bytes
                or region.bytes <= 0
                or region.end > self.address_space_bytes
            ):
                raise MemoryTraceError(
                    f"memory region {region.id} is misaligned or overlapping"
                )
            if region.placement_class not in {
                "immutable_weight",
                "kv",
                "metadata",
            }:
                raise MemoryTraceError(
                    f"memory region {region.id} has an unknown placement class"
                )
            if region.group is not None:
                _integer(region.group, f"memory region {region.id} group")
            prior_end = region.end
        for identifier in (self.kv_region_id, self.metadata_region_id):
            if identifier not in seen:
                raise MemoryTraceError(
                    f"memory layout names an unknown region: {identifier}"
                )
        kv_region = self.region(self.kv_region_id)
        if (
            kv_region.placement_class != "kv"
            or kv_region.bytes
            != self.num_logical_kv_blocks * self.kv_block_stride_bytes
        ):
            raise MemoryTraceError("memory layout KV region is inconsistent")
        if self.region(self.metadata_region_id).placement_class != "metadata":
            raise MemoryTraceError("memory layout metadata region is inconsistent")

    @classmethod
    def build(
        cls,
        model_descriptor_path: Path,
        residency_plan: Mapping[str, Any],
    ) -> "MemoryLayout":
        inputs = derive_model_capacity_inputs(model_descriptor_path)
        try:
            descriptor = json.loads(model_descriptor_path.read_text(encoding="utf-8"))
            architecture = descriptor["architecture"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise MemoryTraceError(
                f"cannot read model vocabulary from {model_descriptor_path}: {error}"
            ) from error
        if not isinstance(architecture, Mapping):
            raise MemoryTraceError("model architecture descriptor must be an object")
        model_name = str(inputs.get("model_name", ""))
        if not model_name:
            raise MemoryTraceError("model descriptor has no model name")
        vocab_size = _integer(
            architecture.get("vocab_size"), "model vocabulary size", minimum=1
        )
        num_layers = _integer(inputs.get("num_layers"), "model num_layers", minimum=1)
        plan_layers = _integer(
            residency_plan.get("num_layers"), "residency num_layers", minimum=1
        )
        if plan_layers != num_layers:
            raise MemoryTraceError("model and residency layer counts differ")
        block_tokens = _integer(
            residency_plan.get("kv_block_size_tokens"),
            "residency KV block size",
            minimum=1,
        )
        kv_page_bytes = _integer(
            residency_plan.get("kv_page_bytes_per_layer"),
            "residency KV page bytes per layer",
            minimum=1,
        )
        if kv_page_bytes % block_tokens:
            raise MemoryTraceError(
                "KV page bytes do not divide into whole per-token traffic"
            )
        kv_stride = _integer(
            residency_plan.get("kv_block_stride_bytes"),
            "residency KV block stride",
            minimum=1,
        )
        if kv_stride != kv_page_bytes * num_layers:
            raise MemoryTraceError("residency KV block stride is inconsistent")
        kv_storage_order = str(
            residency_plan.get("kv_storage_order", "block_major")
        )
        if kv_storage_order not in {"block_major", "layer_major"}:
            raise MemoryTraceError(
                "residency KV storage order must be block_major or layer_major"
            )
        num_blocks = _integer(
            residency_plan.get("num_logical_kv_blocks"),
            "residency logical KV blocks",
            minimum=1,
        )
        objects = inputs["weight_streaming_objects"]
        cursor = 0
        regions: list[MemoryRegion] = []

        def allocate(
            region_id: str,
            byte_count: int,
            placement_class: str,
            group: int | None,
        ) -> None:
            nonlocal cursor
            cursor = _align_up(cursor)
            regions.append(
                MemoryRegion(
                    id=_safe_identifier(region_id, "memory region id"),
                    begin=cursor,
                    bytes=_integer(byte_count, f"{region_id} bytes", minimum=1),
                    placement_class=placement_class,
                    group=group,
                )
            )
            cursor = _align_up(cursor + byte_count)

        allocate(
            "weights/embedding",
            int(objects["embedding"]["bytes_per_object"]),
            "immutable_weight",
            0,
        )
        transformer_layers = objects["transformer_layer"]
        layer_bytes_by_layer = transformer_layers.get(
            "bytes_per_object_by_layer"
        )
        if layer_bytes_by_layer is None:
            layer_bytes_by_layer = [
                int(transformer_layers["bytes_per_object"])
            ] * num_layers
        elif len(layer_bytes_by_layer) != num_layers:
            raise MemoryTraceError(
                "per-layer weight byte list does not match the layer count"
            )
        for layer in range(num_layers):
            allocate(
                f"weights/layer/{layer}",
                int(layer_bytes_by_layer[layer]),
                "immutable_weight",
                layer + 1,
            )
        allocate(
            "weights/final_norm",
            int(objects["final_norm"]["bytes_per_object"]),
            "immutable_weight",
            num_layers + 1,
        )
        allocate(
            "weights/output_head",
            int(objects["output_head"]["bytes_per_object"]),
            "immutable_weight",
            num_layers + 2,
        )
        kv_region_id = "kv/physical_blocks"
        allocate(
            kv_region_id,
            num_blocks * kv_stride,
            "kv",
            None,
        )
        block_table_bytes = residency_plan.get(
            "canonical_block_table_region_bytes"
        )
        if block_table_bytes is not None:
            allocate(
                "metadata/block_table",
                _integer(
                    block_table_bytes,
                    "residency block-table bytes",
                    minimum=1,
                ),
                "metadata",
                None,
            )
        metadata_region_id = "metadata/runtime"
        allocate(
            metadata_region_id,
            _integer(
                residency_plan.get("runtime_overhead_bytes"),
                "residency runtime overhead",
                minimum=1,
            ),
            "metadata",
            None,
        )
        if cursor >= 2**63:
            raise MemoryTraceError(
                "canonical address space exceeds the HBF user-LPN namespace"
            )
        immutable_bytes = sum(
            region.bytes
            for region in regions
            if region.placement_class == "immutable_weight"
        )
        if immutable_bytes != _integer(
            residency_plan.get("immutable_weight_backing_bytes"),
            "residency immutable weight bytes",
            minimum=1,
        ):
            raise MemoryTraceError(
                "canonical weight layout disagrees with Frontier residency"
            )
        return cls(
            model_descriptor_sha256=str(inputs["model_descriptor"]["sha256"]),
            model_name=model_name,
            vocab_size=vocab_size,
            alignment_bytes=PAGE_SIZE_BYTES,
            regions=tuple(regions),
            address_space_bytes=cursor,
            num_layers=num_layers,
            block_size_tokens=block_tokens,
            bytes_per_token_per_layer=kv_page_bytes // block_tokens,
            kv_block_stride_bytes=kv_stride,
            num_logical_kv_blocks=num_blocks,
            kv_region_id=kv_region_id,
            metadata_region_id=metadata_region_id,
            kv_storage_order=kv_storage_order,
        )

    def region(self, region_id: str) -> MemoryRegion:
        for region in self.regions:
            if region.id == region_id:
                return region
        raise MemoryTraceError(f"unknown canonical memory region: {region_id}")

    def region_for_range(self, address: int, byte_count: int) -> MemoryRegion:
        end = address + byte_count
        matches = [
            region
            for region in self.regions
            if region.begin <= address and end <= region.end
        ]
        if len(matches) != 1:
            raise MemoryTraceError(
                f"logical range [{address}, {end}) is not inside one region"
            )
        return matches[0]

    @property
    def kv_page_bytes_per_layer(self) -> int:
        return self.block_size_tokens * self.bytes_per_token_per_layer

    def kv_address(
        self,
        *,
        block_id: int,
        layer: int,
        token_offset: int = 0,
    ) -> int:
        _integer(block_id, "KV block id")
        _integer(layer, "KV layer")
        _integer(token_offset, "KV token offset")
        if block_id >= self.num_logical_kv_blocks:
            raise MemoryTraceError("KV block id is outside the layout")
        if layer >= self.num_layers:
            raise MemoryTraceError("KV layer is outside the layout")
        if token_offset >= self.block_size_tokens:
            raise MemoryTraceError("KV token offset is outside the block")
        region = self.region(self.kv_region_id)
        token_bytes = token_offset * self.bytes_per_token_per_layer
        if self.kv_storage_order == "block_major":
            offset = (
                block_id * self.kv_block_stride_bytes
                + layer * self.kv_page_bytes_per_layer
                + token_bytes
            )
        else:
            offset = (
                (layer * self.num_logical_kv_blocks + block_id)
                * self.kv_page_bytes_per_layer
                + token_bytes
            )
        return region.begin + offset

    def kv_block_for_address(self, address: int) -> int:
        region = self.region(self.kv_region_id)
        if not region.begin <= address < region.end:
            raise MemoryTraceError("KV address is outside the layout")
        offset = address - region.begin
        if self.kv_storage_order == "block_major":
            return offset // self.kv_block_stride_bytes
        layer_span = self.num_logical_kv_blocks * self.kv_page_bytes_per_layer
        return (offset % layer_span) // self.kv_page_bytes_per_layer

    def canonical(self) -> dict[str, Any]:
        result = {
            "schema": ROUTING_SIDECAR_SCHEMA,
            "model_descriptor_sha256": self.model_descriptor_sha256,
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "alignment_bytes": self.alignment_bytes,
            "address_space_bytes": self.address_space_bytes,
            "num_layers": self.num_layers,
            "block_size_tokens": self.block_size_tokens,
            "bytes_per_token_per_layer": self.bytes_per_token_per_layer,
            "kv_block_stride_bytes": self.kv_block_stride_bytes,
            "num_logical_kv_blocks": self.num_logical_kv_blocks,
            "regions": [region.canonical() for region in self.regions],
        }
        if self.kv_storage_order != "block_major":
            result["kv_storage_order"] = self.kv_storage_order
        return result

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(self.canonical())


@dataclass(frozen=True)
class CanonicalTraceBatch:
    batch_id: int
    transactions: tuple[LogicalTransaction, ...]
    layout: MemoryLayout
    routing: Mapping[str, Mapping[str, Any]]
    audit_labels: Mapping[str, Mapping[str, Any]]
    contract_sha256: str

    def __post_init__(self) -> None:
        _integer(self.batch_id, "batch_id")
        if len(self.contract_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.contract_sha256
        ):
            raise MemoryTraceError("memory contract digest is malformed")
        if not self.transactions:
            raise MemoryTraceError("canonical trace batch must not be empty")
        seen: set[str] = set()
        for transaction in self.transactions:
            transaction.validate()
            if transaction.id in seen:
                raise MemoryTraceError(
                    f"duplicate logical transaction id: {transaction.id}"
                )
            for dependency in transaction.dependencies:
                if dependency not in seen:
                    raise MemoryTraceError(
                        f"logical transaction {transaction.id} has a forward or "
                        f"missing dependency {dependency}"
                    )
            seen.add(transaction.id)
        memory_ids = {
            transaction.id
            for transaction in self.transactions
            if not transaction.is_barrier
        }
        if set(self.routing) != memory_ids:
            raise MemoryTraceError(
                "routing sidecar must describe every and only memory transaction"
            )
        transaction_by_id = {
            transaction.id: transaction for transaction in self.memory_transactions
        }
        for identifier, raw_route in self.routing.items():
            if not isinstance(raw_route, Mapping) or set(raw_route) != {
                "region_id",
                "group",
            }:
                raise MemoryTraceError(
                    f"routing record {identifier} does not match schema v1"
                )
            region_id = raw_route.get("region_id")
            if not isinstance(region_id, str):
                raise MemoryTraceError(
                    f"routing record {identifier} has no region ID"
                )
            transaction = transaction_by_id[identifier]
            region = self.layout.region_for_range(
                transaction.addr, transaction.bytes
            )
            if region.id != region_id:
                raise MemoryTraceError(
                    f"routing record {identifier} disagrees with its address"
                )
            group = raw_route.get("group")
            if group is not None:
                _integer(group, f"routing record {identifier} group")
        if not set(self.audit_labels).issubset(seen):
            raise MemoryTraceError("audit labels name an unknown transaction")
        if any(
            not isinstance(label, Mapping)
            for label in self.audit_labels.values()
        ):
            raise MemoryTraceError("audit labels must be objects")

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": LOGICAL_TRACE_SCHEMA,
            "batch_id": self.batch_id,
            "transactions": [
                transaction.canonical() for transaction in self.transactions
            ],
        }

    @cached_property
    def digest(self) -> str:
        # This digest is consumed repeatedly by every remapping plane.  Build it
        # incrementally so a large serving batch never needs a second in-memory
        # copy of its complete canonical JSON document.
        digest = hashlib.sha256()
        digest.update(
            (
                '{"batch_id":'
                + json.dumps(self.batch_id, allow_nan=False)
                + ',"schema":'
                + _canonical_json(LOGICAL_TRACE_SCHEMA)
                + ',"transactions":['
            ).encode("utf-8")
        )
        for index, transaction in enumerate(self.transactions):
            if index:
                digest.update(b",")
            digest.update(_canonical_json(transaction.canonical()).encode("utf-8"))
        digest.update(b"]}")
        return digest.hexdigest()

    @cached_property
    def routing_digest(self) -> str:
        return canonical_sha256(
            {
                "schema": ROUTING_SIDECAR_SCHEMA,
                "layout_sha256": self.layout.digest,
                "records": self.routing,
            }
        )

    @cached_property
    def memory_transactions(self) -> tuple[LogicalTransaction, ...]:
        return tuple(
            transaction
            for transaction in self.transactions
            if not transaction.is_barrier
        )

    @cached_property
    def logical_bytes(self) -> int:
        return sum(transaction.bytes for transaction in self.memory_transactions)

    def with_audit_labels(
        self, labels: Mapping[str, Mapping[str, Any]]
    ) -> "CanonicalTraceBatch":
        return CanonicalTraceBatch(
            batch_id=self.batch_id,
            transactions=self.transactions,
            layout=self.layout,
            routing=self.routing,
            audit_labels=labels,
            contract_sha256=self.contract_sha256,
        )

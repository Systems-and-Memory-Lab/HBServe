#!/usr/bin/env python3
"""Address-only remapping from canonical addresses to physical traffic.

The fixed-slot paper paths deliberately ignore placement classes, routing
labels, and workload semantics.  HBFSim receives only explicit memory-system
targets, addresses, operations, byte counts, timing, and dependencies.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from hbfsim_client.transaction_protocol import (
    HbfGeometry,
    TRANSACTION_TARGETS,
    TransactionProtocolError as RemapError,
    Transaction,
    TransactionBatch,
    hbf_dense_mapping_pages,
    hbf_link_bytes_by_stack as _link_bytes_by_stack,
    require_integer as _integer,
)
from hbserve.windows.memory_trace import (
    CanonicalTraceBatch,
    MemoryLayout,
    LogicalTransaction,
    canonical_sha256,
)


REMAP_RECEIPT_SCHEMA = {
    "name": "hbfsim.memory_remap_receipt",
    "version": 2,
}


def _align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise RemapError("invalid alignment operands")
    return (value + alignment - 1) // alignment * alignment


def _mapped_census(
    transactions: Iterable[Transaction],
) -> dict[str, dict[str, int]]:
    census = {
        target: {"transactions": 0, "bytes": 0}
        for target in TRANSACTION_TARGETS
    }
    for transaction in transactions:
        census[transaction.target]["transactions"] += 1
        census[transaction.target]["bytes"] += transaction.bytes
    return census


def _blocking_frontier(
    transactions: Sequence[Transaction],
    detached: Iterable[str],
) -> tuple[str, ...]:
    """Sinks of the batch DAG that are not detached work.

    Their completion is the batch's blocking frontier: ``elapsed_ns`` and the
    next batch's origin follow them, while detached transactions (for example
    an HBF-to-external offload that only a later restore waits for) still
    execute, still contend for devices, and stay dependable across batches.
    """

    detached_ids = set(detached)
    has_successor = {
        dependency
        for transaction in transactions
        for dependency in transaction.dependencies
    }
    return tuple(
        transaction.id
        for transaction in transactions
        if transaction.id not in has_successor
        and transaction.id not in detached_ids
    )


def _retained_ids(*held: Iterable[str | None]) -> tuple[str, ...]:
    """Every transaction id a remapper still holds as a future dependency.

    The engine forgets ids older than its dependency window unless each
    batch's ``retain`` names them, so the complete held set is declared on
    every batch.
    """

    return tuple(
        dict.fromkeys(
            identifier
            for group in held
            for identifier in group
            if identifier is not None
        )
    )


def _build_receipt(
    *,
    topology: str,
    batch: CanonicalTraceBatch,
    mapped: tuple[Transaction, ...],
    projection: Mapping[str, tuple[str, ...]],
    projected_bytes: Mapping[str, int],
    policy: Mapping[str, Any],
    routing_sidecar_consumed: bool,
    placement_classes_consumed: bool,
) -> dict[str, Any]:
    memory = batch.memory_transactions
    if set(projection) != {transaction.id for transaction in memory}:
        raise RemapError(
            "logical projection must cover every and only memory transaction"
        )
    logical_by_id = {transaction.id: transaction for transaction in memory}
    mapped_ids = {transaction.id for transaction in mapped}
    for logical_id, transaction_ids in projection.items():
        if not transaction_ids or not set(transaction_ids).issubset(mapped_ids):
            raise RemapError(
                f"logical transaction {logical_id} has an invalid projection"
            )
        if projected_bytes.get(logical_id) != logical_by_id[logical_id].bytes:
            raise RemapError(
                f"logical transaction {logical_id} byte projection diverged"
            )
    if sum(projected_bytes.values()) != batch.logical_bytes:
        raise RemapError("remapped logical bytes do not conserve")
    projection_document = {
        logical_id: list(transaction_ids)
        for logical_id, transaction_ids in projection.items()
    }
    return {
        "schema": REMAP_RECEIPT_SCHEMA,
        "result": "pass",
        "topology": topology,
        "source": {
            "canonical_trace_sha256": batch.digest,
            "routing_sidecar_sha256": batch.routing_digest,
            "layout_sha256": batch.layout.digest,
            "contract_sha256": batch.contract_sha256,
        },
        "policy": dict(policy),
        "invariants": {
            "logical_memory_transactions": len(memory),
            "logical_bytes": batch.logical_bytes,
            "projected_memory_transactions": len(projection),
            "projected_bytes": sum(projected_bytes.values()),
            "address_rw_bytes_conserved": True,
            "dependency_order_preserved": True,
            "same_canonical_trace_for_all_scenarios": True,
            "audit_labels_consumed": False,
            "routing_sidecar_consumed_by_policy": routing_sidecar_consumed,
            "placement_classes_consumed_by_policy": placement_classes_consumed,
            "semantic_fields_at_execution_boundary": False,
        },
        "mapped": {
            "transactions": len(mapped),
            "bytes": sum(transaction.bytes for transaction in mapped),
            "by_target": _mapped_census(mapped),
        },
        "projection": {
            "records": len(projection_document),
            "sha256": canonical_sha256(projection_document),
        },
    }


class DirectAttachedRemapper:
    """Static direct attachment across the active HBM and HBF stacks.

    The default balances capacity across canonical addresses. An optional
    complete priority order selects which coarse units occupy HBM instead.
    Both paths compact addresses within each medium and use the logical HBF
    interface without staging or migration.
    """

    topology = "gpu-direct-attached"

    def __init__(
        self,
        *,
        address_space_bytes: int,
        hbm_capacity_bytes: int,
        hbf_geometry: HbfGeometry,
        hbm_stacks: int,
        hbf_stacks: int,
        placement_granularity_bytes: int,
        reserved_hbm_bytes: int = 0,
        reserved_hbf_bytes: int = 0,
        hbm_priority_units: Sequence[int] | None = None,
    ) -> None:
        self.geometry = hbf_geometry
        self.page_size = hbf_geometry.page_size_bytes
        self.address_space_bytes = _integer(
            address_space_bytes, "direct address space", minimum=1
        )
        self.hbm_stacks = _integer(hbm_stacks, "direct HBM stacks")
        self.hbf_stacks = _integer(hbf_stacks, "direct HBF stacks")
        if self.hbm_stacks + self.hbf_stacks == 0:
            raise RemapError("direct attachment needs at least one active stack")
        if self.hbf_stacks and self.geometry.stacks != self.hbf_stacks:
            raise RemapError("direct HBF stack count differs from HBF geometry")
        self.hbm_capacity_bytes = _integer(
            hbm_capacity_bytes, "direct HBM capacity"
        )
        if self.hbm_stacks and self.hbm_capacity_bytes == 0:
            raise RemapError("direct HBM stacks have no physical capacity")
        if not self.hbm_stacks and self.hbm_capacity_bytes != 0:
            raise RemapError("direct HBM capacity is nonzero without HBM stacks")
        self.reserved_hbm_payload_bytes = _integer(
            reserved_hbm_bytes, "direct HBM reserved capacity"
        )
        self.reserved_hbf_payload_bytes = _integer(
            reserved_hbf_bytes, "direct HBF reserved capacity"
        )
        self.reserved_hbm_bytes = _align_up(
            self.reserved_hbm_payload_bytes, self.page_size
        )
        self.reserved_hbf_bytes = _align_up(
            self.reserved_hbf_payload_bytes, self.page_size
        )
        if self.hbm_stacks:
            if self.reserved_hbf_bytes:
                raise RemapError(
                    "direct mixed/all-HBM placement reserves metadata only in HBM"
                )
            if self.reserved_hbm_bytes >= self.hbm_capacity_bytes:
                raise RemapError("direct HBM reservation exhausts capacity")
        else:
            if self.reserved_hbm_bytes:
                raise RemapError("all-HBF direct placement cannot reserve HBM")
            if not self.hbf_stacks:
                raise RemapError("all-HBF direct placement has no HBF stacks")
        self.granularity = _integer(
            placement_granularity_bytes,
            "direct placement granularity",
            minimum=self.page_size,
        )
        if (
            self.address_space_bytes % self.page_size
            or self.granularity % self.page_size
        ):
            raise RemapError(
                "direct address space and placement granularity must be "
                "HBF-page aligned"
            )

        self.total_units = (
            self.address_space_bytes + self.granularity - 1
        ) // self.granularity
        usable_hbm = (
            self.hbm_capacity_bytes - self.reserved_hbm_bytes
            if self.hbm_stacks
            else 0
        )
        if not self.hbf_stacks:
            if self.address_space_bytes > usable_hbm:
                raise RemapError(
                    "all-HBM direct placement cannot contain the canonical "
                    "address space and reservation"
                )
            self.hbm_units = self.total_units
        else:
            self.hbm_units = min(
                self.total_units, usable_hbm // self.granularity
            )
        self.hbf_units = self.total_units - self.hbm_units
        if self.hbf_stacks == 0 and self.hbf_units:
            raise RemapError("direct placement spilled without an HBF tier")
        self._selected_hbm_units: tuple[int, ...] | None = None
        if hbm_priority_units is not None:
            priority = tuple(hbm_priority_units)
            if (len(priority) != self.total_units or
                    any(isinstance(unit, bool) or not isinstance(unit, int) or
                        not 0 <= unit < self.total_units for unit in priority) or
                    len(set(priority)) != self.total_units):
                raise RemapError("HBM priority must rank every placement unit exactly once")
            self._selected_hbm_units = tuple(sorted(priority[:self.hbm_units]))

        # The balanced placement is a closed-form Beatty partition.  Never
        # materialize one target/rank entry per placement unit. A large
        # all-HBM address space would otherwise allocate millions of Python
        # entries even though its mapping is simply the identity.
        final_unit = self.total_units - 1
        final_unit_bytes = (
            self.address_space_bytes - final_unit * self.granularity
        )
        final_target, _ = self._unit_target_and_rank(final_unit)
        final_slack = self.granularity - final_unit_bytes
        self.hbm_resident_payload_bytes = (
            self.hbm_units * self.granularity
            - (final_slack if final_target == "HBM" else 0)
        )
        self.hbf_resident_payload_bytes = (
            self.hbf_units * self.granularity
            - (final_slack if final_target == "HBF_LOGICAL" else 0)
        )
        if (
            self.hbm_resident_payload_bytes
            + self.hbf_resident_payload_bytes
            != self.address_space_bytes
        ):
            raise RemapError("direct placement lost payload bytes")
        # Target-local ranks are dense, so the compact allocation extent equals
        # the payload after accounting for a partial final canonical unit.
        self.hbm_allocation_bytes = self.hbm_resident_payload_bytes
        self.hbf_allocation_bytes = self.hbf_resident_payload_bytes
        if (
            self.reserved_hbm_bytes + self.hbm_allocation_bytes
            > self.hbm_capacity_bytes
        ):
            raise RemapError("direct placement exceeds physical HBM capacity")

        self.logical_population_bytes = (
            self.reserved_hbf_bytes + self.hbf_allocation_bytes
        )
        if self.hbf_units or self.reserved_hbf_bytes:
            logical_pages = self.logical_population_bytes // self.page_size
            mapping_pages = hbf_dense_mapping_pages(
                0, logical_pages, self.geometry
            )
            total_pages = self.geometry.capacity_bytes // self.page_size
            if logical_pages + mapping_pages >= total_pages:
                raise RemapError(
                    "direct HBF image and mapping pages exceed raw capacity"
                )
        self._seen_batches: set[int] = set()
        self.policy_name = "address_only_capacity_balanced_direct_v1"
        self.policy_detail: dict[str, Any] = {}

    @property
    def initial_hbf_logical_first_lpn(self) -> int:
        return 0

    @property
    def initial_hbf_logical_pages(self) -> int:
        return self.logical_population_bytes // self.page_size

    @property
    def post_serving_flush_required(self) -> bool:
        return False

    def _hbm_units_before(self, unit: int) -> int:
        if self._selected_hbm_units is not None:
            return bisect_left(self._selected_hbm_units, unit)
        return unit * self.hbm_units // self.total_units

    def _unit_target_and_rank(self, unit: int) -> tuple[str, int]:
        if not 0 <= unit < self.total_units:
            raise RemapError("direct placement unit is out of range")
        hbm_before = self._hbm_units_before(unit)
        hbm_after = self._hbm_units_before(unit + 1)
        if hbm_after > hbm_before:
            return "HBM", hbm_before
        return "HBF_LOGICAL", unit - hbm_before

    def resident_hbf_bytes(self, address: int, byte_count: int) -> int:
        """HBF-resident bytes of one canonical range under this placement.

        Count whole units through the same rank function used by request
        remapping, and inspect the partial edge units. This supports both
        the balanced partition and a precomputed static priority plan.
        """

        end = address + byte_count
        if address < 0 or byte_count <= 0 or end > self.address_space_bytes:
            raise RemapError(
                "direct residency query exceeds the canonical address space"
            )
        if self.hbm_units == self.total_units:
            return 0
        if self.hbf_units == self.total_units:
            return byte_count
        first_unit = address // self.granularity
        last_unit = (end - 1) // self.granularity
        total = 0
        head_end = min(end, (first_unit + 1) * self.granularity)
        if self._unit_target_and_rank(first_unit)[0] == "HBF_LOGICAL":
            total += head_end - address
        if last_unit > first_unit:
            tail_begin = last_unit * self.granularity
            if self._unit_target_and_rank(last_unit)[0] == "HBF_LOGICAL":
                total += end - tail_begin
            begin_full = first_unit + 1
            full_units = last_unit - begin_full
            hbm_full = (
                self._hbm_units_before(last_unit)
                - self._hbm_units_before(begin_full)
            )
            total += (full_units - hbm_full) * self.granularity
        return total

    def _physical_segments(
        self, address: int, byte_count: int
    ) -> tuple[tuple[str, int, int], ...]:
        end = address + byte_count
        if address < 0 or byte_count <= 0 or end > self.address_space_bytes:
            raise RemapError("direct access exceeds the canonical address space")
        if self.hbm_units == self.total_units:
            return (("HBM", self.reserved_hbm_bytes + address, byte_count),)
        if self.hbf_units == self.total_units:
            return (
                (
                    "HBF_LOGICAL",
                    self.reserved_hbf_bytes + address,
                    byte_count,
                ),
            )
        result: list[tuple[str, int, int]] = []
        cursor = address
        while cursor < end:
            unit = cursor // self.granularity
            unit_begin = unit * self.granularity
            offset = cursor - unit_begin
            take = min(end - cursor, self.granularity - offset)
            target, rank = self._unit_target_and_rank(unit)
            base = (
                self.reserved_hbm_bytes
                if target == "HBM"
                else self.reserved_hbf_bytes
            )
            physical = base + rank * self.granularity + offset
            if (
                result
                and result[-1][0] == target
                and result[-1][1] + result[-1][2] == physical
            ):
                old_target, old_address, old_bytes = result[-1]
                result[-1] = (
                    old_target,
                    old_address,
                    old_bytes + take,
                )
            else:
                result.append((target, physical, take))
            cursor += take
        if sum(segment[2] for segment in result) != byte_count:
            raise RemapError("direct physical segmentation lost bytes")
        return tuple(result)

    def remap(self, batch: CanonicalTraceBatch) -> TransactionBatch:
        if batch.batch_id in self._seen_batches:
            raise RemapError("direct remapper saw a duplicate batch")
        if batch.layout.address_space_bytes != self.address_space_bytes:
            raise RemapError("canonical address space changed during direct replay")
        transactions: list[Transaction] = []
        terminal: dict[str, str] = {}
        projection: dict[str, tuple[str, ...]] = {}
        projected_bytes: dict[str, int] = {}
        bytes_by_id: dict[str, int] = {}
        counter = 0

        def emit(
            *,
            target: str,
            op: str | None,
            addr: int,
            byte_count: int,
            issue_ns: float,
            dependencies: Iterable[str],
            duration_ns: float = 0.0,
        ) -> str:
            nonlocal counter
            identifier = f"direct/b{batch.batch_id}/p{counter}"
            counter += 1
            transactions.append(
                Transaction(
                    id=identifier,
                    target=target,
                    op=op,
                    addr=addr,
                    bytes=byte_count,
                    issue_ns=issue_ns,
                    duration_ns=duration_ns,
                    dependencies=tuple(dict.fromkeys(dependencies)),
                )
            )
            bytes_by_id[identifier] = byte_count
            return identifier

        for logical in batch.transactions:
            dependencies = tuple(terminal[item] for item in logical.dependencies)
            if logical.is_barrier:
                terminal[logical.id] = emit(
                    target="BARRIER",
                    op=None,
                    addr=0,
                    byte_count=0,
                    issue_ns=logical.issue_ns,
                    duration_ns=logical.duration_ns,
                    dependencies=dependencies,
                )
                continue
            projected: list[str] = []
            for target, physical, segment_bytes in self._physical_segments(
                logical.addr, logical.bytes
            ):
                projected.append(
                    emit(
                        target=target,
                        op=logical.op,
                        addr=physical,
                        byte_count=segment_bytes,
                        issue_ns=logical.issue_ns,
                        dependencies=dependencies,
                    )
                )
            if len(projected) == 1:
                terminal[logical.id] = projected[0]
            else:
                terminal[logical.id] = emit(
                    target="BARRIER",
                    op=None,
                    addr=0,
                    byte_count=0,
                    issue_ns=logical.issue_ns,
                    dependencies=projected,
                )
            projection[logical.id] = tuple(projected)
            projected_bytes[logical.id] = sum(
                bytes_by_id[identifier] for identifier in projected
            )

        immutable = tuple(transactions)
        receipt = _build_receipt(
            topology=self.topology,
            batch=batch,
            mapped=immutable,
            projection=projection,
            projected_bytes=projected_bytes,
            policy={
                "name": self.policy_name,
                **self.policy_detail,
                "decision_inputs": ["addr", "bytes"],
                "placement_granularity_bytes": self.granularity,
                "canonical_address_space_bytes": self.address_space_bytes,
                "total_placement_units": self.total_units,
                "hbm_placement_units": self.hbm_units,
                "hbf_placement_units": self.hbf_units,
                "hbm_stacks": self.hbm_stacks,
                "hbf_stacks": self.hbf_stacks,
                "hbm_capacity_bytes": self.hbm_capacity_bytes,
                "reserved_hbm_payload_bytes": self.reserved_hbm_payload_bytes,
                "reserved_hbm_allocation_bytes": self.reserved_hbm_bytes,
                "reserved_hbf_payload_bytes": self.reserved_hbf_payload_bytes,
                "reserved_hbf_allocation_bytes": self.reserved_hbf_bytes,
                "hbm_resident_payload_bytes": (
                    self.hbm_resident_payload_bytes
                ),
                "hbf_resident_payload_bytes": (
                    self.hbf_resident_payload_bytes
                ),
                "hbm_allocation_extent_bytes": self.hbm_allocation_bytes,
                "hbf_allocation_extent_bytes": self.hbf_allocation_bytes,
                "initial_hbf_logical_image": {
                    "mode": (
                        "preloaded_mutable_dense"
                        if self.hbf_units or self.reserved_hbf_bytes
                        else "none"
                    ),
                    "first_lpn": 0,
                    "pages": self.initial_hbf_logical_pages,
                    "contents": "compacted_address_only_hbf_resident_units",
                    "installation_accounting": "setup_excluded_from_serving",
                },
                "gpu_can_address_hbm": bool(self.hbm_stacks),
                "gpu_can_address_hbf": bool(self.hbf_stacks),
                "hbm_staging": False,
                "d2d_migration": False,
                "hbf_write_completion": (
                    "finite_controller_buffer_admission_then_media_drain"
                ),
            },
            routing_sidecar_consumed=False,
            placement_classes_consumed=("layout_placement_class" in self.policy_detail.get("planning_inputs", [])),
        )
        self._seen_batches.add(batch.batch_id)
        return TransactionBatch(
            batch_id=batch.batch_id,
            logical_trace_sha256=batch.digest,
            routing_sidecar_sha256=batch.routing_digest,
            transactions=immutable,
            receipt=receipt,
        )

    def finalize(self) -> TransactionBatch | None:
        return None




@dataclass
class _BackingCacheLine:
    slot: int
    # Half-open backing-page ranges relative to the migration unit.  A
    # migration unit is the cache/replacement object, but it is not the
    # write-back granularity: only backing pages touched by writes become
    # dirty.  Keeping merged ranges avoids a page-sized Python object for
    # every byte of a large, fully dirty cache line.
    dirty_page_ranges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_page_ranges)

    @property
    def dirty_pages(self) -> int:
        return sum(end - begin for begin, end in self.dirty_page_ranges)

    def mark_dirty(
        self,
        *,
        byte_offset: int,
        byte_count: int,
        page_size: int,
        unit_bytes: int,
    ) -> None:
        if (
            byte_offset < 0
            or byte_count <= 0
            or page_size <= 0
            or byte_offset + byte_count > unit_bytes
        ):
            raise RemapError("invalid dirty range inside backing cache line")
        begin = byte_offset // page_size
        end = (byte_offset + byte_count + page_size - 1) // page_size
        merged: list[tuple[int, int]] = []
        inserted = False
        for current_begin, current_end in self.dirty_page_ranges:
            if current_end < begin:
                merged.append((current_begin, current_end))
            elif end < current_begin:
                if not inserted:
                    merged.append((begin, end))
                    inserted = True
                merged.append((current_begin, current_end))
            else:
                begin = min(begin, current_begin)
                end = max(end, current_end)
        if not inserted:
            merged.append((begin, end))
        self.dirty_page_ranges = merged

    def dirty_chunks(
        self,
        *,
        unit_begin: int,
        page_size: int,
        transfer_chunk_bytes: int,
    ) -> Iterable[tuple[int, int]]:
        for page_begin, page_end in self.dirty_page_ranges:
            cursor = unit_begin + page_begin * page_size
            end = unit_begin + page_end * page_size
            while cursor < end:
                byte_count = min(transfer_chunk_bytes, end - cursor)
                yield cursor, byte_count
                cursor += byte_count

    def clear_dirty(self) -> None:
        self.dirty_page_ranges.clear()


@dataclass(frozen=True)
class _ReadAheadCredit:
    release: str
    bytes: int
    generation: int




class HbmFrontedBackingRemapper:
    """Address-only credit-bounded read-ahead/write-back HBM cache.

    Read misses are visible to a finite descriptor window before their GPU
    demand frontier.  Transfer credit remains occupied until the corresponding
    chunk is installed in HBM; installed-but-unconsumed data is separately
    bounded by real HBM cache slots, which cannot be reused before the GPU
    demand completes.  Backing transfers are split into independently
    completed chunks and preserve explicit address-version, cache-slot, and
    architectural-demand dependencies.  Replacement remains migration-unit
    based, while dirty state is tracked and written back at backing-page
    granularity.  No model, layer, phase, KV, or placement annotation is
    consumed.
    """

    def __init__(
        self,
        *,
        address_space_bytes: int,
        hbm_capacity_bytes: int,
        migration_granularity_bytes: int,
        transfer_chunk_bytes: int,
        read_ahead_window_bytes: int,
        backing_kind: str,
        hbf_geometry: HbfGeometry | None = None,
        external_capacity_bytes: int | None = None,
        external_page_size_bytes: int | None = None,
        reserved_hbm_bytes: int = 0,
        policy: str = "address_only_lru",
        kv_priority_ranges: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        if backing_kind not in {"hbf", "external"}:
            raise RemapError("backing kind must be 'hbf' or 'external'")
        if backing_kind == "hbf":
            if hbf_geometry is None:
                raise RemapError("HBF backing requires an HBF geometry")
            if external_capacity_bytes is not None or (
                external_page_size_bytes is not None
            ):
                raise RemapError("HBF backing cannot carry external geometry")
            self.geometry = hbf_geometry
            self.page_size = hbf_geometry.page_size_bytes
            self.backing_capacity_bytes = hbf_geometry.capacity_bytes
            self.backing_target = "HBF_LOGICAL"
            self.topology = "hbm-fronted-hbf"
        else:
            if hbf_geometry is not None:
                raise RemapError("external backing cannot carry HBF geometry")
            self.geometry = None
            self.page_size = _integer(
                external_page_size_bytes,
                "external backing page size",
                minimum=1,
            )
            self.backing_capacity_bytes = _integer(
                external_capacity_bytes,
                "external backing capacity",
                minimum=1,
            )
            self.backing_target = "EXTERNAL"
            self.topology = "hbm-fronted-external"
        self.backing_kind = backing_kind
        self.address_space_bytes = _integer(
            address_space_bytes, "HBM-fronted address space", minimum=1
        )
        self.hbm_capacity_bytes = _integer(
            hbm_capacity_bytes, "HBM-fronted HBM capacity", minimum=1
        )
        self.reserved_hbm_payload_bytes = _integer(
            reserved_hbm_bytes, "HBM-fronted HBM reserved capacity"
        )
        self.reserved_hbm_bytes = _align_up(
            self.reserved_hbm_payload_bytes, self.page_size
        )
        if self.reserved_hbm_bytes >= self.hbm_capacity_bytes:
            raise RemapError(
                "HBM-fronted reservation exhausts physical HBM capacity"
            )
        self.granularity = _integer(
            migration_granularity_bytes,
            "HBM-fronted migration granularity",
            minimum=self.page_size,
        )
        if (
            self.address_space_bytes % self.page_size
            or self.granularity % self.page_size
        ):
            raise RemapError(
                "HBM-fronted address space and migration granularity must be "
                "HBF-page aligned"
            )
        self.transfer_chunk_bytes = _integer(
            transfer_chunk_bytes,
            "HBM-fronted transfer chunk",
            minimum=self.page_size,
        )
        if (
            self.transfer_chunk_bytes % self.page_size
            or self.transfer_chunk_bytes > self.granularity
            or self.granularity % self.transfer_chunk_bytes
        ):
            raise RemapError(
                "HBM-fronted transfer chunk must be page aligned and divide "
                "the migration granularity"
            )
        self.read_ahead_window_bytes = _integer(
            read_ahead_window_bytes,
            "HBM-fronted read-ahead window",
            minimum=self.granularity,
        )
        if self.read_ahead_window_bytes % self.transfer_chunk_bytes:
            raise RemapError(
                "HBM-fronted read-ahead window must be transfer-chunk aligned"
            )
        self.read_ahead_window_chunks = (
            self.read_ahead_window_bytes // self.transfer_chunk_bytes
        )
        if self.backing_kind == "hbf" and self.address_space_bytes > 2**63:
            raise RemapError(
                "HBM-fronted address space exceeds the HBF user namespace"
            )
        if self.address_space_bytes > self.backing_capacity_bytes:
            raise RemapError(
                "canonical address space exceeds the backing capacity"
            )
        # Nonallocating reads still pass through the HBM front. Deduct their
        # bounded staging ring from physical HBM, never grant a free backing
        # read to the GPU or silently allocate extra HBM capacity.
        self.stream_staging_bytes = (
            self.read_ahead_window_bytes if policy != "address_only_lru" else 0
        )
        self.cache_slots = (
            self.hbm_capacity_bytes - self.reserved_hbm_bytes - self.stream_staging_bytes
        ) // self.granularity
        if self.cache_slots <= 0:
            raise RemapError(
                "HBM-fronted cache cannot contain one migration unit"
            )
        if self.read_ahead_window_bytes > self.cache_slots * self.granularity:
            raise RemapError(
                "HBM-fronted read-ahead window exceeds usable HBM cache "
                "capacity"
            )

        if self.backing_kind == "hbf":
            assert self.geometry is not None
            logical_pages = self.address_space_bytes // self.page_size
            mapping_pages = hbf_dense_mapping_pages(
                0, logical_pages, self.geometry
            )
            total_pages = self.geometry.capacity_bytes // self.page_size
            if logical_pages + mapping_pages >= total_pages:
                raise RemapError(
                    "HBM-fronted HBF image and mapping pages exceed raw capacity"
                )
        # Replacement/promotion policy of the HBM front. All four policies
        # share the fill, write-back, credit, and flush machinery; they
        # differ only in which resident unit an eviction selects and in
        # whether a missed READ is admitted (filled into HBM) or served as
        # a nonallocating stream through bounded HBM staging:
        #   address_only_lru     - admit every miss, evict least recently
        #                          used (the reference policy);
        #   decayed_lfu          - fill free slots; at capacity require more
        #                          than a one-observation frequency advantage.
        #                          The triggering read alone cannot win admission
        #                          just by occurring earlier in a cyclic scan;
        #   threshold_promotion  - a read is admitted only on its second
        #                          touch within the decay epoch (single-pass
        #                          streams never enter HBM), writes always
        #                          admit, eviction stays LRU;
        #   class_aware          - KV-range units always admit and are
        #                          evicted (LRU, with write-back) only when
        #                          no non-KV resident remains; non-KV units
        #                          follow threshold promotion and decayed-
        #                          LFU eviction, so weight streams can
        #                          neither flood the cache nor displace the
        #                          KV write front.
        _POLICIES = (
            "address_only_lru",
            "decayed_lfu",
            "threshold_promotion",
            "class_aware",
        )
        if policy not in _POLICIES:
            raise RemapError(
                "HBM-fronted policy must be one of " + ", ".join(_POLICIES)
            )
        self.policy = policy
        if (kv_priority_ranges is not None) != (policy == "class_aware"):
            raise RemapError(
                "kv_priority_ranges is required by exactly the class_aware "
                "policy"
            )
        kv_units: set[int] = set()
        if kv_priority_ranges is not None:
            for begin, end in kv_priority_ranges:
                if not 0 <= begin < end <= self.address_space_bytes:
                    raise RemapError(
                        "KV priority range exceeds the canonical address "
                        "space"
                    )
                kv_units.update(
                    range(begin // self.granularity,
                          (end - 1) // self.granularity + 1)
                )
        self._kv_units = frozenset(kv_units)
        # Decayed access frequencies. _touch_counts covers every unit (it
        # gates threshold promotion); _freq/_freq_buckets cover only the
        # residents an LFU-style eviction may choose from. Counts halve
        # every _decay_every unit accesses, so both stay a bounded window
        # rather than an all-time histogram. Sizes are bounded by the unit
        # count (~30K at full scale), never by traffic.
        self._touch_counts: dict[int, int] = {}
        self._freq: dict[int, int] = {}
        self._freq_buckets: dict[int, OrderedDict[int, None]] = {}
        self._min_freq = 0
        self._accesses_since_decay = 0
        decay_units = (
            (self.address_space_bytes + self.granularity - 1) // self.granularity
            if self.policy == "decayed_lfu" else self.cache_slots
        )
        self._decay_every = max(4 * decay_units, 1024)
        self._resident: OrderedDict[int, _BackingCacheLine] = OrderedDict()
        self._slot_release: dict[int, str] = {}
        self._stream_slot_release: dict[int, str] = {}
        self._next_stream_slot = 0
        # Completion that installs the newest backing-store version for a
        # migration unit.  It is needed only after a dirty eviction; keeping
        # it explicit lets a later speculative read cross unrelated timed
        # barriers without crossing an address dependency.
        self._backing_release: dict[int, str] = {}
        self._backing_version: dict[int, int] = {}
        self._read_ahead_credits: deque[_ReadAheadCredit] = deque()
        self._read_ahead_credit_bytes = 0
        self._read_ahead_credit_generation = 0
        self._read_ahead_peak_credit_bytes = 0
        self._next_unused_slot = 0
        self._seen_batches: set[int] = set()
        self._source_digests: list[str] = []
        self._finalized = False
        self._cumulative = {
            "accessed_bytes": 0,
            "hit_bytes": 0,
            "miss_bytes": 0,
            "promotions": 0,
            "evictions": 0,
            "dirty_eviction_writebacks": 0,
            "eviction_writeback_bytes": 0,
            "eviction_writeback_chunks": 0,
            "fill_bytes": 0,
            "fill_chunks": 0,
            "read_ahead_promotions": 0,
            "read_ahead_bytes": 0,
            "read_ahead_chunks": 0,
            "read_ahead_immediate_chunks": 0,
            "read_ahead_credit_wait_chunks": 0,
            "read_ahead_credit_wait_dependencies": 0,
            "foreground_fill_promotions": 0,
            "foreground_fill_bytes": 0,
            "foreground_fill_chunks": 0,
            "full_overwrite_fill_bypass_bytes": 0,
            "final_dirty_flushes": 0,
            "final_flush_bytes": 0,
            "final_flush_chunks": 0,
            "stream_bypass_reads": 0,
            "stream_bypass_bytes": 0,
            "stream_staging_chunks": 0,
            "stream_transfer_bytes": 0,
            "non_kv_pool_evictions": 0,
            "kv_pool_evictions": 0,
        }

    @property
    def initial_hbf_logical_first_lpn(self) -> int:
        return 0

    @property
    def initial_hbf_logical_pages(self) -> int:
        return (
            self.address_space_bytes // self.page_size
            if self.backing_kind == "hbf"
            else 0
        )

    @property
    def post_serving_flush_required(self) -> bool:
        return True

    def _tracked_in_freq_pool(self, unit: int) -> bool:
        if self.policy == "decayed_lfu":
            return True
        if self.policy == "class_aware":
            return unit not in self._kv_units
        return False

    def _bucket_add(self, unit: int, freq: int) -> None:
        self._freq[unit] = freq
        bucket = self._freq_buckets.get(freq)
        if bucket is None:
            bucket = OrderedDict()
            self._freq_buckets[freq] = bucket
        bucket[unit] = None
        if len(self._freq) == 1 or freq < self._min_freq:
            self._min_freq = freq

    def _bucket_remove(self, unit: int) -> None:
        freq = self._freq.pop(unit, None)
        if freq is None:
            return
        bucket = self._freq_buckets.get(freq)
        if bucket is not None:
            bucket.pop(unit, None)
            if not bucket:
                del self._freq_buckets[freq]

    def _note_access(self, unit: int) -> int:
        """Count one access to `unit`, decay epochs, and return the unit's
        touch count within the current epoch."""
        count = self._touch_counts.get(unit, 0) + 1
        self._touch_counts[unit] = count
        if unit in self._freq:
            self._bucket_remove(unit)
            self._bucket_add(unit, min(count, 1 << 30))
        self._accesses_since_decay += 1
        if self._accesses_since_decay >= self._decay_every:
            self._accesses_since_decay = 0
            self._touch_counts = {
                u: c // 2
                for u, c in self._touch_counts.items()
                if c // 2 > 0
            }
            residents = list(self._freq)
            self._freq.clear()
            self._freq_buckets.clear()
            self._min_freq = 0
            for u in residents:
                self._bucket_add(u, max(self._touch_counts.get(u, 0), 1))
        return self._touch_counts.get(unit, 0)

    def _admit(self, unit: int, op: str, touch_count: int) -> bool:
        if op == "W":
            return True
        if self.policy == "address_only_lru":
            return True
        if self.policy == "decayed_lfu":
            if len(self._resident) < self.cache_slots:
                return True
            while not self._freq_buckets.get(self._min_freq):
                self._min_freq += 1
            # One-observation hysteresis: the current miss alone is not
            # evidence of greater reuse. Without this, a cyclic scan repeatedly
            # evicts equally frequent residents that have not been visited yet
            # in the current pass. Genuinely hotter candidates still enter.
            return touch_count > self._min_freq + 1
        # threshold_promotion and class_aware: reads are admitted only on
        # their second touch within the decay epoch. class_aware KV units
        # enter through writes (the write front) and are then shielded at
        # eviction time; their cold-history READS stream from the backing
        # store like any other single-pass read - admitting them would
        # thrash the write front with data that is read exactly once.
        return touch_count >= 2

    def _on_install(self, unit: int) -> None:
        if self._tracked_in_freq_pool(unit):
            self._bucket_add(
                unit, max(self._touch_counts.get(unit, 0), 1)
            )

    def _pick_victim(self) -> tuple[int, "_BackingCacheLine"]:
        """Pop and return the policy's eviction victim from _resident."""
        if self.policy in {"address_only_lru", "threshold_promotion"}:
            return self._resident.popitem(last=False)
        if self._freq:
            while True:
                bucket = self._freq_buckets.get(self._min_freq)
                if bucket:
                    victim_unit = next(iter(bucket))
                    break
                self._min_freq += 1
            self._bucket_remove(victim_unit)
            if self.policy == "class_aware":
                self._cumulative["non_kv_pool_evictions"] += 1
            return victim_unit, self._resident.pop(victim_unit)
        # class_aware with no non-KV resident left: the KV write front
        # itself overflows, and the coldest KV unit is written back.
        self._cumulative["kv_pool_evictions"] += 1
        return self._resident.popitem(last=False)

    def _unit_bytes(self, unit: int) -> int:
        begin = unit * self.granularity
        if begin >= self.address_space_bytes:
            raise RemapError(
                "HBM-fronted unit starts outside the canonical address space"
            )
        return min(self.granularity, self.address_space_bytes - begin)

    def _link_bytes_by_stack(self, address: int, byte_count: int) -> tuple[int, ...]:
        if self.geometry is None:
            raise RemapError("external backing has no HBF D2D stack mapping")
        return _link_bytes_by_stack(address, byte_count, self.geometry)

    def remap(self, batch: CanonicalTraceBatch) -> TransactionBatch:
        if self._finalized:
            raise RemapError(
                "HBM-fronted remapper cannot accept work after finalization"
            )
        if batch.batch_id in self._seen_batches:
            raise RemapError("HBM-fronted remapper saw a duplicate batch")
        if batch.layout.address_space_bytes != self.address_space_bytes:
            raise RemapError(
                "canonical address space changed during HBM-fronted replay"
            )
        before = dict(self._cumulative)
        resident_units_before = len(self._resident)
        dirty_units_before = sum(line.dirty for line in self._resident.values())
        read_ahead_credit_bytes_before = self._read_ahead_credit_bytes
        read_ahead_credit_entries_before = len(self._read_ahead_credits)
        transactions: list[Transaction] = []
        bytes_by_id: dict[str, int] = {}
        terminal: dict[str, str] = {}
        projection: dict[str, tuple[str, ...]] = {}
        projected_bytes: dict[str, int] = {}
        counter = 0
        logical_by_id = {
            transaction.id: transaction for transaction in batch.transactions
        }

        def emit(
            *,
            target: str,
            op: str | None,
            addr: int,
            byte_count: int,
            issue_ns: float,
            dependencies: Iterable[str],
            duration_ns: float = 0.0,
            stack: int | None = None,
        ) -> str:
            nonlocal counter
            identifier = f"tier/b{batch.batch_id}/p{counter}"
            counter += 1
            transactions.append(
                Transaction(
                    id=identifier,
                    target=target,
                    op=op,
                    addr=addr,
                    bytes=byte_count,
                    issue_ns=issue_ns,
                    duration_ns=duration_ns,
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    stack=stack,
                )
            )
            bytes_by_id[identifier] = byte_count
            return identifier

        def join_at(issue_ns: float, identifiers: Iterable[str]) -> str:
            unique = list(dict.fromkeys(identifiers))
            if not unique:
                raise RemapError("cannot join an empty dependency set")
            if len(unique) == 1:
                return unique[0]
            return emit(
                target="BARRIER",
                op=None,
                addr=0,
                byte_count=0,
                issue_ns=issue_ns,
                dependencies=unique,
            )

        def join(logical: LogicalTransaction, identifiers: list[str]) -> str:
            return join_at(logical.issue_ns, identifiers)

        def chunks(
            address: int, byte_count: int
        ) -> Iterable[tuple[int, int]]:
            cursor = address
            end = address + byte_count
            while cursor < end:
                chunk_bytes = min(self.transfer_chunk_bytes, end - cursor)
                yield cursor, chunk_bytes
                cursor += chunk_bytes

        def hbf_links(
            *,
            target: str,
            op: str,
            address: int,
            byte_count: int,
            issue_ns: float,
            dependency: str,
        ) -> list[str]:
            result: list[str] = []
            for stack, stack_bytes in enumerate(
                self._link_bytes_by_stack(address, byte_count)
            ):
                if stack_bytes:
                    result.append(
                        emit(
                            target=target,
                            op=op,
                            addr=address,
                            byte_count=stack_bytes,
                            issue_ns=issue_ns,
                            dependencies=(dependency,),
                            stack=stack,
                        )
                    )
            return result

        def backing_fill(
            *,
            address: int,
            byte_count: int,
            issue_ns: float,
            dependencies: Iterable[str],
            transfer_dependencies: Iterable[str],
        ) -> list[str]:
            if self.backing_kind == "hbf":
                backing_read = emit(
                    target="HBF_LOGICAL",
                    op="R",
                    addr=address,
                    byte_count=byte_count,
                    issue_ns=issue_ns,
                    dependencies=dependencies,
                )
                transfer_ready = join_at(
                    issue_ns,
                    (backing_read, *transfer_dependencies),
                )
                return hbf_links(
                    target="D2D_HBF_TO_HBM",
                    op="R",
                    address=address,
                    byte_count=byte_count,
                    issue_ns=issue_ns,
                    dependency=transfer_ready,
                )
            backing_read = emit(
                target="EXTERNAL",
                op="R",
                addr=address,
                byte_count=byte_count,
                issue_ns=issue_ns,
                dependencies=dependencies,
            )
            return [
                join_at(
                    issue_ns,
                    (backing_read, *transfer_dependencies),
                )
            ]

        def backing_writeback(
            *,
            address: int,
            byte_count: int,
            issue_ns: float,
            dependency: str,
        ) -> str:
            if self.backing_kind == "hbf":
                write_links = hbf_links(
                    target="D2D_HBM_TO_HBF",
                    op="W",
                    address=address,
                    byte_count=byte_count,
                    issue_ns=issue_ns,
                    dependency=dependency,
                )
                return emit(
                    target="HBF_LOGICAL",
                    op="W",
                    addr=address,
                    byte_count=byte_count,
                    issue_ns=issue_ns,
                    dependencies=write_links,
                )
            return emit(
                target="EXTERNAL",
                op="W",
                addr=address,
                byte_count=byte_count,
                issue_ns=issue_ns,
                dependencies=(dependency,),
            )

        def acquire_read_ahead_credit(
            byte_count: int,
        ) -> tuple[str, ...]:
            if byte_count > self.read_ahead_window_bytes:
                raise RemapError(
                    "one read-ahead chunk exceeds the finite credit window"
                )
            releases: list[str] = []
            while (
                self._read_ahead_credit_bytes + byte_count
                > self.read_ahead_window_bytes
            ):
                if not self._read_ahead_credits:
                    raise RemapError(
                        "read-ahead credit accounting exhausted without a "
                        "completed-install release"
                    )
                credit = self._read_ahead_credits.popleft()
                self._read_ahead_credit_bytes -= credit.bytes
                releases.append(credit.release)
            self._read_ahead_credit_bytes += byte_count
            self._read_ahead_peak_credit_bytes = max(
                self._read_ahead_peak_credit_bytes,
                self._read_ahead_credit_bytes,
            )
            self._cumulative["read_ahead_chunks"] += 1
            if releases:
                unique_releases = tuple(dict.fromkeys(releases))
                self._cumulative["read_ahead_credit_wait_chunks"] += 1
                self._cumulative[
                    "read_ahead_credit_wait_dependencies"
                ] += len(unique_releases)
            else:
                unique_releases = ()
                self._cumulative["read_ahead_immediate_chunks"] += 1
            return unique_releases

        def commit_read_ahead_credit(
            *, release: str, byte_count: int
        ) -> None:
            self._read_ahead_credit_generation += 1
            self._read_ahead_credits.append(
                _ReadAheadCredit(
                    release=release,
                    bytes=byte_count,
                    generation=self._read_ahead_credit_generation,
                )
            )

        def jit_transfer_frontier(
            logical: LogicalTransaction,
            demand_dependencies: tuple[str, ...],
        ) -> tuple[str, ...]:
            """Expose only the immediately preceding timed interval.

            Backing reads are governed by the multi-request byte-credit
            window.  D2D transfer and HBM installation remain just-in-time:
            they may overlap one directly preceding non-memory interval but
            cannot reserve HBM resources across an unbounded demand frontier.
            """

            if logical.op != "R":
                return demand_dependencies
            frontier: list[str] = []
            for dependency_id in logical.dependencies:
                dependency = logical_by_id[dependency_id]
                if dependency.is_barrier and dependency.duration_ns > 0.0:
                    frontier.extend(
                        terminal[parent]
                        for parent in dependency.dependencies
                    )
                else:
                    frontier.append(terminal[dependency_id])
            return tuple(dict.fromkeys(frontier))

        for logical in batch.transactions:
            dependencies = tuple(terminal[item] for item in logical.dependencies)
            if logical.is_barrier:
                terminal[logical.id] = emit(
                    target="BARRIER",
                    op=None,
                    addr=0,
                    byte_count=0,
                    issue_ns=logical.issue_ns,
                    duration_ns=logical.duration_ns,
                    dependencies=dependencies,
                )
                continue

            transfer_dependencies = jit_transfer_frontier(
                logical, dependencies
            )

            logical_end = logical.addr + logical.bytes
            if logical_end > self.address_space_bytes:
                raise RemapError(
                    f"HBM-fronted access {logical.id} exceeds the address space"
                )
            projected: list[str] = []
            completions: list[str] = []
            cursor = logical.addr
            while cursor < logical_end:
                unit = cursor // self.granularity
                unit_begin = unit * self.granularity
                unit_bytes = self._unit_bytes(unit)
                segment_end = min(logical_end, unit_begin + unit_bytes)
                segment_bytes = segment_end - cursor
                self._cumulative["accessed_bytes"] += segment_bytes
                touch_count = self._note_access(unit)

                line = self._resident.pop(unit, None)
                if line is not None:
                    self._cumulative["hit_bytes"] += segment_bytes
                    user_dependencies = list(dependencies)
                    release = self._slot_release.get(line.slot)
                    if release is not None:
                        user_dependencies.append(release)
                    user = emit(
                        target="HBM",
                        op=logical.op,
                        addr=(
                            self.reserved_hbm_bytes
                            + line.slot * self.granularity
                            + (cursor - unit_begin)
                        ),
                        byte_count=segment_bytes,
                        issue_ns=logical.issue_ns,
                        dependencies=user_dependencies,
                    )
                    if logical.op == "W":
                        line.mark_dirty(
                            byte_offset=cursor - unit_begin,
                            byte_count=segment_bytes,
                            page_size=self.page_size,
                            unit_bytes=unit_bytes,
                        )
                    self._resident[unit] = line
                    self._slot_release[line.slot] = user
                    projected.append(user)
                    completions.append(user)
                    cursor = segment_end
                    continue

                self._cumulative["miss_bytes"] += segment_bytes
                if not self._admit(unit, logical.op, touch_count):
                    # Bypass the persistent cache, not the HBM/D2D path.
                    # Each staging slot remains occupied until GPU consumption;
                    # both address-version and slot-release dependencies survive
                    # batch boundaries through retain.
                    bypass_dependencies = list(dependencies)
                    bypass_release = self._backing_release.get(unit)
                    if bypass_release is not None:
                        bypass_dependencies.append(bypass_release)
                    stream_begin = cursor // self.page_size * self.page_size
                    stream_end = _align_up(segment_end, self.page_size)
                    for chunk_address, chunk_bytes in chunks(stream_begin, stream_end - stream_begin):
                        stream_slot = self._next_stream_slot
                        self._next_stream_slot = (
                            stream_slot + 1
                        ) % (self.stream_staging_bytes // self.transfer_chunk_bytes)
                        ready = list(bypass_dependencies)
                        previous_user = self._stream_slot_release.get(stream_slot)
                        if previous_user is not None:
                            ready.append(previous_user)
                        fills = backing_fill(
                            address=chunk_address, byte_count=chunk_bytes,
                            issue_ns=logical.issue_ns, dependencies=ready,
                            transfer_dependencies=transfer_dependencies,
                        )
                        staging_address = (
                            self.reserved_hbm_bytes + self.cache_slots * self.granularity
                            + stream_slot * self.transfer_chunk_bytes
                        )
                        install = emit(
                            target="HBM", op="W", addr=staging_address,
                            byte_count=chunk_bytes, issue_ns=logical.issue_ns,
                            dependencies=fills,
                        )
                        user = emit(
                            target="HBM", op="R",
                            addr=staging_address + max(cursor - chunk_address, 0),
                            byte_count=min(segment_end, chunk_address + chunk_bytes) - max(cursor, chunk_address),
                            issue_ns=logical.issue_ns,
                            dependencies=(install,),
                        )
                        self._stream_slot_release[stream_slot] = user
                        projected.append(user)
                        completions.append(user)
                        self._cumulative["stream_staging_chunks"] += 1
                        self._cumulative["stream_transfer_bytes"] += chunk_bytes
                    self._cumulative["stream_bypass_reads"] += 1
                    self._cumulative["stream_bypass_bytes"] += segment_bytes
                    cursor = segment_end
                    continue
                self._cumulative["promotions"] += 1
                if len(self._resident) < self.cache_slots:
                    slot = self._next_unused_slot
                    self._next_unused_slot += 1
                    eviction_ready: str | None = None
                else:
                    victim_unit, victim = self._pick_victim()
                    slot = victim.slot
                    self._cumulative["evictions"] += 1
                    eviction_ready = self._slot_release.get(slot)
                    if victim.dirty:
                        victim_begin = victim_unit * self.granularity
                        victim_dirty_bytes = (
                            victim.dirty_pages * self.page_size
                        )
                        writeback_completions: list[str] = []
                        for chunk_address, chunk_bytes in victim.dirty_chunks(
                            unit_begin=victim_begin,
                            page_size=self.page_size,
                            transfer_chunk_bytes=self.transfer_chunk_bytes,
                        ):
                            chunk_offset = chunk_address - victim_begin
                            readback = emit(
                                target="HBM",
                                op="R",
                                addr=(
                                    self.reserved_hbm_bytes
                                    + slot * self.granularity
                                    + chunk_offset
                                ),
                                byte_count=chunk_bytes,
                                issue_ns=logical.issue_ns,
                                dependencies=(
                                    ()
                                    if eviction_ready is None
                                    else (eviction_ready,)
                                ),
                            )
                            writeback_completions.append(
                                backing_writeback(
                                    address=chunk_address,
                                    byte_count=chunk_bytes,
                                    issue_ns=logical.issue_ns,
                                    dependency=readback,
                                )
                            )
                            self._cumulative[
                                "eviction_writeback_chunks"
                            ] += 1
                        eviction_ready = join_at(
                            logical.issue_ns, writeback_completions
                        )
                        self._backing_release[victim_unit] = eviction_ready
                        self._backing_version[victim_unit] = (
                            self._backing_version.get(victim_unit, 0) + 1
                        )
                        self._cumulative[
                            "dirty_eviction_writebacks"
                        ] += 1
                        self._cumulative[
                            "eviction_writeback_bytes"
                        ] += victim_dirty_bytes

                full_overwrite = (
                    logical.op == "W"
                    and cursor == unit_begin
                    and segment_bytes == unit_bytes
                )
                if full_overwrite:
                    user_dependencies = list(dependencies)
                    if eviction_ready is not None:
                        user_dependencies.append(eviction_ready)
                    user = emit(
                        target="HBM",
                        op="W",
                        addr=self.reserved_hbm_bytes + slot * self.granularity,
                        byte_count=unit_bytes,
                        issue_ns=logical.issue_ns,
                        dependencies=user_dependencies,
                    )
                    self._cumulative[
                        "full_overwrite_fill_bypass_bytes"
                    ] += unit_bytes
                    line_release = user
                    projected.append(user)
                    completions.append(user)
                else:
                    backing_release = self._backing_release.get(unit)
                    read_ahead = logical.op == "R"
                    chunk_records: list[
                        tuple[str, str | None, int]
                    ] = []
                    unit_users: list[str] = []
                    for chunk_address, chunk_bytes in chunks(
                        unit_begin, unit_bytes
                    ):
                        credit_dependencies = (
                            acquire_read_ahead_credit(chunk_bytes)
                            if read_ahead
                            else ()
                        )
                        fill_dependencies = list(
                            credit_dependencies
                            if read_ahead
                            else dependencies
                        )
                        if backing_release is not None:
                            fill_dependencies.append(backing_release)
                        fill_completions = backing_fill(
                            address=chunk_address,
                            byte_count=chunk_bytes,
                            issue_ns=logical.issue_ns,
                            dependencies=tuple(
                                dict.fromkeys(fill_dependencies)
                            ),
                            transfer_dependencies=transfer_dependencies,
                        )
                        install_dependencies = list(fill_completions)
                        if eviction_ready is not None:
                            install_dependencies.append(eviction_ready)
                        chunk_offset = chunk_address - unit_begin
                        install = emit(
                            target="HBM",
                            op="W",
                            addr=(
                                self.reserved_hbm_bytes
                                + slot * self.granularity
                                + chunk_offset
                            ),
                            byte_count=chunk_bytes,
                            issue_ns=logical.issue_ns,
                            dependencies=install_dependencies,
                        )
                        overlap_begin = max(cursor, chunk_address)
                        overlap_end = min(segment_end, chunk_address + chunk_bytes)
                        user: str | None = None
                        if overlap_begin < overlap_end:
                            user = emit(
                                target="HBM",
                                op=logical.op,
                                addr=(
                                    self.reserved_hbm_bytes
                                    + slot * self.granularity
                                    + (overlap_begin - unit_begin)
                                ),
                                byte_count=overlap_end - overlap_begin,
                                issue_ns=logical.issue_ns,
                                # Prefetch timing is speculative; the GPU
                                # access remains on the complete canonical
                                # dependency frontier.
                                dependencies=tuple(
                                    dict.fromkeys((*dependencies, install))
                                ),
                            )
                            projected.append(user)
                            completions.append(user)
                            unit_users.append(user)
                        chunk_records.append((install, user, chunk_bytes))
                        self._cumulative["fill_chunks"] += 1
                        if not read_ahead:
                            self._cumulative[
                                "foreground_fill_chunks"
                            ] += 1

                    demand_completion = join_at(
                        logical.issue_ns, unit_users
                    )
                    chunk_releases: list[str] = []
                    for install, user, chunk_bytes in chunk_records:
                        line_chunk_release = (
                            user
                            if user is not None
                            else join_at(
                                logical.issue_ns,
                                (install, demand_completion),
                            )
                        )
                        chunk_releases.append(line_chunk_release)
                        if read_ahead:
                            commit_read_ahead_credit(
                                release=install,
                                byte_count=chunk_bytes,
                            )
                    line_release = join_at(
                        logical.issue_ns, chunk_releases
                    )
                    self._cumulative["fill_bytes"] += unit_bytes
                    if read_ahead:
                        self._cumulative["read_ahead_promotions"] += 1
                        self._cumulative["read_ahead_bytes"] += unit_bytes
                    else:
                        self._cumulative["foreground_fill_promotions"] += 1
                        self._cumulative["foreground_fill_bytes"] += unit_bytes

                line = _BackingCacheLine(slot=slot)
                if logical.op == "W":
                    line.mark_dirty(
                        byte_offset=cursor - unit_begin,
                        byte_count=segment_bytes,
                        page_size=self.page_size,
                        unit_bytes=unit_bytes,
                    )
                self._resident[unit] = line
                self._on_install(unit)
                self._slot_release[slot] = line_release
                cursor = segment_end

            terminal[logical.id] = join(logical, completions)
            projection[logical.id] = tuple(projected)
            projected_bytes[logical.id] = sum(
                bytes_by_id[identifier] for identifier in projected
            )
            if projected_bytes[logical.id] != logical.bytes:
                raise RemapError(
                    f"HBM-fronted logical byte projection diverged for {logical.id}"
                )

        batch_counters = {
            key: value - before[key] for key, value in self._cumulative.items()
        }
        queued_credit_bytes = sum(
            credit.bytes for credit in self._read_ahead_credits
        )
        if (
            queued_credit_bytes != self._read_ahead_credit_bytes
            or self._read_ahead_credit_bytes > self.read_ahead_window_bytes
            or self._cumulative["read_ahead_bytes"]
            + self._cumulative["foreground_fill_bytes"]
            != self._cumulative["fill_bytes"]
            or self._cumulative["read_ahead_chunks"]
            + self._cumulative["foreground_fill_chunks"]
            != self._cumulative["fill_chunks"]
            or self._cumulative["read_ahead_immediate_chunks"]
            + self._cumulative["read_ahead_credit_wait_chunks"]
            != self._cumulative["read_ahead_chunks"]
        ):
            raise RemapError("HBM-fronted chunk/credit accounting diverged")
        state_document = {
            "resident_lru": [
                {
                    "unit": unit,
                    "slot": line.slot,
                    "dirty_page_ranges": line.dirty_page_ranges,
                    "dirty_pages": line.dirty_pages,
                }
                for unit, line in self._resident.items()
            ],
            "backing_versions": [
                {"unit": unit, "version": version}
                for unit, version in sorted(self._backing_version.items())
            ],
            "read_ahead_credits": [
                {
                    "generation": credit.generation,
                    "bytes": credit.bytes,
                }
                for credit in self._read_ahead_credits
            ],
            "read_ahead_credit_bytes": self._read_ahead_credit_bytes,
            "next_stream_slot": self._next_stream_slot,
            "touch_counts": sorted(self._touch_counts.items()),
            "accesses_since_decay": self._accesses_since_decay,
        }
        causal_state_document = {
            "slot_release": [
                {"slot": slot, "transaction": transaction}
                for slot, transaction in sorted(self._slot_release.items())
            ],
            "backing_release": [
                {"unit": unit, "transaction": transaction}
                for unit, transaction in sorted(self._backing_release.items())
            ],
            "stream_slot_release": sorted(self._stream_slot_release.items()),
            "read_ahead_credit_release": [
                {
                    "generation": credit.generation,
                    "bytes": credit.bytes,
                    "transaction": credit.release,
                }
                for credit in self._read_ahead_credits
            ],
        }
        immutable = tuple(transactions)
        receipt = _build_receipt(
            topology=self.topology,
            batch=batch,
            mapped=immutable,
            projection=projection,
            projected_bytes=projected_bytes,
            policy={
                "name": (
                    f"{self.policy}_credit_jit_chunked_read_ahead_"
                    "page_dirty_writeback_v7"
                ),
                "decision_inputs": [
                    "addr",
                    "op",
                    "bytes",
                    "issue_ns",
                    "dependencies",
                ],
                "hbm_capacity_bytes": self.hbm_capacity_bytes,
                "reserved_hbm_payload_bytes": self.reserved_hbm_payload_bytes,
                "reserved_hbm_allocation_bytes": self.reserved_hbm_bytes,
                "cache_slots": self.cache_slots,
                "migration_granularity_bytes": self.granularity,
                "transfer_chunk_bytes": self.transfer_chunk_bytes,
                "read_ahead_window_bytes": self.read_ahead_window_bytes,
                "read_ahead_window_chunks": self.read_ahead_window_chunks,
                "cache_usable_bytes": self.cache_slots * self.granularity,
                "stream_staging_bytes": self.stream_staging_bytes,
                "nonallocating_read_path": "backing_read_transfer_hbm_staging_gpu_read",
                "stream_slot_release": "gpu_read_completion",
                "initial_cache_state": "empty_at_session_start",
                "cache_state_persists_across_batches": True,
                "gpu_visible_tier": "HBM_only",
                "backing_kind": self.backing_kind,
                "backing_target": self.backing_target,
                "backing_capacity_bytes": self.backing_capacity_bytes,
                "backing_role": "complete_backing_store",
                "admission": (
                    "free_slot_or_frequency_above_min_resident_plus_one; writes_always"
                    if self.policy == "decayed_lfu" else
                    "second_touch; writes_always"
                    if self.policy in {"threshold_promotion", "class_aware"}
                    else "demand_fill"
                ),
                "fill_scheduling": (
                    "credit_backing_read_ahead_jit_chunked_install"
                ),
                "backing_read_scheduling": (
                    "credit_bounded_multi_request_read_ahead"
                ),
                "transfer_install_scheduling": (
                    "dependency_safe_one_timed_barrier_jit"
                ),
                "fill_completion": (
                    "per_chunk_backing_read_transfer_hbm_install"
                ),
                "read_ahead_descriptor_source": (
                    "canonical_address_dependency_lookahead"
                ),
                "read_ahead_alias_guard": (
                    "per_unit_backing_version_dependency"
                ),
                "read_ahead_bound": (
                    "finite_inflight_byte_credit_plus_hbm_cache_slots"
                ),
                "read_ahead_credit_release": (
                    "hbm_chunk_install_completion"
                ),
                "replacement": self.policy,
                "promotion": (
                    "second_touch_within_decay_epoch"
                    if self.policy in {"threshold_promotion", "class_aware"}
                    else "frequency_admission_with_one_observation_hysteresis"
                    if self.policy == "decayed_lfu"
                    else "every_miss"
                ),
                "touch_decay_every_accesses": self._decay_every,
                "kv_priority_units": len(self._kv_units),
                "write_policy": "write_back",
                "dirty_tracking": "merged_backing_page_ranges",
                "dirty_tracking_granularity_bytes": self.page_size,
                "writeback_scope": "dirty_backing_pages_only",
                "full_unit_write_miss": "skip_obsolete_backing_read",
                "dirty_resident_shutdown_policy": (
                    "required_post_serving_flush_to_backing"
                ),
                "backing_write_completion": (
                    "finite_controller_buffer_admission_then_media_drain"
                    if self.backing_kind == "hbf"
                    else "end_to_end_external_device_completion"
                ),
                "initial_backing_image": {
                    "mode": (
                        "preloaded_mutable_dense"
                        if self.backing_kind == "hbf"
                        else "preexisting_external_dataset"
                    ),
                    "address_base": 0,
                    "bytes": self.address_space_bytes,
                    "contents": "complete_canonical_address_space",
                    "installation_accounting": "setup_excluded_from_serving",
                    "content_values_modeled": False,
                },
                "batch_counters": batch_counters,
                "cumulative_counters": dict(self._cumulative),
                "read_ahead_credit_bytes_before_batch": (
                    read_ahead_credit_bytes_before
                ),
                "read_ahead_credit_entries_before_batch": (
                    read_ahead_credit_entries_before
                ),
                "read_ahead_credit_bytes_after_batch": (
                    self._read_ahead_credit_bytes
                ),
                "read_ahead_credit_entries_after_batch": len(
                    self._read_ahead_credits
                ),
                "read_ahead_peak_credit_bytes": (
                    self._read_ahead_peak_credit_bytes
                ),
                "resident_units_before_batch": resident_units_before,
                "dirty_units_before_batch": dirty_units_before,
                "resident_units_after_batch": len(self._resident),
                "dirty_units_after_batch": sum(
                    line.dirty for line in self._resident.values()
                ),
                "policy_state_sha256": canonical_sha256(state_document),
                "causal_dependency_state_sha256": canonical_sha256(
                    causal_state_document
                ),
            },
            routing_sidecar_consumed=False,
            placement_classes_consumed=False,
        )
        self._seen_batches.add(batch.batch_id)
        self._source_digests.append(batch.digest)
        return TransactionBatch(
            batch_id=batch.batch_id,
            logical_trace_sha256=batch.digest,
            routing_sidecar_sha256=batch.routing_digest,
            transactions=immutable,
            receipt=receipt,
            retain=self._retained_dependency_ids(),
        )

    def _retained_dependency_ids(self) -> tuple[str, ...]:
        return _retained_ids(
            self._slot_release.values(),
            self._stream_slot_release.values(),
            self._backing_release.values(),
            (credit.release for credit in self._read_ahead_credits),
        )

    def finalize(self) -> TransactionBatch | None:
        """Flush dirty backing pages from HBM to the selected medium."""

        if self._finalized:
            raise RemapError("backing remapper was finalized more than once")
        self._finalized = True
        dirty = [
            (unit, line)
            for unit, line in self._resident.items()
            if line.dirty
        ]
        if not dirty:
            return None
        batch_id = max(self._seen_batches, default=-1) + 1
        transactions: list[Transaction] = []
        counter = 0

        def emit(
            *,
            target: str,
            op: str | None,
            addr: int,
            byte_count: int,
            dependencies: Iterable[str],
            stack: int | None = None,
        ) -> str:
            nonlocal counter
            identifier = f"backing/finalize/p{counter}"
            counter += 1
            transactions.append(
                Transaction(
                    id=identifier,
                    target=target,
                    op=op,
                    addr=addr,
                    bytes=byte_count,
                    issue_ns=0.0,
                    dependencies=tuple(dict.fromkeys(dependencies)),
                    stack=stack,
                )
            )
            return identifier

        def join(identifiers: Iterable[str]) -> str:
            unique = list(dict.fromkeys(identifiers))
            if not unique:
                raise RemapError("cannot join an empty flush dependency set")
            if len(unique) == 1:
                return unique[0]
            return emit(
                target="BARRIER",
                op=None,
                addr=0,
                byte_count=0,
                dependencies=unique,
            )

        flushed_units: list[dict[str, Any]] = []
        for unit, line in dirty:
            unit_begin = unit * self.granularity
            dirty_pages = line.dirty_pages
            dirty_bytes = dirty_pages * self.page_size
            release = self._slot_release.get(line.slot)
            durable_chunks: list[str] = []
            chunk_count = 0
            dirty_ranges = list(line.dirty_page_ranges)
            for cursor, chunk_bytes in line.dirty_chunks(
                unit_begin=unit_begin,
                page_size=self.page_size,
                transfer_chunk_bytes=self.transfer_chunk_bytes,
            ):
                chunk_offset = cursor - unit_begin
                readback = emit(
                    target="HBM",
                    op="R",
                    addr=(
                        self.reserved_hbm_bytes
                        + line.slot * self.granularity
                        + chunk_offset
                    ),
                    byte_count=chunk_bytes,
                    dependencies=() if release is None else (release,),
                )
                if self.backing_kind == "hbf":
                    link_ids: list[str] = []
                    for stack, stack_bytes in enumerate(
                        self._link_bytes_by_stack(cursor, chunk_bytes)
                    ):
                        if stack_bytes:
                            link_ids.append(
                                emit(
                                    target="D2D_HBM_TO_HBF",
                                    op="W",
                                    addr=cursor,
                                    byte_count=stack_bytes,
                                    dependencies=(readback,),
                                    stack=stack,
                                )
                            )
                    durable_chunks.append(
                        emit(
                            target="HBF_LOGICAL",
                            op="W",
                            addr=cursor,
                            byte_count=chunk_bytes,
                            dependencies=link_ids,
                        )
                    )
                else:
                    durable_chunks.append(
                        emit(
                            target="EXTERNAL",
                            op="W",
                            addr=cursor,
                            byte_count=chunk_bytes,
                            dependencies=(readback,),
                        )
                    )
                chunk_count += 1
            durable_admission = join(durable_chunks)
            line.clear_dirty()
            self._slot_release[line.slot] = durable_admission
            self._backing_release[unit] = durable_admission
            flushed_units.append(
                {
                    "unit": unit,
                    "slot": line.slot,
                    "bytes": dirty_bytes,
                    "pages": dirty_pages,
                    "page_ranges": dirty_ranges,
                    "chunks": chunk_count,
                }
            )
            self._cumulative["final_dirty_flushes"] += 1
            self._cumulative["final_flush_bytes"] += dirty_bytes
            self._cumulative["final_flush_chunks"] += chunk_count

        source_document = {
            "kind": "post_serving_backing_dirty_flush",
            "backing_kind": self.backing_kind,
            "serving_batch_digests": self._source_digests,
            "flushed_units": flushed_units,
        }
        logical_digest = canonical_sha256(source_document)
        routing_digest = canonical_sha256(
            {"kind": "no_routing_sidecar_for_policy_flush"}
        )
        immutable = tuple(transactions)
        receipt = {
            "schema": REMAP_RECEIPT_SCHEMA,
            "result": "pass",
            "topology": self.topology,
            "kind": "post_serving_policy_flush",
            "source": source_document,
            "policy": {
                "name": "flush_dirty_backing_pages_chunked_v4",
                "backing_kind": self.backing_kind,
                "backing_target": self.backing_target,
                "migration_granularity_bytes": self.granularity,
                "transfer_chunk_bytes": self.transfer_chunk_bytes,
                "dirty_tracking": "merged_backing_page_ranges",
                "dirty_tracking_granularity_bytes": self.page_size,
                "writeback_scope": "dirty_backing_pages_only",
                "flushed_units": len(flushed_units),
                "flushed_chunks": sum(
                    item["chunks"] for item in flushed_units
                ),
                "flushed_bytes": sum(
                    item["bytes"] for item in flushed_units
                ),
                "serving_timing_excluded": True,
                "backing_durability": (
                    "completed_by_subsequent_end_of_session_device_drain"
                    if self.backing_kind == "hbf"
                    else "completed_by_external_transaction_completion"
                ),
                "cumulative_counters": dict(self._cumulative),
            },
            "invariants": {
                "all_dirty_hbm_residents_flushed": not any(
                    line.dirty for line in self._resident.values()
                ),
                "gpu_visible_tier_remained_hbm_only": True,
                "semantic_fields_at_execution_boundary": False,
                "routing_sidecar_consumed_by_policy": False,
                "placement_classes_consumed_by_policy": False,
            },
            "mapped": {
                "transactions": len(immutable),
                "bytes": sum(item.bytes for item in immutable),
                "by_target": _mapped_census(immutable),
            },
        }
        return TransactionBatch(
            batch_id=batch_id,
            logical_trace_sha256=logical_digest,
            routing_sidecar_sha256=routing_digest,
            transactions=immutable,
            receipt=receipt,
            retain=self._retained_dependency_ids(),
        )

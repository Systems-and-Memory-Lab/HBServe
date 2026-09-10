"""Exclusive peer homes for empty-at-start KV and HBF-resident non-KV data."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

from hbfsim_client.transaction_protocol import (
    HbfGeometry,
    Transaction,
    TransactionBatch,
    TransactionProtocolError as RemapError,
    hbf_link_bytes_by_stack,
    require_integer,
)
from hbserve.windows.memory_trace import CanonicalTraceBatch
from hbserve.windows.remap import _align_up, _build_receipt, _retained_ids


PEER_POLICIES = ("static", "capacity_migration")


def _transfer_bytes_by_stack(address: int, byte_count: int, geometry: HbfGeometry) -> tuple[int, ...]:
    counts = [0] * geometry.stacks
    page_size = geometry.page_size_bytes
    if address % page_size:
        first_bytes = min(byte_count, page_size - address % page_size)
        counts[geometry.stack_for_logical_page(address // page_size)] += first_bytes
        address += first_bytes
        byte_count -= first_bytes
    aligned_bytes = byte_count // page_size * page_size
    if aligned_bytes:
        counts = [partial + aligned for partial, aligned in zip(
            counts, hbf_link_bytes_by_stack(address, aligned_bytes, geometry)
        )]
        address += aligned_bytes
        byte_count -= aligned_bytes
    if byte_count:
        counts[geometry.stack_for_logical_page(address // page_size)] += byte_count
    return tuple(counts)


class PeerCapacityError(RemapError):
    """An infeasible object placement, not a failed timing measurement."""

    def __init__(self, code: str, message: str, **details: int) -> None:
        super().__init__(message)
        self.code = code
        self.receipt = {"result": "capacity_oom", "code": code, **details}


@dataclass
class _KvUnit:
    ranges: list[tuple[int, int]] = field(default_factory=list)
    slot: int | None = None
    home: int | None = None
    last_access: str | None = None

    def covers(self, begin: int, end: int) -> bool:
        return any(first <= begin and end <= past for first, past in self.ranges)

    def write(self, begin: int, end: int) -> None:
        merged: list[tuple[int, int]] = []
        for first, past in self.ranges:
            if past < begin:
                merged.append((first, past))
            elif end < first:
                merged.append((begin, end))
                begin, end = first, past
            else:
                begin, end = min(begin, first), max(end, past)
        merged.append((begin, end))
        self.ranges = merged


class PeerKvMigrationRemapper:
    """KV is born on its first write; only capacity pressure can change home.

    Static binds all produced KV to HBM and raises a capacity OOM when it
    fills. Capacity migration instead moves the least recently issued HBM
    unit into a newly allocated HBF home. A demoted unit stays directly
    addressable on HBF; neither mode promotes, duplicates, or preinstalls KV.
    The native engine supplies the usable HBF logical capacity, including
    its mapping, GC, and wear-leveling reserves.
    """

    def __init__(
        self,
        *,
        address_space_bytes: int,
        hbm_capacity_bytes: int,
        migration_granularity_bytes: int,
        transfer_chunk_bytes: int,
        hbf_geometry: HbfGeometry,
        hbf_logical_capacity_bytes: int,
        kv_range: tuple[int, int],
        hbm_stacks: int,
        hbf_stacks: int,
        policy: str = "capacity_migration",
        hbf_mapping_mode: str = "cached",
        initial_kv_state: str = "empty",
    ) -> None:
        if policy not in PEER_POLICIES:
            raise RemapError(f"unsupported peer KV policy: {policy}")
        if hbf_mapping_mode not in {"cached", "full-resident"}:
            raise RemapError("peer KV requires a writable cached or full-resident FTL")
        if initial_kv_state != "empty":
            raise RemapError("peer KV requires empty initial KV; decode-only preinstallation is unsupported")
        self.policy = policy
        self.hbf_mapping_mode = hbf_mapping_mode
        self.address_space_bytes = require_integer(address_space_bytes, "peer address space", minimum=1)
        self.hbm_capacity_bytes = require_integer(hbm_capacity_bytes, "peer HBM capacity", minimum=1)
        self.geometry = hbf_geometry
        self.page_size = self.geometry.page_size_bytes
        self.hbf_logical_capacity_bytes = require_integer(
            hbf_logical_capacity_bytes, "resolved peer HBF logical capacity"
        )
        if (self.hbf_logical_capacity_bytes > self.geometry.capacity_bytes
                or self.hbf_logical_capacity_bytes % self.page_size):
            raise RemapError("resolved peer HBF logical capacity must be page aligned and within raw capacity")
        self.hbm_stacks = require_integer(hbm_stacks, "peer HBM stacks", minimum=1)
        self.hbf_stacks = require_integer(hbf_stacks, "peer HBF stacks", minimum=1)
        if self.hbf_stacks != self.geometry.stacks:
            raise RemapError("peer HBF stack count disagrees with geometry")
        self.granularity = require_integer(migration_granularity_bytes, "peer migration granularity", minimum=1)
        self.transfer_chunk_bytes = require_integer(transfer_chunk_bytes, "peer transfer chunk", minimum=1)
        if self.granularity % self.page_size or self.granularity % self.transfer_chunk_bytes:
            raise RemapError("peer granularity must be a multiple of the HBF page and transfer chunk")
        if not isinstance(kv_range, tuple) or len(kv_range) != 2:
            raise RemapError("peer KV range must be a (begin, end) tuple")
        kv_begin = require_integer(kv_range[0], "peer KV begin")
        kv_end = require_integer(kv_range[1], "peer KV end", minimum=1)
        if not 0 <= kv_begin < kv_end <= self.address_space_bytes:
            raise RemapError("peer KV range is out of bounds")
        if kv_begin % self.page_size or kv_end % self.page_size:
            raise RemapError("peer KV range must be HBF-page aligned; mixed KV/static boundary pages are unsupported")
        self.kv_range = (kv_begin, kv_end)
        self.kv_bytes = kv_end - kv_begin
        self.kv_units = _align_up(self.kv_bytes, self.granularity) // self.granularity
        self.slots = self.hbm_capacity_bytes // self.granularity
        if self.slots == 0:
            raise PeerCapacityError("hbm_no_kv_slot", "peer HBM cannot hold one KV migration unit",
                                    hbm_capacity_bytes=self.hbm_capacity_bytes,
                                    migration_granularity_bytes=self.granularity)
        self.static_flash_bytes = self.address_space_bytes - self.kv_bytes
        self._home_base = _align_up(self.static_flash_bytes, self.page_size)
        if self._home_base > self.hbf_logical_capacity_bytes:
            raise PeerCapacityError("hbf_static_image_capacity_exceeded",
                                    "peer non-KV HBF image exceeds resolved logical capacity",
                                    static_image_bytes=self._home_base,
                                    hbf_logical_capacity_bytes=self.hbf_logical_capacity_bytes)
        self._home_capacity_units = (0 if policy == "static" else min(
            max(0, self.kv_units - self.slots),
            (self.hbf_logical_capacity_bytes - self._home_base) // self.granularity,
        ))
        self.flash_logical_bytes = self._home_base + self._home_capacity_units * self.granularity
        self.topology = "peer_hbm_hbf"
        self._resident: OrderedDict[int, _KvUnit] = OrderedDict()
        self._units: dict[int, _KvUnit] = {}
        self._next_slot = 0
        self._next_home = 0
        self._seen_batches: set[int] = set()
        self._finalized = False
        self._cumulative = {
            "kv_units_homed": 0,
            "demotions": 0,
            "demoted_bytes": 0,
            "hbm_kv_read_bytes": 0,
            "hbm_kv_write_bytes": 0,
            "hbf_kv_read_bytes": 0,
            "hbf_kv_write_bytes": 0,
            "non_kv_read_bytes": 0,
            "non_kv_write_bytes": 0,
        }

    @property
    def initial_hbf_logical_first_lpn(self) -> int:
        return 0

    @property
    def initial_hbf_logical_pages(self) -> int:
        return self._home_base // self.page_size

    @property
    def post_serving_flush_required(self) -> bool:
        return False

    def finalize(self) -> None:
        self._finalized = True
        return None

    def _image_receipt(self) -> dict:
        return {
            "mode": "preloaded_non_kv_only",
            "first_lpn": 0,
            "pages": self.initial_hbf_logical_pages,
            "payload_bytes": self.static_flash_bytes,
            "contents": "non_KV_content_compacted_with_the_entire_KV_range_removed",
            "installation_accounting": "setup_excluded_from_serving",
            "kv_initially_populated_bytes": 0,
            "future_kv_homes_preinstalled": False,
            "all_preinstalled_pages_read_only": False,
        }

    def _segments(self, address: int, byte_count: int):
        end = address + byte_count
        if address < 0 or end > self.address_space_bytes:
            raise RemapError("peer access exceeds the canonical address space")
        kv_begin, kv_end = self.kv_range
        cursor = address
        while cursor < end:
            if cursor < kv_begin:
                past = min(end, kv_begin)
                yield None, cursor, past - cursor
            elif cursor >= kv_end:
                past = end
                yield None, cursor - self.kv_bytes, past - cursor
            else:
                unit, offset = divmod(cursor - kv_begin, self.granularity)
                past = min(end, kv_end, cursor + self.granularity - offset)
                yield unit, offset, past - cursor
            cursor = past

    def _check_capacity(self, live_units: int) -> None:
        if self.policy == "static" and live_units > self.slots:
            raise PeerCapacityError("static_kv_hbm_capacity_exceeded",
                                    "peer static KV capacity exhausted; HBF spill is forbidden",
                                    required_kv_units=live_units, hbm_slots=self.slots,
                                    required_hbm_bytes=live_units * self.granularity,
                                    hbm_capacity_bytes=self.hbm_capacity_bytes)
        if live_units > self.slots + self._home_capacity_units:
            raise PeerCapacityError("peer_total_kv_capacity_exceeded",
                                    "peer KV exceeds HBM slots plus available HBF homes",
                                    required_kv_units=live_units, hbm_slots=self.slots,
                                    hbf_home_capacity_units=self._home_capacity_units)

    def preflight(self, batches: Iterable[CanonicalTraceBatch]) -> dict:
        """Validate the complete empty-to-grown trace without changing placement.

        Virtual address reservations are not live KV. Capacity depends on
        units actually born in this trace. Reads must be covered by earlier
        writes at byte granularity, including holes within a written unit.
        There is no deallocation in this fixed-window contract.
        """
        units: dict[int, _KvUnit] = {}
        seen: set[int] = set()
        for batch in batches:
            if batch.layout.address_space_bytes != self.address_space_bytes:
                raise RemapError("canonical address space changed during peer preflight")
            if batch.batch_id in seen:
                raise RemapError("peer preflight saw a duplicate batch")
            seen.add(batch.batch_id)
            for logical in batch.memory_transactions:
                for unit, offset, byte_count in self._segments(logical.addr, logical.bytes):
                    if unit is None:
                        continue
                    state = units.get(unit)
                    if logical.op == "R":
                        if state is None or not state.covers(offset, offset + byte_count):
                            raise RemapError(f"peer KV read before write: {logical.id}, unit {unit}, offset {offset}")
                    else:
                        if state is None:
                            self._check_capacity(len(units) + 1)
                            state = units[unit] = _KvUnit()
                        state.write(offset, offset + byte_count)
        if not seen:
            raise RemapError("peer preflight requires a nonempty trace")
        return {
            "result": "pass",
            "policy": self.policy,
            "capacity_semantics": "exclusive_KV_homes_additive_HBM_and_available_HBF",
            "initial_hbf_logical_image": self._image_receipt(),
            "initial_hbm_kv_bytes": 0,
            "trace_batches": len(seen),
            "declared_virtual_kv_bytes": self.kv_bytes,
            "peak_live_kv_units": len(units),
            "peak_written_kv_bytes": sum(past - first for state in units.values() for first, past in state.ranges),
            "peak_hbm_kv_slot_bytes": min(len(units), self.slots) * self.granularity,
            "peak_demoted_home_bytes": max(0, len(units) - self.slots) * self.granularity,
            "hbf_static_payload_bytes": self.static_flash_bytes,
            "hbf_logical_capacity_bytes": self.hbf_logical_capacity_bytes,
            "hbf_home_capacity_units": self._home_capacity_units,
            "kv_read_before_write": False,
        }

    def remap(self, batch: CanonicalTraceBatch) -> TransactionBatch:
        if self._finalized:
            raise RemapError("peer remapper is finalized")
        if batch.batch_id in self._seen_batches:
            raise RemapError("peer remapper saw a duplicate batch")
        if batch.layout.address_space_bytes != self.address_space_bytes:
            raise RemapError("canonical address space changed during peer remapping")
        self._seen_batches.add(batch.batch_id)
        before = dict(self._cumulative)
        transactions: list[Transaction] = []
        bytes_by_id: dict[str, int] = {}
        terminal: dict[str, str] = {}
        projection: dict[str, tuple[str, ...]] = {}
        projected_bytes: dict[str, int] = {}

        def emit(target, op, address, byte_count, issue_ns, dependencies=(), duration_ns=0.0, stack=None):
            identifier = f"peer/b{batch.batch_id}/p{len(transactions)}"
            transactions.append(Transaction(
                id=identifier, target=target, op=op, addr=address, bytes=byte_count,
                issue_ns=issue_ns, duration_ns=duration_ns,
                dependencies=tuple(dict.fromkeys(dependencies)), stack=stack,
            ))
            bytes_by_id[identifier] = byte_count
            return identifier

        def join(identifiers, issue_ns):
            unique = tuple(dict.fromkeys(identifiers))
            return unique[0] if len(unique) == 1 else emit("BARRIER", None, 0, 0, issue_ns, unique)

        def demote(issue_ns, dependencies):
            state = self._resident.popitem(last=False)[1]
            slot = state.slot
            state.home = self._next_home
            self._next_home += 1
            source_dependencies = list(dependencies)
            if state.last_access is not None:
                source_dependencies.append(state.last_access)
            writes = []
            copied_bytes = 0
            for begin, end in state.ranges:
                cursor = begin
                while cursor < end:
                    byte_count = min(self.transfer_chunk_bytes, end - cursor)
                    address = self._home_base + state.home * self.granularity + cursor
                    read = emit("HBM", "R", slot * self.granularity + cursor, byte_count,
                                issue_ns, source_dependencies)
                    links = [emit("D2D_HBM_TO_HBF", "W", address, stack_bytes,
                                  issue_ns, (read,), stack=stack)
                             for stack, stack_bytes in enumerate(_transfer_bytes_by_stack(address, byte_count, self.geometry))
                             if stack_bytes]
                    writes.append(emit("HBF_LOGICAL", "W", address, byte_count, issue_ns, links))
                    copied_bytes += byte_count
                    cursor += byte_count
            state.last_access = join(writes, issue_ns)
            state.slot = None
            self._cumulative["demotions"] += 1
            self._cumulative["demoted_bytes"] += copied_bytes
            return slot, state.last_access

        for logical in batch.transactions:
            dependencies = tuple(terminal[identifier] for identifier in logical.dependencies)
            if logical.is_barrier:
                terminal[logical.id] = emit("BARRIER", None, 0, 0, logical.issue_ns,
                                            dependencies, logical.duration_ns)
                continue
            projected = []
            for unit, offset, byte_count in self._segments(logical.addr, logical.bytes):
                if unit is None:
                    projected.append(emit("HBF_LOGICAL", logical.op, offset, byte_count,
                                          logical.issue_ns, dependencies))
                    counter = "non_kv_read_bytes" if logical.op == "R" else "non_kv_write_bytes"
                    self._cumulative[counter] += byte_count
                    continue
                state = self._units.get(unit)
                if logical.op == "R" and (state is None or not state.covers(offset, offset + byte_count)):
                    raise RemapError(f"peer KV read before write: {logical.id}, unit {unit}, offset {offset}")
                user_dependencies = list(dependencies)
                if state is None:
                    self._check_capacity(len(self._units) + 1)
                    if self._next_slot < self.slots:
                        slot = self._next_slot
                        self._next_slot += 1
                    else:
                        slot, released = demote(logical.issue_ns, dependencies)
                        user_dependencies.append(released)
                    state = self._units[unit] = _KvUnit(slot=slot)
                    self._resident[unit] = state
                    self._cumulative["kv_units_homed"] += 1
                if state.last_access is not None:
                    user_dependencies.append(state.last_access)
                if state.slot is not None:
                    target = "HBM"
                    address = state.slot * self.granularity + offset
                    self._resident.move_to_end(unit)
                else:
                    target = "HBF_LOGICAL"
                    address = self._home_base + state.home * self.granularity + offset
                user = emit(target, logical.op, address, byte_count, logical.issue_ns, user_dependencies)
                state.last_access = user
                if logical.op == "W":
                    state.write(offset, offset + byte_count)
                prefix = "hbm" if target == "HBM" else "hbf"
                suffix = "read_bytes" if logical.op == "R" else "write_bytes"
                self._cumulative[f"{prefix}_kv_{suffix}"] += byte_count
                projected.append(user)
            terminal[logical.id] = join(projected, logical.issue_ns)
            projection[logical.id] = tuple(projected)
            projected_bytes[logical.id] = sum(bytes_by_id[identifier] for identifier in projected)

        immutable = tuple(transactions)
        retained = _retained_ids(state.last_access for state in self._units.values())
        receipt = _build_receipt(
            topology=self.topology, batch=batch, mapped=immutable,
            projection=projection, projected_bytes=projected_bytes,
            policy={
                "name": self.policy,
                "decision_inputs": ["configured_kv_range", "addr", "op", "bytes", "logical_issue_order"],
                "placement_granularity_bytes": self.granularity,
                "transfer_chunk_bytes": self.transfer_chunk_bytes,
                "kv_range": list(self.kv_range),
                "hbm_capacity_bytes": self.hbm_capacity_bytes,
                "hbf_logical_capacity_bytes": self.hbf_logical_capacity_bytes,
                "hbf_mapping_mode": self.hbf_mapping_mode,
                "hbm_slots": self.slots,
                "hbf_home_capacity_units": self._home_capacity_units,
                "hbm_live_kv_units": len(self._resident),
                "hbf_live_kv_units": self._next_home,
                "capacity_semantics": "exclusive_home_no_duplicate_cache",
                "kv_birth": "first_write_only",
                "demotion": "none" if self.policy == "static" else "least_recently_issued_HBM_KV_unit",
                "re_promotion": "none",
                "d2d_migration": self.policy == "capacity_migration",
                "copied_ranges": "only_initialized_bytes_no_unwritten_holes",
                "slot_reuse": "after_source_read_D2D_and_HBF_write_admission_complete",
                "initial_hbf_logical_image": self._image_receipt(),
                "batch_counters": {key: self._cumulative[key] - before[key] for key in before},
                "cumulative_counters": dict(self._cumulative),
                "hbf_write_completion": "controller_admission_then_native_session_final_media_drain",
                "shutdown_flush": "no_duplicate_cache_to_flush_HBM_KV_remains_volatile",
                "batch_frontier": "all_mapped_work_no_detached_transfers",
            },
            routing_sidecar_consumed=False, placement_classes_consumed=True,
        )
        return TransactionBatch(batch_id=batch.batch_id, logical_trace_sha256=batch.digest,
                                routing_sidecar_sha256=batch.routing_digest,
                                transactions=immutable, receipt=receipt, retain=retained)

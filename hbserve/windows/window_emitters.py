#!/usr/bin/env python3
"""Deterministic primitives and phase assembly for the window generator.

Everything here is emission machinery with no validation and no policy:
hashing/permutation/routing-draw primitives, the operation and phase
dataclasses, canonical phase assembly, and the per-context block
partition. The orchestrator in `fixed_footprint_trace` sequences these
into windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping, Sequence

from hbserve.windows.memory_trace import (
    CanonicalTraceBatch,
    MemoryLayout,
    LogicalTransaction,
)
from hbserve.windows.window_contract import _fail

def _mix64(value: int) -> int:
    mask = 2**64 - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _permutation(size: int, seed: int) -> tuple[int, ...]:
    """Return a deterministic Fisher-Yates permutation."""

    values = list(range(size))
    state = seed & (2**64 - 1)
    for index in range(size - 1, 0, -1):
        state = _mix64(state)
        selected = state % (index + 1)
        values[index], values[selected] = values[selected], values[index]
    return tuple(values)


def _zipf_cumulative(size: int, exponent: float) -> tuple[float, ...]:
    """Cumulative routing mass over expert ranks (rank 0 is the hottest)."""

    weights = [1.0 / float(rank + 1) ** exponent for rank in range(size)]
    total = sum(weights)
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cumulative.append(running)
    cumulative[-1] = 1.0
    return tuple(cumulative)


def _routed_expert_draw(
    *,
    cumulative: Sequence[float],
    rank_to_expert: Sequence[int],
    activated: int,
    seed: int,
    layer: int,
    context_id: int,
    token: int,
    previous: Sequence[int] | None = None,
    reuse_probability: float = 0.0,
) -> tuple[int, ...]:
    """Draw the token's distinct routed experts for one layer.

    Deterministic inverse-CDF sampling with rejection on duplicates: the same
    (seed, layer, context, token) always routes identically, independent of
    which other tokens share the batch. When the previous decode step's
    selection is supplied, each slot re-selects its previous expert with the
    declared reuse probability (the temporal-locality anchor) and draws fresh
    otherwise.
    """

    def _state(salt: int) -> int:
        return _mix64(
            (seed & (2**64 - 1))
            ^ (layer * 0x9E3779B97F4A7C15)
            ^ (context_id * 0xC2B2AE3D27D4EB4F)
            ^ (token * 0x165667B19E3779F9)
            ^ salt
        )

    selected: list[int] = []
    attempt = 0
    for slot in range(activated):
        if (
            previous is not None
            and slot < len(previous)
            and reuse_probability > 0.0
        ):
            reuse_position = _state(0x52455553 + slot) / 2.0**64
            if (
                reuse_position < reuse_probability
                and previous[slot] not in selected
            ):
                selected.append(previous[slot])
                continue
        while True:
            position = _state(attempt) / 2.0**64
            attempt += 1
            low, high = 0, len(cumulative) - 1
            while low < high:
                middle = (low + high) // 2
                if cumulative[middle] < position:
                    low = middle + 1
                else:
                    high = middle
            expert = rank_to_expert[low]
            if expert not in selected:
                selected.append(expert)
                break
    return tuple(selected)


@dataclass(frozen=True)
class _Operation:
    op: str
    address: int
    byte_count: int
    region_id: str
    access_pattern: str
    write_scenario: str | None = None
    dependency_indices: tuple[int, ...] = ()
    detail: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FixedFootprintPhase:
    id: str
    stage: str
    layer: int | None
    object_class: str
    objects: tuple[str, ...]
    trace_group: CanonicalTraceBatch
    # Decode-step ordinal for multi-step windows; None in single-step windows
    # so their canonical form (and digest) is unchanged.
    step: int | None = None

    @cached_property
    def read_bytes(self) -> int:
        return sum(
            transaction.bytes
            for transaction in self.trace_group.memory_transactions
            if transaction.op == "R"
        )

    @cached_property
    def write_bytes(self) -> int:
        return sum(
            transaction.bytes
            for transaction in self.trace_group.memory_transactions
            if transaction.op == "W"
        )

    @cached_property
    def access_mix(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        by_id = {
            transaction.id: transaction
            for transaction in self.trace_group.memory_transactions
        }
        for identifier, label in self.trace_group.audit_labels.items():
            transaction = by_id.get(identifier)
            if transaction is None:
                continue
            pattern = str(label.get("access_pattern", "unspecified"))
            row = result.setdefault(
                pattern,
                {
                    "read_operations": 0,
                    "read_bytes": 0,
                    "write_operations": 0,
                    "write_bytes": 0,
                },
            )
            direction = "read" if transaction.op == "R" else "write"
            row[f"{direction}_operations"] += 1
            row[f"{direction}_bytes"] += transaction.bytes
        return result

    @cached_property
    def write_scenarios(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        by_id = {
            transaction.id: transaction
            for transaction in self.trace_group.memory_transactions
        }
        for identifier, label in self.trace_group.audit_labels.items():
            transaction = by_id.get(identifier)
            if transaction is None or transaction.op != "W":
                continue
            scenario = str(label.get("write_scenario", "unspecified"))
            row = result.setdefault(scenario, {"operations": 0, "bytes": 0})
            row["operations"] += 1
            row["bytes"] += transaction.bytes
        return result

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "layer": self.layer,
            **({"step": self.step} if self.step is not None else {}),
            "object_class": self.object_class,
            "objects": list(self.objects),
            "trace_group_sha256": self.trace_group.digest,
            "memory_transactions": len(self.trace_group.memory_transactions),
            "dependency_edges": sum(
                len(transaction.dependencies)
                for transaction in self.trace_group.transactions
            ),
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "access_mix": self.access_mix,
            "write_scenarios": self.write_scenarios,
        }



def _make_phase(
    *,
    layout: MemoryLayout,
    protocol_id: int,
    identifier: str,
    stage: str,
    layer: int | None,
    object_class: str,
    objects: Sequence[str],
    operations: Sequence[_Operation],
    contract_sha256: str,
    step: int | None = None,
) -> FixedFootprintPhase:
    if not operations:
        _fail(f"phase {identifier} has no operations")
    transactions: list[LogicalTransaction] = []
    routing: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}
    for index, operation in enumerate(operations):
        if operation.op not in {"R", "W"} or operation.byte_count <= 0:
            _fail(f"phase {identifier} operation {index} is malformed")
        if any(dependency >= index for dependency in operation.dependency_indices):
            _fail(f"phase {identifier} operation {index} has a forward dependency")
        region = layout.region(operation.region_id)
        if not (
            region.begin <= operation.address
            and operation.address + operation.byte_count <= region.end
        ):
            _fail(f"phase {identifier} operation {index} escapes its object")
        transaction_id = f"trace/{identifier}/m{index}"
        dependency_ids = tuple(
            f"trace/{identifier}/m{dependency}"
            for dependency in operation.dependency_indices
        )
        transactions.append(
            LogicalTransaction(
                id=transaction_id,
                op=operation.op,
                addr=operation.address,
                bytes=operation.byte_count,
                issue_ns=0.0,
                dependencies=dependency_ids,
            )
        )
        routing[transaction_id] = {
            "region_id": region.id,
            "group": layer if layer is not None else region.group,
        }
        label = {
            "stage": stage,
            "object_class": object_class,
            "object": region.id,
            "access_pattern": operation.access_pattern,
        }
        if operation.write_scenario is not None:
            label["write_scenario"] = operation.write_scenario
        if operation.detail is not None:
            label.update(dict(operation.detail))
        labels[transaction_id] = label
    trace_group = CanonicalTraceBatch(
        batch_id=protocol_id,
        transactions=tuple(transactions),
        layout=layout,
        routing=routing,
        audit_labels=labels,
        contract_sha256=contract_sha256,
    )
    return FixedFootprintPhase(
        id=identifier,
        stage=stage,
        layer=layer,
        object_class=object_class,
        objects=tuple(objects),
        trace_group=trace_group,
        step=step,
    )


def _context_partition(
    *, context_id: int, context_count: int, total_blocks: int
) -> tuple[int, int]:
    base_count, remainder = divmod(total_blocks, context_count)
    count = base_count + (1 if context_id < remainder else 0)
    begin = context_id * base_count + min(context_id, remainder)
    return begin, count

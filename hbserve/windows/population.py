#!/usr/bin/env python3
"""Construct a logical population independently of physical topology.

Every HBM/HBF organization projects the same population onto its own physical
capacities, rather than sizing the workload to each device independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hbserve.windows.memory_trace import (
    MemoryLayout,
    derive_model_capacity_inputs,
)


EXPLICIT_POPULATION_SCHEMA = {
    "name": "hbserve.fixed_population",
    "version": 1,
}
ACTIVE_WEIGHT_BUFFER_COUNT = 2


class ExplicitPopulationError(ValueError):
    """An explicit population target cannot hold the declared model state."""


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExplicitPopulationError(f"{description} must be an integer > 0")
    return value




def build_explicit_fixed_population(
    *,
    model_descriptor_path: Path,
    target_population_bytes: int,
    runtime_overhead_bytes: int,
) -> tuple[MemoryLayout, dict[str, Any]]:
    """Build the largest whole-KV-block population within an explicit target.

    ``target_population_bytes`` bounds the aligned canonical address space, not
    a ratio to HBM and not a topology-specific placement. Weight objects,
    block-table metadata, the layer-major KV slot arena, and runtime overhead
    are all charged. The returned layout can therefore be reused byte-for-byte
    by every organization in a matched experiment.
    """

    target = _positive_integer(
        target_population_bytes, "target_population_bytes"
    )
    runtime = _positive_integer(
        runtime_overhead_bytes, "runtime_overhead_bytes"
    )
    descriptor = model_descriptor_path.resolve()
    inputs = derive_model_capacity_inputs(descriptor)
    weights = _positive_integer(
        inputs.get("immutable_weight_backing_bytes"),
        "immutable_weight_backing_bytes",
    )
    layers = _positive_integer(inputs.get("num_layers"), "num_layers")
    block_tokens = _positive_integer(
        inputs.get("kv_block_size_tokens"), "kv_block_size_tokens"
    )
    page_per_layer = _positive_integer(
        inputs.get("kv_page_bytes_per_layer"),
        "kv_page_bytes_per_layer",
    )
    block_table_entry_bytes = _positive_integer(
        inputs.get("block_table_entry_bytes"),
        "block_table_entry_bytes",
    )
    active_buffer_per_slot = _positive_integer(
        inputs.get("active_weight_buffer_bytes_per_slot"),
        "active_weight_buffer_bytes_per_slot",
    )
    kv_stride = page_per_layer * layers
    bytes_per_block_with_metadata = kv_stride + block_table_entry_bytes
    fixed_payload = weights + runtime
    if fixed_payload + bytes_per_block_with_metadata > target:
        raise ExplicitPopulationError(
            "population target cannot hold weights, runtime overhead, and one "
            "complete KV block"
        )

    logical_blocks = (
        target - fixed_payload
    ) // bytes_per_block_with_metadata
    while logical_blocks > 0:
        block_table_bytes = logical_blocks * block_table_entry_bytes
        logical_kv_bytes = logical_blocks * kv_stride
        unique_payload_bytes = (
            weights + runtime + block_table_bytes + logical_kv_bytes
        )
        plan = {
            "num_layers": layers,
            "kv_block_size_tokens": block_tokens,
            "kv_page_bytes_per_layer": page_per_layer,
            "kv_block_stride_bytes": kv_stride,
            "kv_storage_order": "layer_major",
            "num_logical_kv_blocks": logical_blocks,
            "immutable_weight_backing_bytes": weights,
            "runtime_overhead_bytes": runtime,
            "block_table_entry_bytes": block_table_entry_bytes,
            "block_table_bytes": block_table_bytes,
            "canonical_block_table_region_bytes": block_table_bytes,
        }
        layout = MemoryLayout.build(descriptor, plan)
        if layout.address_space_bytes <= target:
            break
        logical_blocks -= 1
    else:
        raise ExplicitPopulationError(
            "aligned population target cannot hold one complete KV block"
        )

    alignment_padding = layout.address_space_bytes - unique_payload_bytes
    if alignment_padding < 0:
        raise ExplicitPopulationError(
            "aligned address space is smaller than its logical payload"
        )
    active_buffers = ACTIVE_WEIGHT_BUFFER_COUNT * active_buffer_per_slot
    receipt = {
        "schema": EXPLICIT_POPULATION_SCHEMA,
        "population_basis": "explicit_aligned_address_space_bytes",
        "target_population_bytes": target,
        "allocated_population_bytes": layout.address_space_bytes,
        "target_rounding_slack_bytes": target - layout.address_space_bytes,
        "logical_payload_bytes": unique_payload_bytes,
        "alignment_padding_bytes": alignment_padding,
        "model": str(inputs.get("model_name")),
        "model_family": str(inputs.get("model_family", "dense")),
        "precision_profile": str(inputs.get("precision_profile")),
        "model_descriptor": dict(inputs["model_descriptor"]),
        "immutable_weight_backing_bytes": weights,
        "runtime_overhead_bytes": runtime,
        "active_weight_buffer_count": ACTIVE_WEIGHT_BUFFER_COUNT,
        "active_weight_buffer_bytes_per_slot": active_buffer_per_slot,
        "active_weight_buffer_bytes": active_buffers,
        "block_table_entry_bytes": block_table_entry_bytes,
        "block_table_bytes": block_table_bytes,
        "kv_block_size_tokens": block_tokens,
        "kv_page_bytes_per_layer": page_per_layer,
        "num_layers": layers,
        "kv_block_stride_bytes": kv_stride,
        "kv_storage_order": "layer_major",
        "num_logical_kv_blocks": logical_blocks,
        "logical_kv_bytes": logical_kv_bytes,
        "layout_sha256": layout.digest,
        "topology_independent": True,
        "hbm_pressure_used": False,
    }
    if inputs.get("model_family") == "moe":
        receipt["moe"] = dict(inputs["moe"])
    if (
        receipt["logical_payload_bytes"]
        != weights + runtime + block_table_bytes + logical_kv_bytes
        or receipt["allocated_population_bytes"] > target
        or receipt["target_rounding_slack_bytes"] < 0
    ):
        raise ExplicitPopulationError(
            "explicit population accounting does not conserve"
        )
    return layout, receipt

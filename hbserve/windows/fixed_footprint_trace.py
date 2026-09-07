#!/usr/bin/env python3
"""Build the topology-independent shared serving memory window.

The public entry point is `build_fixed_footprint_trace(layout, population,
workload)`. The workload declaration is validated by
`window_contract.resolve_window_plan`; deterministic primitives and phase
assembly live in `window_emitters`; this module holds the trace dataclass,
its receipt, and the per-window-shape emission sequence (mixed
prefill/decode, decode-only multi-step, prefill growth). The measured
window models only traffic justified by the model descriptor: weight and
expert reads, embedding/output reads, attention KV reads, KV appends, and
block-table metadata. Uncalibrated activation/scratch traffic remains
zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping, Sequence

from hbserve.windows.memory_trace import (
    MemoryLayout,
    canonical_sha256,
)
from hbserve.windows.window_contract import (
    TRACE_CONTRACT_SCHEMA,
    FixedFootprintTraceError,
    _fail,
    _integer,
    resolve_window_plan,
)
from hbserve.windows.window_emitters import (
    FixedFootprintPhase,
    _Operation,
    _context_partition,
    _make_phase,
    _mix64,
    _permutation,
    _routed_expert_draw,
    _zipf_cumulative,
)

__all__ = [
    "TRACE_SCHEMA",
    "TRACE_CONTRACT_SCHEMA",
    "PAGE_SIZE_BYTES",
    "FixedFootprintTraceError",
    "FixedFootprintPhase",
    "FixedFootprintTrace",
    "build_fixed_footprint_trace",
]

TRACE_SCHEMA = {
    "name": "hbfsim.hbf_fixed_footprint_trace",
    "version": 2,
}
PAGE_SIZE_BYTES = 4096

@dataclass(frozen=True)
class FixedFootprintTrace:
    layout: MemoryLayout
    population: Mapping[str, Any]
    workload: Mapping[str, Any]
    inference_source: Mapping[str, Any]
    phases: tuple[FixedFootprintPhase, ...]
    prefill_contexts: int
    prefill_tokens_per_context: int
    decode_contexts: int
    decode_tokens_per_context: int
    random_decode_contexts: int
    idle_contexts: int
    random_read_chunk_blocks: int
    window_shape: str = "mixed_prefill_decode"
    moe_routing: Mapping[str, Any] | None = None
    declared_decode_prior_tokens: int | None = None
    decode_steps: int = 1
    prefill_growth: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.phases:
            _fail("fixed-footprint trace has no phases")
        ids = [phase.id for phase in self.phases]
        if len(ids) != len(set(ids)):
            _fail("fixed-footprint phase IDs are not unique")
        for index, phase in enumerate(self.phases):
            if phase.trace_group.batch_id != index:
                _fail("trace-group protocol IDs are not contiguous")
            if phase.trace_group.layout.digest != self.layout.digest:
                _fail("trace group changed the fixed memory layout")
        if self.read_bytes <= 0 or self.write_bytes <= 0:
            _fail("fixed-footprint window must contain reads and writes")
        random_bytes = sum(
            row["read_bytes"]
            for pattern, row in self.access_mix.items()
            if pattern in {"paged_random_extent", "indexed_random"}
        )
        sequential_bytes = sum(
            row["read_bytes"]
            for pattern, row in self.access_mix.items()
            if pattern == "sequential_stream"
        )
        if random_bytes <= 0 or sequential_bytes <= 0:
            _fail("fixed-footprint reads must mix sequential and random access")

    @cached_property
    def read_bytes(self) -> int:
        return sum(phase.read_bytes for phase in self.phases)

    @cached_property
    def write_bytes(self) -> int:
        return sum(phase.write_bytes for phase in self.phases)

    @cached_property
    def access_mix(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for phase in self.phases:
            for pattern, source in phase.access_mix.items():
                target = result.setdefault(
                    pattern,
                    {
                        "read_operations": 0,
                        "read_bytes": 0,
                        "write_operations": 0,
                        "write_bytes": 0,
                    },
                )
                for key, value in source.items():
                    target[key] += value
        return result

    @cached_property
    def write_scenarios(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for phase in self.phases:
            for scenario, source in phase.write_scenarios.items():
                target = result.setdefault(
                    scenario, {"operations": 0, "bytes": 0}
                )
                target["operations"] += source["operations"]
                target["bytes"] += source["bytes"]
        return result

    @cached_property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": TRACE_SCHEMA,
                "layout_sha256": self.layout.digest,
                "workload": dict(self.workload),
                "inference_source": dict(self.inference_source),
                "phases": [phase.canonical() for phase in self.phases],
            }
        )

    def _object_manifest(self) -> dict[str, Any]:
        allocated = {
            "model_weights": sum(
                region.bytes
                for region in self.layout.regions
                if region.placement_class == "immutable_weight"
            ),
            "kv_cache": sum(
                region.bytes
                for region in self.layout.regions
                if region.placement_class == "kv"
            ),
            "metadata": sum(
                region.bytes
                for region in self.layout.regions
                if region.placement_class == "metadata"
            ),
        }
        accessed: dict[str, dict[str, int]] = {
            name: {"read_bytes": 0, "write_bytes": 0}
            for name in allocated
        }
        objects: dict[str, set[str]] = {name: set() for name in allocated}
        for phase in self.phases:
            accessed[phase.object_class]["read_bytes"] += phase.read_bytes
            accessed[phase.object_class]["write_bytes"] += phase.write_bytes
            objects[phase.object_class].update(phase.objects)
        return {
            "allocated_population_bytes": self.layout.address_space_bytes,
            "allocated_bytes_by_object_class": allocated,
            "window_logical_access_bytes_by_object_class": accessed,
            "objects_by_class": {
                name: sorted(values) for name, values in objects.items()
            },
            "object_classes": list(allocated),
        }

    def receipt(self) -> dict[str, Any]:
        total = self.read_bytes + self.write_bytes
        random_read_bytes = sum(
            row["read_bytes"]
            for pattern, row in self.access_mix.items()
            if pattern in {"paged_random_extent", "indexed_random"}
        )
        sequential_read_bytes = self.access_mix.get(
            "sequential_stream", {}
        ).get("read_bytes", 0)
        prefill_tokens = self.prefill_contexts * self.prefill_tokens_per_context
        decode_tokens = (
            self.decode_contexts
            * self.decode_tokens_per_context
            * self.decode_steps
        )
        step_traffic: list[dict[str, int]] | None = None
        stationarity: dict[str, Any] | None = None
        growth_steps = (
            int(self.prefill_growth["growth_steps"])
            if self.prefill_growth is not None
            else 0
        )
        total_steps = growth_steps + self.decode_steps
        if self.decode_steps > 1 or growth_steps:
            per_step: dict[int, dict[str, int]] = {
                step: {"step": step, "read_bytes": 0, "write_bytes": 0}
                for step in range(total_steps)
            }
            for phase in self.phases:
                if phase.step is None:
                    continue
                per_step[phase.step]["read_bytes"] += phase.read_bytes
                per_step[phase.step]["write_bytes"] += phase.write_bytes
            step_traffic = [per_step[step] for step in range(total_steps)]
            if self.window_shape == "decode_only_step":
                reads = [entry["read_bytes"] for entry in step_traffic]
                mean_reads = sum(reads) / len(reads)
                stationarity = {
                    "steps": self.decode_steps,
                    "mean_step_read_bytes": mean_reads,
                    "max_relative_read_deviation_from_mean": max(
                        abs(value - mean_reads) / mean_reads
                        for value in reads
                    ),
                }
        return {
            "schema": TRACE_SCHEMA,
            "trace_sha256": self.digest,
            "layout_sha256": self.layout.digest,
            "measurement_window": {
                "start": (
                    "model_weights_installed_and_kv_arena_empty;_prefill_growth_window"
                    if self.prefill_growth is not None
                    else "model_weights_installed_and_long_context_decode_KV_valid;_"
                    "prefill_KV_is_created_inside_the_window"
                    if self.prefill_contexts
                    else "model_weights_installed_and_decode_KV_valid;_"
                    "decode_only_step_window"
                ),
                "window_shape": self.window_shape,
                "stages": (
                    ["prefill_growth", "decode", "final_drain"]
                    if self.prefill_growth is not None and self.decode_steps
                    else ["prefill_growth", "final_drain"]
                    if self.prefill_growth is not None
                    else ["prefill", "decode", "final_drain"]
                    if self.prefill_contexts
                    else ["decode", "final_drain"]
                ),
                **(
                    {"prefill_growth": dict(self.prefill_growth)}
                    if self.prefill_growth is not None
                    else {}
                ),
                "prefill_contexts": self.prefill_contexts,
                "prefill_tokens_per_context": self.prefill_tokens_per_context,
                "prefill_tokens": prefill_tokens,
                "decode_contexts": self.decode_contexts,
                "decode_tokens_per_context": self.decode_tokens_per_context,
                "decode_steps_per_window": self.decode_steps,
                "decode_tokens": decode_tokens,
                **(
                    {
                        "step_traffic": step_traffic,
                        "stationarity": stationarity,
                    }
                    if step_traffic is not None
                    else {}
                ),
                "random_decode_contexts": self.random_decode_contexts,
                "sequential_decode_contexts": (
                    self.decode_contexts - self.random_decode_contexts
                ),
                "idle_reserved_contexts": self.idle_contexts,
                "declared_decode_context_prior_tokens": (
                    self.declared_decode_prior_tokens
                ),
                "random_read_chunk_blocks": self.random_read_chunk_blocks,
                "random_read_chunk_bytes": (
                    self.random_read_chunk_blocks
                    * self.layout.kv_page_bytes_per_layer
                ),
            },
            "inference_source": dict(self.inference_source),
            "context_count": (
                self.prefill_contexts + self.decode_contexts + self.idle_contexts
            ),
            "layer_count": self.layout.num_layers,
            "phase_count": len(self.phases),
            "traffic": {
                "bytes": total,
                "reads": {
                    "bytes": self.read_bytes,
                    "fraction": self.read_bytes / total,
                },
                "writes": {
                    "bytes": self.write_bytes,
                    "fraction": self.write_bytes / total,
                },
            },
            "read_access_mix": {
                "sequential_bytes": sequential_read_bytes,
                "sequential_fraction": sequential_read_bytes / self.read_bytes,
                "random_bytes": random_read_bytes,
                "random_fraction": random_read_bytes / self.read_bytes,
                "detail": self.access_mix,
            },
            "write_scenarios": self.write_scenarios,
            "phases": [phase.canonical() for phase in self.phases],
            "object_manifest": self._object_manifest(),
            "moe_routing": (
                dict(self.moe_routing) if self.moe_routing is not None else None
            ),
            "invariants": {
                "same_trace_for_every_topology": True,
                "topology_independent_population": True,
                "payload_bytes_materialized": False,
                "address_ranges_are_execution_input": True,
                "kv_storage_order": self.layout.kv_storage_order,
                "model_weights_preloaded_before_timing": True,
                "long_context_decode_KV_preloaded_before_timing": True,
                "prefill_KV_generated_inside_timing_window": (
                    self.prefill_contexts > 0
                    or self.prefill_growth is not None
                ),
                "window_starts_from_model_loaded_empty_kv": (
                    self.prefill_growth is not None
                ),
                "cold_model_install_included": False,
                "prior_decode_context_construction_included": False,
                "compute_time_modeled": False,
                "uncalibrated_activation_scratch_traffic_invented": False,
                "sequential_weights_preserved": True,
                "paged_KV_randomness_modeled": True,
                "random_kv_extents_issue_concurrently": True,
                "inference_KV_writes_modeled": True,
                "batch_dependent_expert_union_modeled": (
                    self.moe_routing is not None
                ),
            },
        }



def build_fixed_footprint_trace(
    *,
    layout: MemoryLayout,
    population: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> FixedFootprintTrace:
    """Compile one serving memory window over the fixed layout."""

    _plan = resolve_window_plan(
        layout=layout, population=population, workload=workload
    )
    window_shape = _plan["window_shape"]
    mixed_window = _plan["mixed_window"]
    growth_window = _plan["growth_window"]
    context_count = _plan["context_count"]
    prefill_contexts = _plan["prefill_contexts"]
    prefill_tokens_per_context = _plan["prefill_tokens_per_context"]
    decode_contexts = _plan["decode_contexts"]
    decode_tokens_per_context = _plan["decode_tokens_per_context"]
    random_decode_contexts = _plan["random_decode_contexts"]
    chunk_blocks = _plan["chunk_blocks"]
    declared_prior_tokens = _plan["declared_prior_tokens"]
    decode_steps = _plan["decode_steps"]
    growth_contexts = _plan["growth_contexts"]
    growth_target_tokens = _plan["growth_target_tokens"]
    growth_chunk_tokens = _plan["growth_chunk_tokens"]
    growth_chunks = _plan["growth_chunks"]
    growth_schedule = _plan["growth_schedule"]
    growth_decode_while_filling = _plan["growth_decode_while_filling"]
    locality_seed = _plan["locality_seed"]
    model_family = _plan["model_family"]
    moe_plan = _plan["moe_plan"]
    zipf_exponent = _plan["zipf_exponent"]
    routing_seed = _plan["routing_seed"]
    routing_distribution = _plan["routing_distribution"]
    routing_rank_weights = _plan["routing_rank_weights"]
    step_expert_reuse_probability = _plan["step_expert_reuse_probability"]
    inference_source = _plan["inference_source"]
    contract = _plan["contract"]
    contract_sha256 = _plan["contract_sha256"]
    active_contexts = _plan["active_contexts"]
    # Total prefill-growth iterations: the concurrent schedule runs one
    # iteration per chunk, the sequential schedule one per (context, chunk).
    growth_steps_total = growth_chunks * (
        growth_contexts if growth_schedule == "sequential" else 1
    )
    moe_routing_receipt: dict[str, Any] | None = None

    phases: list[FixedFootprintPhase] = []

    # Batch-dependent MoE expert routing: every scheduled token of a stage
    # draws its distinct routed experts per layer, and the stage reads the
    # union once. The union size is what couples serving batch to expert
    # weight traffic.
    decode_first_context = prefill_contexts
    stage_tokens: dict[str, tuple[tuple[int, int], ...]] = {
        "prefill": tuple(
            (context_id, token)
            for context_id in range(prefill_contexts)
            for token in range(prefill_tokens_per_context)
        ),
        "decode": tuple(
            (context_id, 0)
            for context_id in range(
                decode_first_context,
                decode_first_context + decode_contexts,
            )
        ),
    }
    moe_routed_experts = 0
    moe_activated = 0
    moe_first_layer = 0
    moe_expert_stride = 0
    moe_dense_sublayer_bytes: list[int] = []
    moe_cumulative: tuple[float, ...] = ()
    moe_rank_to_expert: dict[int, tuple[int, ...]] = {}
    moe_union_sizes: dict[str, list[int]] = {"prefill": [], "decode": []}
    moe_expert_read_bytes: dict[str, int] = {"prefill": 0, "decode": 0}
    # Per (layer, context) previous-step expert selection for the temporal
    # reuse anchor, and per-layer union of the previous decode step for the
    # cross-step overlap statistic.
    moe_previous_selection: dict[tuple[int, int], tuple[int, ...]] = {}
    moe_previous_layer_union: dict[int, frozenset[int]] = {}
    moe_cross_step_jaccard: list[float] = []
    if moe_plan is not None:
        moe_routed_experts = _integer(
            moe_plan.get("routed_experts_per_layer"),
            "MoE routed experts per layer",
            minimum=2,
        )
        moe_activated = _integer(
            moe_plan.get("activated_routed_experts_per_token"),
            "MoE activated routed experts",
            minimum=1,
        )
        moe_first_layer = _integer(
            moe_plan.get("first_moe_layer"), "MoE first MoE layer"
        )
        moe_expert_stride = _integer(
            moe_plan.get("expert_stride_bytes"),
            "MoE expert stride bytes",
            minimum=1,
        )
        moe_dense_sublayer_bytes = [
            _integer(value, "MoE dense sublayer bytes", minimum=1)
            for value in moe_plan.get("dense_sublayer_bytes_by_layer", ())
        ]
        if len(moe_dense_sublayer_bytes) != layout.num_layers:
            _fail("MoE dense sublayer list does not match the layer count")
        if routing_rank_weights is not None:
            if len(routing_rank_weights) != moe_routed_experts:
                _fail(
                    "MoE routing rank weights must cover every routed expert"
                )
            total_weight = sum(routing_rank_weights)
            cumulative_weights: list[float] = []
            running = 0.0
            for weight in routing_rank_weights:
                running += weight / total_weight
                cumulative_weights.append(running)
            cumulative_weights[-1] = 1.0
            moe_cumulative = tuple(cumulative_weights)
        else:
            moe_cumulative = _zipf_cumulative(
                moe_routed_experts, zipf_exponent
            )
        for layer in range(moe_first_layer, layout.num_layers):
            moe_rank_to_expert[layer] = _permutation(
                moe_routed_experts,
                routing_seed ^ (layer * 0x9E3779B97F4A7C15),
            )

    def append_phase(
        *,
        identifier: str,
        stage: str,
        layer: int | None,
        object_class: str,
        objects: Sequence[str],
        operations: Sequence[_Operation],
        step: int | None = None,
    ) -> None:
        phases.append(
            _make_phase(
                layout=layout,
                protocol_id=len(phases),
                identifier=identifier,
                stage=stage,
                layer=layer,
                object_class=object_class,
                objects=objects,
                operations=operations,
                contract_sha256=contract_sha256,
                step=step,
            )
        )

    block_table = layout.region("metadata/block_table")
    runtime = layout.region(layout.metadata_region_id)
    metadata_operations: list[_Operation] = []
    runtime_page_count = runtime.bytes // PAGE_SIZE_BYTES
    runtime_page = _mix64(locality_seed ^ 0x4D455441) % runtime_page_count
    metadata_operations.append(
        _Operation(
            op="R",
            address=runtime.begin + runtime_page * PAGE_SIZE_BYTES,
            byte_count=PAGE_SIZE_BYTES,
            region_id=runtime.id,
            access_pattern="indexed_random",
            detail={"role": "scheduler_runtime_state"},
        )
    )
    for context_id in range(
        decode_first_context, decode_first_context + decode_contexts
    ):
        block_begin, block_count = _context_partition(
            context_id=context_id,
            context_count=context_count,
            total_blocks=layout.num_logical_kv_blocks,
        )
        metadata_operations.append(
            _Operation(
                op="R",
                address=block_table.begin + block_begin * 4,
                byte_count=block_count * 4,
                region_id=block_table.id,
                access_pattern="indexed_random",
                detail={
                    "role": "decode_block_table_lookup",
                    "context_id": context_id,
                },
            )
        )
    append_phase(
        identifier="decode_context_metadata_lookup",
        stage="window_setup",
        layer=None,
        object_class="metadata",
        objects=(runtime.id, block_table.id),
        operations=metadata_operations,
    )

    if prefill_contexts:
        allocation_operations: list[_Operation] = []
        for context_id in range(prefill_contexts):
            block_begin, _ = _context_partition(
                context_id=context_id,
                context_count=context_count,
                total_blocks=layout.num_logical_kv_blocks,
            )
            allocation_operations.append(
                _Operation(
                    op="W",
                    address=block_table.begin + block_begin * 4,
                    byte_count=4,
                    region_id=block_table.id,
                    access_pattern="indexed_update",
                    write_scenario="prefill_block_table_allocation",
                    detail={"context_id": context_id},
                )
            )
        append_phase(
            identifier="prefill_block_table_allocation",
            stage="prefill",
            layer=None,
            object_class="metadata",
            objects=(block_table.id,),
            operations=allocation_operations,
        )

    embedding = layout.region("weights/embedding")
    if embedding.bytes % layout.vocab_size:
        _fail("embedding object does not contain integral vocabulary rows")
    embedding_row_bytes = embedding.bytes // layout.vocab_size

    def embedding_operations(
        *, first_context: int, contexts: int, tokens: int, salt: int
    ) -> list[_Operation]:
        result: list[_Operation] = []
        for context_id in range(first_context, first_context + contexts):
            for token in range(tokens):
                row = _mix64(
                    locality_seed ^ salt ^ (context_id << 16) ^ token
                ) % layout.vocab_size
                result.append(
                    _Operation(
                        op="R",
                        address=embedding.begin + row * embedding_row_bytes,
                        byte_count=embedding_row_bytes,
                        region_id=embedding.id,
                        access_pattern="indexed_random",
                        detail={
                            "context_id": context_id,
                            "scheduled_token": token,
                            "embedding_row": row,
                        },
                    )
                )
        return result

    if prefill_contexts:
        append_phase(
            identifier="prefill_embedding",
            stage="prefill",
            layer=None,
            object_class="model_weights",
            objects=(embedding.id,),
            operations=embedding_operations(
                first_context=0,
                contexts=prefill_contexts,
                tokens=prefill_tokens_per_context,
                salt=0x50524546,
            ),
        )

    kv_region = layout.region(layout.kv_region_id)

    def weight_phase(
        stage: str,
        layer: int,
        tokens: Sequence[tuple[int, int]],
        *,
        step: int | None = None,
        identifier_suffix: str = "",
        track_reuse: bool = False,
        full_sweep_tokens: int | None = None,
    ) -> None:
        region = layout.region(f"weights/layer/{layer}")
        if moe_plan is None or layer < moe_first_layer:
            operations: list[_Operation] = [
                _Operation(
                    op="R",
                    address=region.begin,
                    byte_count=region.bytes,
                    region_id=region.id,
                    access_pattern="sequential_stream",
                    detail={"role": "layer_weight_stream"},
                )
            ]
        else:
            # Dense sublayer (attention, norms, router, shared expert) is
            # read once per batch; routed experts are read once per distinct
            # expert the batch activated in this layer.
            operations = [
                _Operation(
                    op="R",
                    address=region.begin,
                    byte_count=moe_dense_sublayer_bytes[layer],
                    region_id=region.id,
                    access_pattern="sequential_stream",
                    detail={"role": "moe_dense_sublayer_stream"},
                )
            ]
            union: dict[int, int] = {}
            if full_sweep_tokens is not None:
                # At prefill-iteration batch sizes the probability that any
                # routed expert goes unactivated is negligible, so the sweep
                # is exact rather than sampled: expected unactivated experts
                # = N(1-k/N)^tokens.
                expected_missing = moe_routed_experts * (
                    (1.0 - moe_activated / moe_routed_experts)
                    ** full_sweep_tokens
                )
                if expected_missing > 1e-6:
                    _fail(
                        "prefill chunk batch is too small for the exact "
                        "full-expert-sweep form; draw per token instead"
                    )
                expected_activations = max(
                    1,
                    round(
                        full_sweep_tokens
                        * moe_activated
                        / moe_routed_experts
                    ),
                )
                for expert in range(moe_routed_experts):
                    union[expert] = expected_activations
            for context_id, token in tokens:
                previous = (
                    moe_previous_selection.get((layer, context_id))
                    if track_reuse
                    else None
                )
                selection = _routed_expert_draw(
                    cumulative=moe_cumulative,
                    rank_to_expert=moe_rank_to_expert[layer],
                    activated=moe_activated,
                    seed=routing_seed,
                    layer=layer,
                    context_id=context_id,
                    token=token,
                    previous=previous,
                    reuse_probability=step_expert_reuse_probability,
                )
                if track_reuse:
                    moe_previous_selection[(layer, context_id)] = selection
                for expert in selection:
                    union[expert] = union.get(expert, 0) + 1
            for expert in sorted(union):
                operations.append(
                    _Operation(
                        op="R",
                        address=(
                            region.begin
                            + moe_dense_sublayer_bytes[layer]
                            + expert * moe_expert_stride
                        ),
                        byte_count=moe_expert_stride,
                        region_id=region.id,
                        access_pattern="paged_random_extent",
                        detail={
                            "role": "routed_expert_stream",
                            "expert_id": expert,
                            "batch_activations": union[expert],
                        },
                    )
                )
            moe_union_sizes[stage].append(len(union))
            moe_expert_read_bytes[stage] += len(union) * moe_expert_stride
            if track_reuse:
                current_union = frozenset(union)
                previous_union = moe_previous_layer_union.get(layer)
                if previous_union is not None:
                    merged = len(previous_union | current_union)
                    if merged:
                        moe_cross_step_jaccard.append(
                            len(previous_union & current_union) / merged
                        )
                moe_previous_layer_union[layer] = current_union
        append_phase(
            identifier=f"{stage}_layer_{layer:03d}_weights{identifier_suffix}",
            stage=stage,
            layer=layer,
            object_class="model_weights",
            objects=(region.id,),
            operations=operations,
            step=step,
        )

    if prefill_contexts:
        for layer in range(layout.num_layers):
            weight_phase("prefill", layer, stage_tokens["prefill"])
            writes: list[_Operation] = []
            for context_id in range(prefill_contexts):
                block_begin, _ = _context_partition(
                    context_id=context_id,
                    context_count=context_count,
                    total_blocks=layout.num_logical_kv_blocks,
                )
                writes.append(
                    _Operation(
                        op="W",
                        address=layout.kv_address(
                            block_id=block_begin, layer=layer, token_offset=0
                        ),
                        byte_count=(
                            prefill_tokens_per_context
                            * layout.bytes_per_token_per_layer
                        ),
                        region_id=kv_region.id,
                        access_pattern="sequential_append",
                        write_scenario="prefill_KV_append",
                        detail={"context_id": context_id, "layer": layer},
                    )
                )
            append_phase(
                identifier=f"prefill_layer_{layer:03d}_kv_append",
                stage="prefill",
                layer=layer,
                object_class="kv_cache",
                objects=(kv_region.id,),
                operations=writes,
            )

    final_norm = layout.region("weights/final_norm")
    output_head = layout.region("weights/output_head")
    if prefill_contexts:
        for identifier, region in (
            ("prefill_final_norm", final_norm),
            ("prefill_output_head", output_head),
        ):
            append_phase(
                identifier=identifier,
                stage="prefill",
                layer=None,
                object_class="model_weights",
                objects=(region.id,),
                operations=(
                    _Operation(
                        op="R",
                        address=region.begin,
                        byte_count=region.bytes,
                        region_id=region.id,
                        access_pattern="sequential_stream",
                    ),
                ),
            )

    if growth_window:
        # Chunked prefill from the model-loaded state: chunk c processes
        # tokens [c*S, (c+1)*S) of every growth context - allocates the
        # chunk's KV blocks, reads the weights once per iteration (an exact
        # full expert sweep at prefill batch sizes), reads the history the
        # earlier chunks grew, and writes the chunk's own KV per layer.
        growth_sample_ops = min(growth_chunk_tokens, 256)
        if growth_chunk_tokens % growth_sample_ops:
            _fail(
                "prefill chunk tokens must divide into the aggregated "
                "embedding extents"
            )
        growth_rows_per_op = growth_chunk_tokens // growth_sample_ops
        growth_blocks_per_chunk = (
            growth_chunk_tokens // layout.block_size_tokens
        )
        for context_id in range(growth_contexts):
            _, block_count = _context_partition(
                context_id=context_id,
                context_count=context_count,
                total_blocks=layout.num_logical_kv_blocks,
            )
            if (
                block_count * layout.block_size_tokens
                < growth_target_tokens + decode_steps
            ):
                _fail(
                    "growth context partition cannot hold the target length "
                    "and its decode handoff"
                )
        # One growth iteration per (step, active-context set, chunk index):
        # the concurrent schedule batches every context into each chunk
        # (they share the weight stream); the sequential schedule grows one
        # context to its target before the next arrives, so earlier
        # contexts sit idle-resident with genuinely cold KV while each
        # arrival pays its own weight sweeps.
        if growth_schedule == "sequential":
            growth_iterations = [
                (
                    ctx * growth_chunks + chunk,
                    (ctx,),
                    chunk,
                    f"prefill_ctx{ctx:02d}_chunk_{chunk:03d}",
                    f"_ctx{ctx:02d}_chunk{chunk:03d}",
                )
                for ctx in range(growth_contexts)
                for chunk in range(growth_chunks)
            ]
        else:
            growth_iterations = [
                (
                    chunk,
                    tuple(range(growth_contexts)),
                    chunk,
                    f"prefill_chunk_{chunk:03d}",
                    f"_chunk{chunk:03d}",
                )
                for chunk in range(growth_chunks)
            ]
        for (
            growth_step,
            iteration_contexts,
            growth_chunk,
            iteration_tag,
            weight_suffix,
        ) in growth_iterations:
            allocation_ops: list[_Operation] = []
            for context_id in iteration_contexts:
                block_begin, _ = _context_partition(
                    context_id=context_id,
                    context_count=context_count,
                    total_blocks=layout.num_logical_kv_blocks,
                )
                first_block = (
                    block_begin + growth_chunk * growth_blocks_per_chunk
                )
                allocation_ops.append(
                    _Operation(
                        op="W",
                        address=block_table.begin + first_block * 4,
                        byte_count=growth_blocks_per_chunk * 4,
                        region_id=block_table.id,
                        access_pattern="indexed_update",
                        write_scenario="prefill_block_table_allocation",
                        detail={
                            "context_id": context_id,
                            "chunk": growth_chunk,
                        },
                    )
                )
            append_phase(
                identifier=f"{iteration_tag}_block_allocation",
                stage="prefill",
                layer=None,
                object_class="metadata",
                objects=(block_table.id,),
                operations=allocation_ops,
                step=growth_step,
            )
            embedding_ops: list[_Operation] = []
            for context_id in iteration_contexts:
                for op_index in range(growth_sample_ops):
                    row = _mix64(
                        locality_seed
                        ^ 0x47524F57
                        ^ (context_id << 24)
                        ^ (growth_chunk << 12)
                        ^ op_index
                    ) % (layout.vocab_size - growth_rows_per_op + 1)
                    embedding_ops.append(
                        _Operation(
                            op="R",
                            address=(
                                embedding.begin + row * embedding_row_bytes
                            ),
                            byte_count=(
                                growth_rows_per_op * embedding_row_bytes
                            ),
                            region_id=embedding.id,
                            access_pattern="indexed_random",
                            detail={
                                "role": "chunk_embedding_rows_aggregated",
                                "rows_per_extent": growth_rows_per_op,
                            },
                        )
                    )
            append_phase(
                identifier=f"{iteration_tag}_embedding",
                stage="prefill",
                layer=None,
                object_class="model_weights",
                objects=(embedding.id,),
                operations=embedding_ops,
                step=growth_step,
            )
            filled_contexts = (
                range(iteration_contexts[0])
                if growth_decode_while_filling
                and len(iteration_contexts) == 1
                else range(0)
            )
            if len(filled_contexts):
                dwf_embedding_ops = [
                    _Operation(
                        op="R",
                        address=(
                            embedding.begin
                            + (
                                _mix64(
                                    locality_seed
                                    ^ 0x44574643
                                    ^ (filled_id << 24)
                                    ^ growth_step
                                )
                                % layout.vocab_size
                            )
                            * embedding_row_bytes
                        ),
                        byte_count=embedding_row_bytes,
                        region_id=embedding.id,
                        access_pattern="indexed_random",
                        detail={
                            "role": "decode_while_filling_token_row",
                            "context_id": filled_id,
                        },
                    )
                    for filled_id in filled_contexts
                ]
                append_phase(
                    identifier=f"{iteration_tag}_dwf_embedding",
                    stage="decode",
                    layer=None,
                    object_class="model_weights",
                    objects=(embedding.id,),
                    operations=dwf_embedding_ops,
                    step=growth_step,
                )
            for layer in range(layout.num_layers):
                weight_phase(
                    "prefill",
                    layer,
                    (),
                    step=growth_step,
                    identifier_suffix=weight_suffix,
                    full_sweep_tokens=(
                        growth_chunk_tokens * len(iteration_contexts)
                        if moe_plan is not None
                        else None
                    ),
                )
                growth_kv_ops: list[_Operation] = []
                for context_id in iteration_contexts:
                    block_begin, _ = _context_partition(
                        context_id=context_id,
                        context_count=context_count,
                        total_blocks=layout.num_logical_kv_blocks,
                    )
                    prior_tokens = growth_chunk * growth_chunk_tokens
                    write_dependencies: tuple[int, ...] = ()
                    if prior_tokens:
                        growth_kv_ops.append(
                            _Operation(
                                op="R",
                                address=layout.kv_address(
                                    block_id=block_begin,
                                    layer=layer,
                                    token_offset=0,
                                ),
                                byte_count=(
                                    prior_tokens
                                    * layout.bytes_per_token_per_layer
                                ),
                                region_id=kv_region.id,
                                access_pattern="sequential_stream",
                                detail={
                                    "context_id": context_id,
                                    "layer": layer,
                                    "prior_tokens": prior_tokens,
                                },
                            )
                        )
                        write_dependencies = (len(growth_kv_ops) - 1,)
                    growth_kv_ops.append(
                        _Operation(
                            op="W",
                            address=layout.kv_address(
                                block_id=(
                                    block_begin
                                    + growth_chunk * growth_blocks_per_chunk
                                ),
                                layer=layer,
                                token_offset=0,
                            ),
                            byte_count=(
                                growth_chunk_tokens
                                * layout.bytes_per_token_per_layer
                            ),
                            region_id=kv_region.id,
                            access_pattern="sequential_append",
                            write_scenario="prefill_KV_append",
                            dependency_indices=write_dependencies,
                            detail={
                                "context_id": context_id,
                                "layer": layer,
                                "chunk": growth_chunk,
                            },
                        )
                    )
                append_phase(
                    identifier=f"{iteration_tag}_layer_{layer:03d}_kv",
                    stage="prefill",
                    layer=layer,
                    object_class="kv_cache",
                    objects=(kv_region.id,),
                    operations=growth_kv_ops,
                    step=growth_step,
                )
                if len(filled_contexts):
                    dwf_kv_ops: list[_Operation] = []
                    for filled_id in filled_contexts:
                        filled_begin, filled_count = _context_partition(
                            context_id=filled_id,
                            context_count=context_count,
                            total_blocks=layout.num_logical_kv_blocks,
                        )
                        filled_prior = growth_target_tokens + (
                            (iteration_contexts[0] - filled_id - 1)
                            * growth_chunks
                            + growth_chunk
                        )
                        if filled_prior + 1 > (
                            filled_count * layout.block_size_tokens
                        ):
                            _fail(
                                "decode-while-filling appends exceed the "
                                "context's block partition"
                            )
                        dwf_kv_ops.append(
                            _Operation(
                                op="R",
                                address=layout.kv_address(
                                    block_id=filled_begin,
                                    layer=layer,
                                    token_offset=0,
                                ),
                                byte_count=(
                                    filled_prior
                                    * layout.bytes_per_token_per_layer
                                ),
                                region_id=kv_region.id,
                                access_pattern="sequential_stream",
                                detail={
                                    "context_id": filled_id,
                                    "layer": layer,
                                    "prior_tokens": filled_prior,
                                },
                            )
                        )
                        dwf_kv_ops.append(
                            _Operation(
                                op="W",
                                address=layout.kv_address(
                                    block_id=(
                                        filled_begin
                                        + filled_prior
                                        // layout.block_size_tokens
                                    ),
                                    layer=layer,
                                    token_offset=(
                                        filled_prior
                                        % layout.block_size_tokens
                                    ),
                                ),
                                byte_count=layout.bytes_per_token_per_layer,
                                region_id=kv_region.id,
                                access_pattern="sequential_append",
                                write_scenario="decode_KV_append",
                                dependency_indices=(len(dwf_kv_ops) - 1,),
                                detail={
                                    "context_id": filled_id,
                                    "layer": layer,
                                },
                            )
                        )
                    append_phase(
                        identifier=(
                            f"{iteration_tag}_layer_{layer:03d}_dwf_kv"
                        ),
                        stage="decode",
                        layer=layer,
                        object_class="kv_cache",
                        objects=(kv_region.id,),
                        operations=dwf_kv_ops,
                        step=growth_step,
                    )
        for identifier, region in (
            ("prefill_growth_final_norm", final_norm),
            ("prefill_growth_output_head", output_head),
        ):
            append_phase(
                identifier=identifier,
                stage="prefill",
                layer=None,
                object_class="model_weights",
                objects=(region.id,),
                operations=(
                    _Operation(
                        op="R",
                        address=region.begin,
                        byte_count=region.bytes,
                        region_id=region.id,
                        access_pattern="sequential_stream",
                    ),
                ),
                step=growth_steps_total - 1,
            )

    # Tokens each context already appended while later arrivals were
    # filling (empty unless decode-while-filling is enabled); the
    # decode tail continues from those grown histories.
    dwf_tokens_by_context: dict[int, int] = (
        {
            context_id: (growth_contexts - 1 - context_id)
            * growth_chunks
            for context_id in range(growth_contexts)
        }
        if growth_window and growth_decode_while_filling
        else {}
    )
    for decode_step in range(decode_steps):
        multi_step = decode_steps > 1 or growth_window
        step_suffix = f"_step{decode_step:02d}" if multi_step else ""
        step_field = (
            growth_steps_total + decode_step if growth_window
            else decode_step if multi_step
            else None
        )
        step_tokens = tuple(
            (context_id, decode_step)
            for context_id in range(
                decode_first_context,
                decode_first_context + decode_contexts,
            )
        )
        append_phase(
            identifier=f"decode_embedding{step_suffix}",
            stage="decode",
            layer=None,
            object_class="model_weights",
            objects=(embedding.id,),
            operations=embedding_operations(
                first_context=decode_first_context,
                contexts=decode_contexts,
                tokens=decode_tokens_per_context,
                salt=0x4445434F ^ (decode_step * 0x9E3779B1),
            ),
            step=step_field,
        )

        for layer in range(layout.num_layers):
            weight_phase(
                "decode",
                layer,
                step_tokens,
                step=step_field,
                identifier_suffix=step_suffix,
                track_reuse=True,
            )
            kv_operations: list[_Operation] = []
            for decode_ordinal, context_id in enumerate(
                range(
                    decode_first_context,
                    decode_first_context + decode_contexts,
                )
            ):
                block_begin, block_count = _context_partition(
                    context_id=context_id,
                    context_count=context_count,
                    total_blocks=layout.num_logical_kv_blocks,
                )
                capacity_tokens = block_count * layout.block_size_tokens
                prior_tokens = (
                    declared_prior_tokens
                    if declared_prior_tokens is not None
                    else capacity_tokens - 1
                ) + decode_step + dwf_tokens_by_context.get(context_id, 0)
                if prior_tokens <= 0:
                    _fail(
                        "decode context partition has no prior token capacity"
                    )
                if prior_tokens >= capacity_tokens:
                    _fail(
                        "declared decode context length leaves no room for "
                        "every appended step token inside the context's "
                        "block partition"
                    )
                dependency_indices: tuple[int, ...]
                if decode_ordinal < random_decode_contexts:
                    extent_tokens_full = (
                        chunk_blocks * layout.block_size_tokens
                    )
                    extent_count = (
                        prior_tokens + extent_tokens_full - 1
                    ) // extent_tokens_full
                    order = _permutation(
                        extent_count,
                        locality_seed
                        ^ (context_id * 0x9E3779B1)
                        ^ (decode_step * 0xC2B2AE3D),
                    )
                    # Paged attention issues every history extent of a
                    # context concurrently; only the append waits for all of
                    # them, so fragmentation is measured, not serialization.
                    extent_indices: list[int] = []
                    for extent in order:
                        extent_first_token = extent * extent_tokens_full
                        extent_tokens = min(
                            extent_tokens_full,
                            prior_tokens - extent_first_token,
                        )
                        byte_count = (
                            extent_tokens * layout.bytes_per_token_per_layer
                        )
                        extent_blocks = (
                            extent_tokens + layout.block_size_tokens - 1
                        ) // layout.block_size_tokens
                        kv_operations.append(
                            _Operation(
                                op="R",
                                address=layout.kv_address(
                                    block_id=(
                                        block_begin + extent * chunk_blocks
                                    ),
                                    layer=layer,
                                    token_offset=0,
                                ),
                                byte_count=byte_count,
                                region_id=kv_region.id,
                                access_pattern="paged_random_extent",
                                detail={
                                    "context_id": context_id,
                                    "layer": layer,
                                    "extent_ordinal": extent,
                                    "extent_blocks": extent_blocks,
                                },
                            )
                        )
                        extent_indices.append(len(kv_operations) - 1)
                    dependency_indices = tuple(extent_indices)
                else:
                    kv_operations.append(
                        _Operation(
                            op="R",
                            address=layout.kv_address(
                                block_id=block_begin,
                                layer=layer,
                                token_offset=0,
                            ),
                            byte_count=(
                                prior_tokens
                                * layout.bytes_per_token_per_layer
                            ),
                            region_id=kv_region.id,
                            access_pattern="sequential_stream",
                            detail={
                                "context_id": context_id,
                                "layer": layer,
                                "prior_tokens": prior_tokens,
                            },
                        )
                    )
                    dependency_indices = (len(kv_operations) - 1,)
                kv_operations.append(
                    _Operation(
                        op="W",
                        address=layout.kv_address(
                            block_id=(
                                block_begin
                                + prior_tokens // layout.block_size_tokens
                            ),
                            layer=layer,
                            token_offset=(
                                prior_tokens % layout.block_size_tokens
                            ),
                        ),
                        byte_count=layout.bytes_per_token_per_layer,
                        region_id=kv_region.id,
                        access_pattern="sequential_append",
                        write_scenario="decode_KV_append",
                        dependency_indices=dependency_indices,
                        detail={"context_id": context_id, "layer": layer},
                    )
                )
            append_phase(
                identifier=f"decode_layer_{layer:03d}_kv_access{step_suffix}",
                stage="decode",
                layer=layer,
                object_class="kv_cache",
                objects=(kv_region.id,),
                operations=kv_operations,
                step=step_field,
            )

        for identifier, region in (
            (f"decode_final_norm{step_suffix}", final_norm),
            (f"decode_output_head{step_suffix}", output_head),
        ):
            append_phase(
                identifier=identifier,
                stage="decode",
                layer=None,
                object_class="model_weights",
                objects=(region.id,),
                operations=(
                    _Operation(
                        op="R",
                        address=region.begin,
                        byte_count=region.bytes,
                        region_id=region.id,
                        access_pattern="sequential_stream",
                    ),
                ),
                step=step_field,
            )

    if moe_plan is not None:
        def _stage_summary(stage: str) -> dict[str, Any]:
            sizes = moe_union_sizes[stage]
            return {
                "layer_phases": len(sizes),
                "expert_union_min": min(sizes) if sizes else 0,
                "expert_union_mean": (
                    sum(sizes) / len(sizes) if sizes else 0.0
                ),
                "expert_union_max": max(sizes) if sizes else 0,
                "routed_expert_read_bytes": moe_expert_read_bytes[stage],
            }

        moe_routing_receipt = {
            "distribution": routing_distribution,
            "zipf_exponent": (
                zipf_exponent if routing_rank_weights is None else None
            ),
            "empirical_rank_weights": (
                list(routing_rank_weights)
                if routing_rank_weights is not None
                else None
            ),
            "step_expert_reuse_probability": step_expert_reuse_probability,
            "seed": routing_seed,
            "routed_experts_per_layer": moe_routed_experts,
            "activated_routed_experts_per_token": moe_activated,
            "expert_stride_bytes": moe_expert_stride,
            "expert_reads_cover_alignment_padding": True,
            "per_layer_rank_rotation": True,
            "stages": {
                "prefill": _stage_summary("prefill"),
                "decode": _stage_summary("decode"),
            },
        }
        if moe_cross_step_jaccard:
            moe_routing_receipt["cross_step_union_jaccard"] = {
                "samples": len(moe_cross_step_jaccard),
                "mean": (
                    sum(moe_cross_step_jaccard)
                    / len(moe_cross_step_jaccard)
                ),
                "min": min(moe_cross_step_jaccard),
                "max": max(moe_cross_step_jaccard),
            }

    return FixedFootprintTrace(
        layout=layout,
        population=dict(population),
        workload=dict(workload),
        inference_source=inference_source,
        phases=tuple(phases),
        prefill_contexts=prefill_contexts,
        prefill_tokens_per_context=prefill_tokens_per_context,
        decode_contexts=decode_contexts,
        decode_tokens_per_context=decode_tokens_per_context,
        random_decode_contexts=random_decode_contexts,
        idle_contexts=context_count - active_contexts,
        random_read_chunk_blocks=chunk_blocks,
        window_shape=window_shape,
        moe_routing=moe_routing_receipt,
        declared_decode_prior_tokens=declared_prior_tokens,
        decode_steps=decode_steps,
        prefill_growth=(
            {
                "contexts": growth_contexts,
                "target_tokens": growth_target_tokens,
                "chunk_tokens": growth_chunk_tokens,
                "chunks": growth_chunks,
                "schedule": growth_schedule,
                "growth_steps": growth_steps_total,
                "decode_while_filling": growth_decode_while_filling,
                "decode_tokens_during_fill": (
                    sum(dwf_tokens_by_context.values())
                ),
                "kv_written_bytes": (
                    growth_contexts
                    * growth_target_tokens
                    * layout.bytes_per_token_per_layer
                    * layout.num_layers
                ),
            }
            if growth_window
            else None
        ),
    )

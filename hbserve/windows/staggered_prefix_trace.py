#!/usr/bin/env python3
"""Parallel serving-window generator: staggered arrivals + shared KV prefix.

A sibling of `build_fixed_footprint_trace`, deliberately independent of it:
this module never touches the mainline window contract or emission sequence.
It reuses only the shared emission primitives (`_Operation`, `_make_phase`)
so its trace groups stay protocol-identical for the remappers.

Window semantics
----------------
`contexts` requests arrive at deterministic pseudo-random chunk offsets and
fill concurrently once arrived (overlapping fills share each iteration's
weight sweep, continuous-batching style). Every context's attention history
is `shared_prefix_tokens` of common prefix KV plus its private suffix. The
prefix KV is written exactly once, by the earliest arrival; later arrivals
read it without writing. A decode tail of `decode_steps` follows, each step
reading every context's full history (prefix + suffix) and appending one
token. Decode-while-filling is intentionally not modeled here.

KV addressing: the shared prefix and each private suffix are layer-major
stripes inside the canonical KV region, so per-layer history reads carry
`prior_tokens * bytes_per_token_per_layer` exactly as the mainline window.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping

from hbserve.windows.memory_trace import (
    MemoryLayout,
    canonical_sha256,
)
from hbserve.windows.window_emitters import (
    FixedFootprintPhase,
    _Operation,
    _make_phase,
    _mix64,
)

__all__ = ["StaggeredPrefixTrace", "build_staggered_prefix_trace"]

TRACE_SCHEMA = {
    "name": "hbfsim.hbf_staggered_prefix_trace",
    "version": 1,
}


class StaggeredPrefixTraceError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise StaggeredPrefixTraceError(message)


def _pos_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class StaggeredPrefixTrace:
    layout: MemoryLayout
    workload: Mapping[str, Any]
    phases: tuple[FixedFootprintPhase, ...]
    contexts: int
    target_tokens: int
    chunk_tokens: int
    shared_prefix_tokens: int
    arrival_offsets_chunks: tuple[int, ...]
    decode_steps: int
    kv_write_tokens: int

    @cached_property
    def read_bytes(self) -> int:
        return sum(phase.read_bytes for phase in self.phases)

    @cached_property
    def write_bytes(self) -> int:
        return sum(phase.write_bytes for phase in self.phases)


def build_staggered_prefix_trace(
    *,
    layout: MemoryLayout,
    population: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> StaggeredPrefixTrace:
    contexts = _pos_int(workload.get("contexts"), "contexts")
    target = _pos_int(workload.get("prefill_target_tokens"), "target tokens")
    chunk = _pos_int(workload.get("prefill_chunk_tokens"), "chunk tokens")
    prefix = workload.get("shared_prefix_tokens", 0)
    if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
        _fail("shared prefix tokens must be a non-negative integer")
    decode_steps = workload.get("decode_steps", 0)
    if not isinstance(decode_steps, int) or decode_steps < 0:
        _fail("decode steps must be a non-negative integer")
    seed = _pos_int(workload.get("arrival_seed", 20260830), "arrival seed")
    max_stagger = workload.get("arrival_max_stagger_chunks", 0)
    if not isinstance(max_stagger, int) or max_stagger < 0:
        _fail("arrival max stagger must be a non-negative integer")
    if target % chunk or (prefix and prefix % chunk):
        _fail("chunk tokens must divide the target and the shared prefix")
    if prefix >= target:
        _fail("shared prefix must be shorter than the context target")

    bpl = layout.bytes_per_token_per_layer
    layers = layout.num_layers
    kv = layout.region(layout.kv_region_id)
    suffix = target - prefix
    suffix_capacity = suffix + decode_steps
    kv_tokens_needed = prefix + contexts * suffix_capacity
    kv_capacity_tokens = kv.bytes // (bpl * layers)
    if kv_tokens_needed > kv_capacity_tokens:
        _fail(
            "shared prefix window needs "
            f"{kv_tokens_needed} KV tokens but the arena holds "
            f"{kv_capacity_tokens}"
        )

    # Layer-major stripes: [prefix | ctx0 suffix | ctx1 suffix | ...].
    def stripe_base(span_begin: int, span_tokens: int, layer: int) -> int:
        return span_begin + layer * span_tokens * bpl

    prefix_begin = kv.begin
    prefix_span = prefix * bpl * layers

    def suffix_begin(context: int) -> int:
        return prefix_begin + prefix_span + context * suffix_capacity * bpl * layers

    offsets = tuple(
        (_mix64(seed ^ (context + 1)) % (max_stagger + 1)) if max_stagger else 0
        for context in range(contexts)
    )
    prefix_writer = min(range(contexts), key=lambda c: (offsets[c], c))

    contract_sha256 = canonical_sha256({
        "schema": TRACE_SCHEMA,
        "contexts": contexts,
        "target_tokens": target,
        "chunk_tokens": chunk,
        "shared_prefix_tokens": prefix,
        "decode_steps": decode_steps,
        "arrival_seed": seed,
        "arrival_offsets_chunks": list(offsets),
        "layout_digest": layout.digest,
    })

    embedding = layout.region("weights/embedding")
    head = layout.region("weights/output_head")
    prefix_chunks = prefix // chunk
    suffix_chunks = suffix // chunk
    # A context's own fill length: the prefix writer also writes the prefix.
    fill_chunks = {
        c: (prefix_chunks if c == prefix_writer else 0) + suffix_chunks
        for c in range(contexts)
    }
    finish = {c: offsets[c] + fill_chunks[c] for c in range(contexts)}
    total_iterations = max(finish.values())

    def filled_tokens(context: int, iteration: int) -> tuple[int, int]:
        """(prefix_tokens_visible, own_suffix_tokens) before this iteration."""
        done = max(0, min(iteration - offsets[context], fill_chunks[context]))
        if context == prefix_writer:
            own_prefix = min(done, prefix_chunks) * chunk
            own_suffix = max(0, done - prefix_chunks) * chunk
            return own_prefix, own_suffix
        # Non-writers see whatever prefix exists globally.
        writer_done = max(
            0, min(iteration - offsets[prefix_writer], prefix_chunks)
        )
        return writer_done * chunk, done * chunk

    phases: list[FixedFootprintPhase] = []

    def add_phase(identifier, stage, layer, object_class, objects, operations):
        phases.append(_make_phase(
            layout=layout,
            protocol_id=len(phases),
            identifier=identifier,
            stage=stage,
            layer=layer,
            object_class=object_class,
            objects=tuple(objects),
            operations=operations,
            contract_sha256=contract_sha256,
        ))

    kv_write_tokens = 0

    def history_ops(context, layer, pre_tokens, own_tokens):
        ops = []
        if pre_tokens:
            ops.append(_Operation(
                op="R",
                address=stripe_base(prefix_begin, prefix, layer),
                byte_count=pre_tokens * bpl,
                region_id=kv.id,
                access_pattern="sequential_stream",
            ))
        if own_tokens:
            ops.append(_Operation(
                op="R",
                address=stripe_base(suffix_begin(context), suffix_capacity, layer),
                byte_count=own_tokens * bpl,
                region_id=kv.id,
                access_pattern="sequential_stream",
            ))
        return ops

    for iteration in range(total_iterations):
        active = [
            c for c in range(contexts)
            if offsets[c] <= iteration < finish[c]
        ]
        if not active:
            continue
        add_phase(
            f"sp_iter{iteration:03d}_embedding", "prefill_growth", None,
            "model_weights", (embedding.id,),
            [_Operation(op="R", address=embedding.begin,
                        byte_count=embedding.bytes, region_id=embedding.id,
                        access_pattern="sequential_stream")],
        )
        for layer in range(layers):
            region = layout.region(f"weights/layer/{layer}")
            ops = [_Operation(op="R", address=region.begin,
                              byte_count=region.bytes, region_id=region.id,
                              access_pattern="sequential_stream")]
            for c in active:
                pre, own = filled_tokens(c, iteration)
                ops.extend(history_ops(c, layer, pre, own))
                writing_prefix = (
                    c == prefix_writer
                    and (iteration - offsets[c]) < prefix_chunks
                )
                if writing_prefix:
                    address = stripe_base(prefix_begin, prefix, layer) + pre * bpl
                else:
                    address = (stripe_base(suffix_begin(c), suffix_capacity, layer)
                               + own * bpl)
                ops.append(_Operation(
                    op="W", address=address, byte_count=chunk * bpl,
                    region_id=kv.id, access_pattern="sequential_append",
                    write_scenario="prefill_KV_append",
                ))
            add_phase(
                f"sp_iter{iteration:03d}_layer_{layer:03d}", "prefill_growth",
                layer, "kv_cache", (kv.id,), ops,
            )
        kv_write_tokens += chunk * len(active) * 1
        add_phase(
            f"sp_iter{iteration:03d}_head", "prefill_growth", None,
            "model_weights", (head.id,),
            [_Operation(op="R", address=head.begin, byte_count=head.bytes,
                        region_id=head.id,
                        access_pattern="sequential_stream")],
        )

    for step in range(decode_steps):
        for layer in range(layers):
            region = layout.region(f"weights/layer/{layer}")
            ops = [_Operation(op="R", address=region.begin,
                              byte_count=region.bytes, region_id=region.id,
                              access_pattern="sequential_stream")]
            for c in range(contexts):
                ops.extend(history_ops(c, layer, prefix, suffix + step))
                ops.append(_Operation(
                    op="W",
                    address=(stripe_base(suffix_begin(c), suffix_capacity, layer)
                             + (suffix + step) * bpl),
                    byte_count=bpl, region_id=kv.id,
                    access_pattern="sequential_append",
                    write_scenario="decode_KV_append",
                ))
            add_phase(
                f"sp_decode{step:03d}_layer_{layer:03d}", "decode", layer,
                "kv_cache", (kv.id,), ops,
            )
        kv_write_tokens += contexts

    return StaggeredPrefixTrace(
        layout=layout,
        workload=dict(workload),
        phases=tuple(phases),
        contexts=contexts,
        target_tokens=target,
        chunk_tokens=chunk,
        shared_prefix_tokens=prefix,
        arrival_offsets_chunks=offsets,
        decode_steps=decode_steps,
        kv_write_tokens=kv_write_tokens,
    )

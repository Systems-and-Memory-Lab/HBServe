#!/usr/bin/env python3
"""Parallel serving-window generator: bounded KV with drop-oldest overwrite.

A sibling of `build_fixed_footprint_trace`, deliberately independent of it:
this module never touches the mainline window contract or emission sequence.
It reuses only the shared emission primitives (`_Operation`, `_make_phase`)
so its trace groups stay protocol-identical for the remappers.

Window semantics
----------------
One context fills to `prefill_target_tokens` by `prefill_chunk_tokens`
chunks, but at most `kv_window_tokens` of KV stay resident: the KV stripe of
every layer is a ring of `kv_window_tokens` slots and each new chunk lands at
`token_index mod window`, overwriting the coldest tokens in place. Attention
history reads cover exactly the resident window - the growing prefix while
the ring is filling, the whole stripe once it has wrapped. A decode tail of
`decode_steps` follows, each step reading the resident window and appending
one token into the ring.

Every offered token is still written exactly once (write volume is invariant
in the window size); what the window changes is how much history each
iteration re-reads. Evicted tokens are simply gone - the quality cost of
serving with a bounded window (or of recomputing on demand) is outside the
memory model. With `kv_window_tokens >= prefill_target_tokens + decode_steps`
nothing wraps and every iteration re-reads its full produced history, the
same growth law as the mainline single-context window.
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
)

__all__ = ["WindowedKvTrace", "build_windowed_kv_trace"]

TRACE_SCHEMA = {
    "name": "hbfsim.hbf_windowed_kv_trace",
    "version": 1,
}


class WindowedKvTraceError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise WindowedKvTraceError(message)


def _pos_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class WindowedKvTrace:
    layout: MemoryLayout
    workload: Mapping[str, Any]
    phases: tuple[FixedFootprintPhase, ...]
    target_tokens: int
    chunk_tokens: int
    window_tokens: int
    decode_steps: int
    kv_write_tokens: int
    dropped_tokens: int

    @cached_property
    def read_bytes(self) -> int:
        return sum(phase.read_bytes for phase in self.phases)

    @cached_property
    def write_bytes(self) -> int:
        return sum(phase.write_bytes for phase in self.phases)


def build_windowed_kv_trace(
    *,
    layout: MemoryLayout,
    population: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> WindowedKvTrace:
    target = _pos_int(workload.get("prefill_target_tokens"), "target tokens")
    chunk = _pos_int(workload.get("prefill_chunk_tokens"), "chunk tokens")
    window = _pos_int(workload.get("kv_window_tokens"), "window tokens")
    decode_steps = workload.get("decode_steps", 0)
    if not isinstance(decode_steps, int) or decode_steps < 0:
        _fail("decode steps must be a non-negative integer")
    if target % chunk or window % chunk:
        _fail("chunk tokens must divide the target and the KV window")

    bpl = layout.bytes_per_token_per_layer
    layers = layout.num_layers
    kv = layout.region(layout.kv_region_id)
    kv_capacity_tokens = kv.bytes // (bpl * layers)
    if window > kv_capacity_tokens:
        _fail(
            f"KV window of {window} tokens needs "
            f"{window * bpl * layers} bytes but the arena holds "
            f"{kv_capacity_tokens} tokens"
        )

    contract_sha256 = canonical_sha256({
        "schema": TRACE_SCHEMA,
        "target_tokens": target,
        "chunk_tokens": chunk,
        "window_tokens": window,
        "decode_steps": decode_steps,
        "layout_digest": layout.digest,
    })

    embedding = layout.region("weights/embedding")
    head = layout.region("weights/output_head")

    def stripe_base(layer: int) -> int:
        return kv.begin + layer * window * bpl

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

    def history_op(layer: int, produced_tokens: int) -> list[_Operation]:
        resident = min(produced_tokens, window)
        if not resident:
            return []
        return [_Operation(
            op="R",
            address=stripe_base(layer),
            byte_count=resident * bpl,
            region_id=kv.id,
            access_pattern="sequential_stream",
        )]

    iterations = target // chunk
    for iteration in range(iterations):
        produced = iteration * chunk
        add_phase(
            f"wk_iter{iteration:03d}_embedding", "prefill_growth", None,
            "model_weights", (embedding.id,),
            [_Operation(op="R", address=embedding.begin,
                        byte_count=embedding.bytes, region_id=embedding.id,
                        access_pattern="sequential_stream")],
        )
        slot = produced % window
        for layer in range(layers):
            region = layout.region(f"weights/layer/{layer}")
            ops = [_Operation(op="R", address=region.begin,
                              byte_count=region.bytes, region_id=region.id,
                              access_pattern="sequential_stream")]
            ops.extend(history_op(layer, produced))
            ops.append(_Operation(
                op="W", address=stripe_base(layer) + slot * bpl,
                byte_count=chunk * bpl, region_id=kv.id,
                access_pattern="sequential_append",
                write_scenario="prefill_KV_append",
            ))
            add_phase(
                f"wk_iter{iteration:03d}_layer_{layer:03d}", "prefill_growth",
                layer, "kv_cache", (kv.id,), ops,
            )
        add_phase(
            f"wk_iter{iteration:03d}_head", "prefill_growth", None,
            "model_weights", (head.id,),
            [_Operation(op="R", address=head.begin, byte_count=head.bytes,
                        region_id=head.id,
                        access_pattern="sequential_stream")],
        )

    for step in range(decode_steps):
        produced = target + step
        slot = produced % window
        for layer in range(layers):
            region = layout.region(f"weights/layer/{layer}")
            ops = [_Operation(op="R", address=region.begin,
                              byte_count=region.bytes, region_id=region.id,
                              access_pattern="sequential_stream")]
            ops.extend(history_op(layer, produced))
            ops.append(_Operation(
                op="W", address=stripe_base(layer) + slot * bpl,
                byte_count=bpl, region_id=kv.id,
                access_pattern="sequential_append",
                write_scenario="decode_KV_append",
            ))
            add_phase(
                f"wk_decode{step:03d}_layer_{layer:03d}", "decode", layer,
                "kv_cache", (kv.id,), ops,
            )

    return WindowedKvTrace(
        layout=layout,
        workload=dict(workload),
        phases=tuple(phases),
        target_tokens=target,
        chunk_tokens=chunk,
        window_tokens=window,
        decode_steps=decode_steps,
        kv_write_tokens=target + decode_steps,
        dropped_tokens=max(0, target + decode_steps - window),
    )

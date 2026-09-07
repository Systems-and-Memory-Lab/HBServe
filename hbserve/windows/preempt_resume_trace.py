#!/usr/bin/env python3
"""Parallel serving-window generator: sequential arrivals with preemption
and resumption.

A sibling of `build_fixed_footprint_trace`, deliberately independent of it:
this module never touches the mainline window contract or emission sequence
and reuses only the shared emission primitives (`_Operation`, `_make_phase`).

Window semantics
----------------
`contexts` requests arrive one after another and fill to
`prefill_target_tokens` by `prefill_chunk_tokens` chunks. Only
`active_slots` contexts are served at any time: while a request fills, the
`active_slots - 1` most recently filled requests decode one token per
iteration alongside it, and every older request is PREEMPTED - it issues no
memory traffic at all until it is resumed. After the last fill, `resume_rounds`
rounds each resume `active_slots` contexts in round-robin order for
`resume_steps` decode steps (full-history reads plus one-token appends)
while the others stay preempted. With `resume_prefetch_steps` > 0 the
scheduler is assumed to know the next round: during the last that many
steps before a round switch (or the last iterations of the final fill), the
contexts that will resume are touched layer by layer - read-only prefetch
accesses spread evenly across those steps - so a memory policy that swaps
whole units back on first touch performs the swap-in under the preceding
round's traffic instead of at the moment of resumption. Prefetch is only
sound when the resuming contexts fit in the capacity left free by the
active set (an N+1-slot pipeline); prefetching a whole round on top of a
full active set evicts hot units and thrashes - measured on mini 4H4F with
three active slots: 3.9x the swap volume and slower than no prefetch.

A preempted request's KV is untouched for whole iterations, so a
capacity-pressured memory system can move it out as whole units and bring
it back when the request resumes - the block-granular offload pattern of a
real serving stack, as opposed to page-granular read-through of KV that is
touched every iteration. The trace itself is address-only; which tier holds
a preempted request's KV is the memory policy's decision.

KV addressing: each context owns a layer-major stripe inside the canonical
KV region, sized for its target plus every decode token it can accumulate.
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

__all__ = ["PreemptResumeTrace", "build_preempt_resume_trace"]

TRACE_SCHEMA = {
    "name": "hbfsim.hbf_preempt_resume_trace",
    "version": 1,
}


class PreemptResumeTraceError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise PreemptResumeTraceError(message)


def _pos_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class PreemptResumeTrace:
    layout: MemoryLayout
    workload: Mapping[str, Any]
    phases: tuple[FixedFootprintPhase, ...]
    contexts: int
    target_tokens: int
    chunk_tokens: int
    active_slots: int
    resume_rounds: int
    resume_steps: int
    resume_prefetch_steps: int
    capacity_tokens: int
    kv_write_tokens: int
    preempted_context_iterations: int
    resume_schedule: tuple[tuple[int, ...], ...]

    @cached_property
    def read_bytes(self) -> int:
        return sum(phase.read_bytes for phase in self.phases)

    @cached_property
    def write_bytes(self) -> int:
        return sum(phase.write_bytes for phase in self.phases)


def build_preempt_resume_trace(
    *,
    layout: MemoryLayout,
    population: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> PreemptResumeTrace:
    contexts = _pos_int(workload.get("contexts"), "contexts")
    target = _pos_int(workload.get("prefill_target_tokens"), "target tokens")
    chunk = _pos_int(workload.get("prefill_chunk_tokens"), "chunk tokens")
    active = _pos_int(workload.get("active_slots"), "active slots")
    rounds = workload.get("resume_rounds", -(-contexts // active))
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        _fail("resume rounds must be a non-negative integer")
    steps = workload.get("resume_steps", 0)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        _fail("resume steps must be a non-negative integer")
    prefetch = workload.get("resume_prefetch_steps", 0)
    if not isinstance(prefetch, int) or isinstance(prefetch, bool) or prefetch < 0:
        _fail("resume prefetch steps must be a non-negative integer")
    if target % chunk:
        _fail("chunk tokens must divide the target")
    if active > contexts:
        _fail("active slots cannot exceed the context count")

    bpl = layout.bytes_per_token_per_layer
    layers = layout.num_layers
    kv = layout.region(layout.kv_region_id)
    chunks_per_fill = target // chunk
    # Decode tokens a context can accumulate: one per iteration while it is
    # an active neighbour of a later fill, plus its share of resume steps.
    fill_growth = (active - 1) * chunks_per_fill
    resume_rounds_per_context = (rounds * active + contexts - 1) // contexts
    capacity = target + fill_growth + resume_rounds_per_context * steps
    kv_capacity_tokens = kv.bytes // (bpl * layers)
    if contexts * capacity > kv_capacity_tokens:
        _fail(
            f"preempt-resume window needs {contexts * capacity} KV tokens "
            f"but the arena holds {kv_capacity_tokens}"
        )

    def stripe_base(context: int, layer: int) -> int:
        return kv.begin + (context * layers + layer) * capacity * bpl

    schedule = tuple(
        tuple((r * active + k) % contexts for k in range(active))
        for r in range(rounds)
    )
    contract_sha256 = canonical_sha256({
        "schema": TRACE_SCHEMA,
        "contexts": contexts,
        "target_tokens": target,
        "chunk_tokens": chunk,
        "active_slots": active,
        "resume_rounds": rounds,
        "resume_steps": steps,
        "resume_prefetch_steps": prefetch,
        "capacity_tokens": capacity,
        "layout_digest": layout.digest,
    })

    embedding = layout.region("weights/embedding")
    head = layout.region("weights/output_head")
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

    def weight_op(region):
        return _Operation(op="R", address=region.begin, byte_count=region.bytes,
                          region_id=region.id, access_pattern="sequential_stream")

    def history_op(context, layer, tokens):
        return _Operation(op="R", address=stripe_base(context, layer),
                          byte_count=tokens * bpl, region_id=kv.id,
                          access_pattern="sequential_stream")

    def append_op(context, layer, at_token, tokens, scenario):
        return _Operation(op="W",
                          address=stripe_base(context, layer) + at_token * bpl,
                          byte_count=tokens * bpl, region_id=kv.id,
                          access_pattern="sequential_append",
                          write_scenario=scenario)

    produced = [0] * contexts
    kv_write_tokens = 0
    preempted_iterations = 0

    def prefetch_layers(remaining: int, window: int) -> range:
        """Layers touched when `remaining` steps are left before a switch
        (1 = last step), spreading all layers evenly over `window` steps."""
        if prefetch == 0 or remaining > window or remaining < 1:
            return range(0)
        slot = window - remaining
        per = -(-layers // window)
        return range(slot * per, min(layers, (slot + 1) * per))

    def prefetch_ops(next_set, current_active, layer):
        return [history_op(c, layer, produced[c])
                for c in next_set
                if c not in current_active and produced[c]]

    for filling in range(contexts):
        decoding = [c for c in range(max(0, filling - active + 1), filling)]
        preempted = filling - len(decoding)
        current = set(decoding) | {filling}
        for iteration in range(chunks_per_fill):
            tag = f"pr_fill{filling:02d}_it{iteration:03d}"
            remaining = chunks_per_fill - iteration
            touch_layers = (
                prefetch_layers(remaining, min(prefetch, chunks_per_fill))
                if filling == contexts - 1 and schedule else range(0)
            )
            add_phase(f"{tag}_embedding", "prefill_growth", None,
                      "model_weights", (embedding.id,), [weight_op(embedding)])
            for layer in range(layers):
                ops = [weight_op(layout.region(f"weights/layer/{layer}"))]
                if produced[filling]:
                    ops.append(history_op(filling, layer, produced[filling]))
                ops.append(append_op(filling, layer, produced[filling], chunk,
                                     "prefill_KV_append"))
                for c in decoding:
                    ops.append(history_op(c, layer, produced[c]))
                    ops.append(append_op(c, layer, produced[c], 1,
                                         "decode_KV_append"))
                if layer in touch_layers:
                    ops.extend(prefetch_ops(schedule[0], current, layer))
                add_phase(f"{tag}_layer_{layer:03d}", "prefill_growth", layer,
                          "kv_cache", (kv.id,), ops)
            add_phase(f"{tag}_head", "prefill_growth", None,
                      "model_weights", (head.id,), [weight_op(head)])
            produced[filling] += chunk
            for c in decoding:
                produced[c] += 1
            kv_write_tokens += chunk + len(decoding)
            preempted_iterations += preempted

    for r, active_set in enumerate(schedule):
        next_set = schedule[r + 1] if r + 1 < len(schedule) else ()
        for step in range(steps):
            tag = f"pr_resume{r:02d}_step{step:03d}"
            touch_layers = prefetch_layers(steps - step, min(prefetch, steps))
            for layer in range(layers):
                ops = [weight_op(layout.region(f"weights/layer/{layer}"))]
                for c in active_set:
                    ops.append(history_op(c, layer, produced[c]))
                    ops.append(append_op(c, layer, produced[c], 1,
                                         "decode_KV_append"))
                if layer in touch_layers:
                    ops.extend(prefetch_ops(next_set, set(active_set), layer))
                add_phase(f"{tag}_layer_{layer:03d}", "decode", layer,
                          "kv_cache", (kv.id,), ops)
            for c in active_set:
                produced[c] += 1
            kv_write_tokens += len(active_set)
            preempted_iterations += contexts - len(active_set)

    return PreemptResumeTrace(
        layout=layout,
        workload=dict(workload),
        phases=tuple(phases),
        contexts=contexts,
        target_tokens=target,
        chunk_tokens=chunk,
        active_slots=active,
        resume_rounds=rounds,
        resume_steps=steps,
        resume_prefetch_steps=prefetch,
        capacity_tokens=capacity,
        kv_write_tokens=kv_write_tokens,
        preempted_context_iterations=preempted_iterations,
        resume_schedule=schedule,
    )

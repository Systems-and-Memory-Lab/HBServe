# Realism and trace integration

Realism is not a property of a generator name. It is a chain of evidence from
the serving population to the question a result is allowed to answer.

## Current fidelity layers

HBServe currently provides:

- request-level arrivals, prompt/output lengths, model mix, and optional token
  IDs;
- exact MoE routes when a router trace is supplied, or a labeled synthetic
  hot-set/Zipf sensitivity model;
- continuous batching, chunked prefill, decode, paged KV allocation,
  migration, and preemption;
- architecture-derived object sizes and exact byte conservation through
  placement;
- a persistent memory-system backend with causal completion feedback.

It does not yet reconstruct GPU kernel issue timing, SM scheduling, cache hits,
coalescing, or collective communication. Consequently, object-level runs can
support capacity and modeled memory-traffic questions, while end-to-end GPU
latency remains outside their evidence boundary.

## From a real serving run to address and arrival traces

The practical pipeline has five stages:

1. **Capture request semantics.** Instrument the serving layer at enqueue,
   schedule, prefill/decode completion, and eviction. Record a monotonic
   timestamp, stable request/model IDs, prompt/output token counts, batch
   membership, KV block-table mutations, and—when policy permits—token IDs and
   MoE expert choices. Hash or redact user content before it reaches the trace.
2. **Capture GPU execution.** Correlate each scheduler iteration with runtime
   kernel and copy activities using correlation IDs. Aggregate profilers are
   useful for calibration, but a complete effective-address stream requires
   GPU-side instruction instrumentation or a trace-capable execution model.
3. **Normalize memory events.** Convert effective addresses into a stable
   allocation-relative form `(allocation_id, offset, size, op, issue_ns)`.
   Never publish process virtual addresses. Preserve warp/lane or transaction
   grouping long enough to reproduce coalescing, then expand or fold events at
   the target cache-line size with an explicit policy.
4. **Join semantic objects.** Bind allocation intervals to weights, KV blocks,
   activations, and runtime metadata. The join must fail on overlaps, unknown
   lifetimes, missing allocation generations, and byte-count disagreement.
5. **Validate before replay.** Compare independent holdout runs on per-layer
   bytes, read/write mix, reuse distance, inter-arrival distribution, burst
   size, concurrency, cache counters, and iteration latency. Calibration and
   validation requests must be disjoint.

The normalized event contract should minimally carry:

```text
trace_id, clock_domain, timestamp_ns, operation,
allocation_id, allocation_generation, byte_offset, byte_length,
request_id, iteration_id, layer_id, object_kind,
kernel_correlation_id, source_provenance
```

Arrival time means the time the transaction becomes eligible at the modeled
memory interface—not request arrival, kernel launch, or kernel completion.
Those clocks must remain distinct and their synchronization error must be
reported. Cache-line expansion must also preserve the original byte range so
alignment amplification can be audited rather than silently counted as model
traffic.

## Acceptance gates for a trace backend

A future HBServe trace backend should not be called hardware-faithful until it
passes all of these gates:

- deterministic schema validation and stable trace digests;
- allocation-lifetime and address-bound checks with zero unresolved events;
- exact accounting from raw byte ranges to cache-line transactions;
- explicit clock synchronization and quantified timestamp uncertainty;
- agreement with independent profiler counters within declared tolerances;
- calibration/holdout separation and versioned hardware/software identity;
- privacy review proving that no prompts, tokens, virtual addresses, hostnames,
  or account identifiers leak into the public artifact.

Until then, HBServe deliberately reports the current object-level backend and
its limitations rather than upgrading synthetic detail into a measurement
claim.

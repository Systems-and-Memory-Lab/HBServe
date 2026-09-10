# Closed-loop prefix caching

Prefix caching is opt-in: set `--prefix-cache-bytes N` on `hbserve run --requests`
or use `prefix_cache_bytes` in a placement JSON. Optional
`--prefix-cache-ttl-ns N` expires entries after the stated time from publication.
The budget is a maximum within the shared HBM KV pool, not extra HBM or a
reserved allocation. Pressure can evict cached ownership before migrating
waiting requests or preempting work.

## Identity and lifecycle

A reusable full token block is identified by the model descriptor digest,
`cache_salt`, its parent block hash, and exact token IDs. Equal prompt lengths
alone never produce hits. Different preceding tokens prevent a downstream
block match. Use a salt distinguishing tenants, weight revisions, adapters, or
other execution state not already distinguished by the descriptor. Conversation
ancestry is not inferred from request names.

Blocks become visible only after the physical batch has completed their writes.
The cache retains one reference per layer, and active requests hold independent
references. LRU or TTL eviction releases cache ownership without freeing blocks
still used by an active request. Partial blocks are never shared. At least the
final prompt token executes to produce logits, even for a fully matching prompt.
Thus output tokens remain unchanged while cached prefill work is skipped.

This cache is **volatile HBM**. It is not an HBF persistent prefix store, does not
survive process/device loss, and does not model CPU hash-table lookup time. It
does not infer kernel-level cache reuse or numerical inference correctness.
It currently accepts dense models only. Synthetic MoE routing may select
different experts for identical tokens in different requests; a token hash
alone cannot establish reusable KV identity there. MoE prefix caching is
rejected until per-prefix route identity is included, rather than granting
false hits. MoE execution without prefix caching remains supported.

## Reproducible synthetic reuse

Synthetic request configuration accepts `shared_prefix_tokens`,
`shared_prefix_groups`, and `prefix_reuse_probability`. With a nonzero shared
length, generated requests contain explicit synthetic token IDs from an
eight-symbol alphabet. These are controlled identity/locality inputs, not
model-generated natural-language text. A separate random stream preserves
arrivals, lengths, and model choices when reuse settings change.

Configured reuse probability is not a measured cache-hit ratio: admission time,
partial blocks, working-set size, and eviction determine actual hits. Result
request rows and summary report `prefix_hit_tokens`; placement receipts report
resident bytes, hits, eviction, expiration, and physical block ownership.

Compare identical generated requests with caching disabled and enabled using
the same named compute model. Fixed-window prefix read distributions remain
useful demand controls but are not substitutes for these request-level results.

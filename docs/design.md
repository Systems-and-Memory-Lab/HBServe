# HBServe design

Status: experimental; schema compatibility is not yet promised.

HBServe has one workload CLI with two explicit execution modes. The closed-loop
serving path is:

```text
hbserve.public_model or hbserve.model
        +
request trace or labeled synthetic request specification
        │
        ▼
request scheduler
  continuous batching · chunked prefill · decode · paged KV
        │
        ▼
placement-independent compiler
  object DAG · exact reads/writes · per-layer compute barriers
        │
        ▼
placement
  objects → HBM / HBF / external addressed transactions
        │
        ▼
external HBFSim persistent session
  physical completion frontier → next scheduler iteration
```

The compiler may define which semantic objects are accessed, but it cannot
choose an address or memory tier. Placement may choose addresses and tiers,
but it cannot alter canonical object bytes. This split is checked in every
mapped-batch receipt.

The fixed-window path is `experiment -> public model ledger -> fixed population
-> logical address phases -> remapper -> external HBFSim`. It uses the same
generator at every scale and holds demand fixed across topologies. It does not
run the request scheduler or report serving latency. See [Fixed windows](windows.md).

## Contracts

All JSON inputs have a `{name, version}` schema field and are parsed strictly.
Unknown or inconsistent data fails before physical execution.

### Models

`hbserve.public_model` v1 contains source-attributed architecture dimensions
and an explicit precision profile. `hbserve model` derives the canonical
`hbserve.model` v2 ledger:

- embedding, final norm, and LM-head bytes;
- attention, dense FFN, router, shared-expert, and routed-expert objects for
  every layer;
- KV bytes per token per layer;
- linear FLOPs per token, pre-routing FLOPs, attention FLOPs per context token,
  and LM-head FLOPs.

Object alignment is explicit. Quantization scale overhead and embedding
precision are never silently replaced by a parameter-count heuristic. Catalog
descriptors in `models/` are memory-capacity and traffic evidence; they are not
by themselves GPU-latency evidence.

Precision supports unquantized matrices, blockwise 128x128 scales, and symmetric
per-output-channel scales. Embeddings default to non-matrix precision; the 70B
miniquick descriptor explicitly selects matrix precision including row scales.

HBServe represents one memory domain. Tensor, pipeline, and expert parallelism
must therefore use a descriptor for the local shard. Collective traffic and
network timing are not inferred.

### Requests and routing

A request is:

```text
(request_id, arrival_ns, model_id, prompt_tokens,
 output_tokens, token_ids | null)
```

For prompt length `P` and output length `O`, the model processes `P + O - 1`
input tokens. The final prefill chunk emits the first output token; each of the
remaining decode iterations consumes the previous token and emits the next.

Token IDs select embedding rows when provided. Otherwise HBServe uses a stable
SHA-256 surrogate and marks the result so it cannot be used as locality
evidence.

For MoE, `hbserve.router_trace` supplies one unique expert set per processed
token and MoE layer. Coverage, bounds, and `top_k` are checked. The alternative
`hbserve.synthetic_router` hot-set/Zipf generator is deterministic sensitivity
input, not measured routing.

### Scheduler and KV

Each iteration admits all runnable decode requests for one token and fills the
remaining token/request budget with oldest-first prefill chunks. This permits
real mixed prefill/decode iterations rather than fixed request cohorts.

KV is allocated in 16-token blocks. Block-table entries appear incrementally,
KV reads stop at valid tokens, appends target exact offsets, and contiguous
pieces may merge only after byte conservation is established. When hot KV does
not fit, the engine migrates cold whole-request KV or preempts the youngest
request. Without a cold tier, preemption discards KV and later recomputes the
prompt. Neither path creates free capacity by bookkeeping alone.

### Compiler and timing

For each scheduled slice the compiler emits:

- row-addressed embedding reads;
- per-layer attention/norm and dense or routed-expert weight reads;
- KV block-table and KV data reads/writes;
- per-layer compute, split at routing availability for MoE;
- final norm and LM-head reads for token emission.

The compiler reads the union of experts selected by the current batch once per
MoE layer. It does not multiply a weight sweep by request count.

Timing providers are named in the result:

- `roofline`: FLOP ledger divided by declared peak throughput and efficiency;
- `memory_only`: zero modeled compute, so TTFT/TPOT claims are disabled;
- `linear`: explicit fixed and per-token layer costs for tests or sensitivity.

Prefetch depth controls how far later-layer memory can issue while current
compute proceeds. Selected experts cannot issue before their own layer's
attention/router reads and routing-ready compute finish. Roofline uses the
pre-routing FLOP ledger; linear MoE timing requires an explicit routing fraction.
No provider is described as hardware measurement.

### Placement and execution

Placement allocates immutable model objects and dynamic KV/block-table regions
in HBM, HBF, or external memory. Presets derive capacities from the resolved
system configuration. The full state—including ranges and digests—is
independent of the session completion frontier.

The HBFSim adapter opens one persistent process. Each canonical batch maps to a
transaction DAG at the current completion origin. Its blocking completion is
fed back to the scheduler; background migrations may remain outside that
frontier but retain explicit dependencies. Finalization drains outstanding
work and emits the physical measurement receipt.

`--simulator` is mandatory. HBServe never guesses a binary inside its own
installation.

## Reproducibility and claim eligibility

Every result identifies model, request, router, placement, run config, system
config, simulator executable, canonical batch, and mapped trace with SHA-256.
The result also states whether requests/router are source-qualified or
synthetic, whether compute is included, and that the current backend is
single-device with no network timing.

A digest proves identity, not realism. Claim eligibility follows the weakest
input: a synthetic arrival stream remains synthetic even when it is replayed
through a detailed physical simulator. See `realism.md` for profiler trace and
holdout-validation requirements.

## Commands

```bash
hbserve capabilities

hbserve model models/llama31-8b-w8-kv-bf16.json \
  --output out/llama31-8b.json

hbserve generate --config examples/quickstart-requests.json \
  --output out/requests.json

hbserve run \
  --simulator /path/to/HBFSim/build/hbfsim \
  --model models/llama31-8b-w8-kv-bf16.json \
  --system configs/4hbm-4hbf-miniquick.cfg \
  --requests examples/quickstart-requests.json \
  --placement weights-hbf-kv-hbm \
  --out out/quickstart
```

`run` accepts repeatable models and system overlays, an optional MoE router,
an explicit placement JSON or checked-in preset, and an optional
`hbserve.run_config` v2 file. Each invocation creates a fresh timestamped
output directory and never overwrites an earlier run.

## Verification

The self-contained suite covers strict JSON parsing, model derivation, mixed
iteration byte accounting, compiler DAGs, roofline math, MoE expert unions,
router coverage, deterministic generators, incremental KV blocks, migration,
preemption, model-cache eviction, stripe mapping, placement derivation,
arrival admission, chunked-prefill emission, token budgets, and timing claim
selection.

```bash
python3 -B tests/test_hbserve.py
```

With a compatible simulator, the physical suite additionally covers detached
migration dependencies, the checked-in CLI example, and simultaneous HBF plus
external-memory execution:

```bash
python3 -B tests/test_hbserve.py --simulator /path/to/hbfsim
```

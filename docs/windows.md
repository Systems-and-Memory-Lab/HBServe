# Fixed memory windows

`hbserve run` accepts exactly one workload source:

| Input | Execution | Suitable metric |
| --- | --- | --- |
| `--requests` | Closed-loop continuous batching, paged KV, named compute timing | Modeled request latency and throughput; TTFT/TPOT only with compute |
| `--experiment` | Deterministic fixed work mapped onto each declared topology | Memory service and final-drain time for identical demand |

Neither mode is a measured GPU kernel/cache model. Fixed-window results must not
be presented as end-to-end serving latency or throughput under request feedback.

## Included experiments

- `configs/windows/miniquick-serving.json`: 100 GB population, Llama-3.1-70B W8
  with per-output-channel BF16 scales, four 20,480-token contexts, chunked
  prefill growth while completed contexts decode, then four decode steps.
- `configs/windows/miniquick-decode.json`: the same model/population with a
  preconstructed KV history and four decode steps; history construction is
  outside the timed window.
- `configs/windows/full-scale-serving.json`: 1 TB population, DeepSeek-V3 FP8,
  four 1,048,576-token contexts, 65,536-token prefill chunks and four decode
  steps. This long-context point is a memory-system stress assumption, not a
  claim that the unmodified model supports useful inference at that length.

Miniquick reduces capacities and host simulation work. Timing parameters are
not multiplied by the capacity scale, and miniquick timings cannot be scaled
arithmetically into full-scale predictions. Both use one implementation.

## Run

From a clean source checkout:

```bash
hbserve run --experiment configs/windows/miniquick-serving.json \
  --preflight-only --out out/preflight

hbserve run --experiment configs/windows/miniquick-serving.json \
  --simulator /path/to/hbfsim --topologies all-hbm,4h4f --out out/miniquick
```

Omitting `--topologies` executes all declared rows; first inspect preflight for
capacity-OOM rows and cost. Preflight always validates the whole matrix and
cannot be combined with `--topologies`. Config, model, and optional inference
source paths resolve relative to the experiment file, not the checkout or CWD.
The lower-level Python APIs resolve relative artifact paths from the CWD.
No HBFSim source tree, Frontier package, model weights, or third-party Python
libraries are required. Empty-to-grown peer-KV preflight also needs
`--simulator` to resolve usable HBF capacity including mapping/GC reserves;
ordinary static/tiered preflight remains engine-free. Physical execution needs a compatible external HBFSim
binary with persistent-session support, `--describe-system`, and build-time
source-SHA receipts. Older incompatible engines fail explicitly rather than
falling back to guessed HBF capacity or incomplete provenance.

Each command creates a unique output directory with `experiment.json`,
`result.json`, and `headline.txt`. Receipts record `execution_mode`, model/config
artifacts, layout and trace digests, phase traffic, and topology outcomes.
`--allow-dirty` permits exploratory use from modified or non-Git installations;
it does not make their results paper-ready.

## Interpretation

The logical address trace is identical across compared topologies. Remappers
provide static direct HBM/HBF placement or HBM-fronted HBF/external backing.
The capacity-relaxed all-HBM row is a timing reference, not an equal-capacity
product or a guaranteed mathematical lower bound. The thermal overlays describe a sustained-serving ceiling start;
their boundary points are validated but are not automatically executed as a
thermal sweep. Final drain includes the declared external writeback boundary;
cached CXL-SSD acceptance excludes later internal NAND destage.

The full-scale default uses synthetic rank-uniform MoE routing with declared
temporal reuse, not measured DeepSeek-V3 token routing. Empirical rank weights
can drive the fixed-window distribution; exact per-token/layer router traces
belong to the closed-loop input contract. Neither a fitted histogram nor a
seeded synthetic stream proves realistic temporal or cross-request correlation.

The migration regression compares every logical transaction, phase ordering,
and total read/write bytes against HBFSim `70c3fd5`. Descriptor/schema changes
intentionally change provenance hashes, not memory demand. This is a software
correctness check, not external validation. Paper claims still need target
serving traces, calibration/holdout comparisons, and sensitivity to scheduling,
locality, routing, and compute assumptions.

Additional Python builders in `hbserve.windows` model bounded sliding KV,
staggered prefix populations, and preempt/resume traffic. These are controlled
memory-demand generators, separate from the identity-aware
[closed-loop prefix cache](prefix-caching.md). Paper-specific analysis runners and private measurement
receipts are not part of this package.

## Tiered HBF and cache policies

A topology may declare `integration_mode: "hbm_fronted_hbf"` with both HBM and
HBF stacks. The common backing remapper handles HBF and external offload;
`mapping.hbf_tiering` or `mapping.external_offload` selects its configuration.
Supported policies are `address_only_lru`, `decayed_lfu`,
`threshold_promotion`, and `class_aware` (KV ranges come from the object layout).
`migration_granularity_bytes`, `transfer_chunk_bytes`,
`read_ahead_window_bytes`, and `reserved_hbm_bytes` are explicit parameters.
Per-topology `mapping` sections override the corresponding experiment section.

Tiered HBM is an inclusive cache: it does not add unique capacity to the HBF
backing. Miss fills, dirty eviction, links, and final dirty writeback are timed
physical transactions, not free remapping. Preflight reports usable cache
capacity and the complete initial backing population. Topology identifiers and
subsets are arbitrary; stack counts and resolved capacities must agree.

## Strict binding versus peer migration

`integration_mode: "peer_hbm_hbf"` uses an empty-to-grown prefill window and
`mapping.peer_kv` with `policy: "static"` or `"capacity_migration"`,
`migration_granularity_bytes`, and `transfer_chunk_bytes`. Both bind non-KV
objects to HBF and create KV on its first HBM write. Static binding forbids
spill; capacity migration moves the least-recently-issued HBM unit to a new
exclusive HBF home. Subsequent access to that home goes directly to HBF.

Only non-KV content is preinstalled. Virtual KV reservation is not live data;
preflight follows the actual write/read order, rejects reads of unwritten
bytes, and reports the peak live allocation. It can identify a static HBM OOM
that migration rescues, or an aggregate HBM+HBF OOM that neither can rescue.
Copies include source reads, D2D payload and destination FTL programs. This is
neither the inclusive tiered cache nor whole-request closed-loop swap.

## Static placement controls

`mapping.direct_placement.policy` selects `capacity_balanced` (default),
`weights_first`, `kv_first`, or `profiled_hotset` for directly attached tiers.
Object priorities put metadata first, then the named class. Every policy uses
the same capacity-bounded remapper, logical HBF interface and physical engine.
No policy disables FTL, GC, refresh or thermal behavior.

Generate an independent training episode with a different locality seed and
the same layout and placement granularity:

```sh
python -m hbserve profile --experiment training.json --output training-profile.json
```

Set `mapping.direct_placement.profile` to that profile path for `profiled_hotset`.
Written units rank first, followed by training read-plus-write byte frequency,
with canonical address breaking ties. The receipt records training and priority
hashes. Measurement cannot be its own training profile. Profile generation and
initial data placement are outside the timed window; this is an offline-trained
static placement policy, not an online cache or migration implementation. The
measurement window still replays every logical access, including its writes.

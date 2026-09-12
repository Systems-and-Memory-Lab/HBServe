# Reference traffic through the native fixed-window interface

This feature branch keeps current upstream coarse execution as the default.
Historical `simple`, its activation supplement and private simulator client
are excluded. The earlier published snapshot is `e42490f`, preserved in the
maintainer's local `archive/simple-trace-20260912` archive and original worktree.
This is a separate feature branch; it is not merged into `main`.

## Interface and execution

For the runnable synthetic example, preparation commands and exact current
model/configuration support, start with [Reference preparation](reference-preparation.md).
For retained real source inputs and portable export, see the
[template catalog](../reference_templates/README.md).

The reference provider plugs into `load_experiment_context`, the same Python
API used by the native experiment runner and HBFSim's paper-suite preparation.
It produces the existing `CanonicalTraceBatch` objects. Native remappers,
placement policies, session submission and metrics then execute those batches.
The reference path does not use the historical ordered-stream client or change
HBFSim core. It replaces traffic only; it does not add native coarse traffic.

1. Generate one reference export with the existing explicit plan/cache inputs:

   ```bash
   hbserve trace reference --plan /data/plan.json \
     --cache-config /data/cache.json --output-root /data/generated-reference
   ```

   A continuous GPU-cache transform is applied across the entire plan. No flat
   pre-cache trace is written. This reference path still materializes the
   post-cache binary for validation/reuse; it is not the fast coarse estimator.

2. Bind every source object to a region of the native model/population layout,
   using the source binding below. Offsets, R/W directions, sizes, record order
   and counts are preserved; object bases are explicitly translated into the
   native canonical address space. Target device placement happens afterward.

3. Use the unchanged native command and topology selectors:

   ```bash
   hbserve run --experiment reference-experiment.json \
     --preflight-only --allow-dirty --out out/reference-check
   hbserve run --experiment reference-experiment.json \
     --simulator /path/to/hbfsim --topologies all-hbm,4h4f \
     --allow-dirty --out out/reference-run
   ```

`--allow-dirty` labels local development, not publication-quality provenance.
No generator is selected implicitly. Existing `--requests` and native
fixed-window configurations retain their original behavior.

## Source binding

Keep the original native experiment schema, population, thermal contract and
topology definitions. Replace its `workload` object with:

```json
{
  "kind": "reference_post_cache_memory_window",
  "reference_source": "reference-binding.json",
  "locality_seed": 0,
  "compute_time": "not_modeled",
  "initial_state": "Describe the exact source GPU-cache and native device initial states",
  "same_trace_for_every_topology": true
}
```

`locality_seed` is retained for the native profile schema. It does not shuffle
or generate this source's addresses. Old coarse context/growth knobs cannot
be included here: they are rejected instead of silently overriding a source.

The binding file has exactly these fields (replace explanatory placeholders
with real values, and include an object binding for every plan object):

```json
{
  "schema": {"name": "hbserve.reference_window_binding", "version": 1},
  "plan": "plan.json",
  "post_cache_root": "generated-reference",
  "plan_sha256": "SHA256 of plan.json",
  "post_cache_sha256": "SHA256 of post-cache.bin",
  "layout_sha256": "Native MemoryLayout.digest for this model/population",
  "source_workload": {"copy": "the exact workload object from plan.json"},
  "schedule": "serial-kernels",
  "object_bindings": [
    {"object_id": "exact plan object ID", "region_id": "exact native region ID", "offset_bytes": 0}
  ],
  "max_records": 1000000
}
```

Relative paths resolve from the containing JSON file. Inspect the native
layout using `load_experiment_context(original_experiment).layout` or
`build_explicit_fixed_population` before replacing the workload. Bindings must
conserve source object size, placement class and within-4-KiB-page offsets;
unowned objects, overlap, overflow, changed artifacts, kernel-order regression,
unsupported schedules and excessive record counts fail before simulation.
Source layer count must match the declared model layout. Source metadata and
coverage are copied into results; a partial source does not become full-model
evidence because it executes through the native runner.

Weights bind to `immutable_weight`, KV to `kv`, and activation/anonymous
workspace objects to `metadata` regions. Runtime buffers remain separately
labeled in `traffic_by_source_kind`; this classification is for native
placement, not a claim that activations are model metadata. A constant object
base translation is not sufficient for layouts requiring internal permutation;
prepare a correct source/layout binding instead of relabeling data.

## Dependencies, states and costs

Each source kernel becomes one batch with a final barrier depending on **all**
its reads and writes. The next kernel starts at the native completion frontier.
Within a kernel, requests are ready together: GPU instruction dependencies,
computation and precise issue/return times are not recovered. Terminal GPU-cache
drain records stay with their source kernel; these are not measured writeback
timestamps. Native end-of-window persistence/final-drain semantics are unchanged.

The adapter verifies source digests before simulation, then buffers the selected
window as native Python transactions. `max_records` is an explicit memory-safety
limit, not a coarsening or scheduling parameter. Reusing an export avoids GPU
collection and cache regeneration; it does not avoid fine simulation cost.
Each topology gets a fresh native device session over the same immutable source.

Direct HBM/HBF placement and address-based backing paths retain their native
remappers. Peer KV migration is deliberately rejected: token/block ownership
and KV-layout conversion require a separate binding beyond object ranges.
Dynamic serving, arbitrary new models/contexts and the research coarse-fit
backend are not implemented by this integration.

## Simulator compatibility

The tested executable is HBFSim `dad246f007ab47f2864358cafc87a4951ce7b699`.
The bundled `hbfsim_client/simulation_session.py` matches that checkout's
current client: quiescence/read-engine receipts, `raw-physical` mapping,
controller-memory accounting and optional zone/wear operations are compatible.
This synchronization is separate from the reference traffic provider. No
simulator core source is carried or patched by this branch.

The twelve existing system profiles were migrated to that engine's accepted
OCP/JEDEC configuration fields. This includes changed hardware assumptions,
not just renamed options: for example the 4+4 profile uses 8 rather than
9.6 Gb/s HBM pins, 16 rather than 4 HBF channels, and 4 rather than 1 us
media read latency. The upstream parameter-provenance ledger is included in
`configs/parameter-provenance.json`; some entries concern upstream profiles
not shipped here. Per-profile headers refer to that root-level ledger.
Do not compare old and new timings as a trace-only effect without freezing
the same engine, resolved configuration, initial state and placement.

Existing native `--requests` and native fixed-window workload semantics are
unchanged. New reference traffic is **opt-in** through the fixed-window
`workload.kind`. A wheel installs Python runtime code; clone this branch or
use the source distribution for templates, example configs and C++ source.

## User-supplied experiment instructions

The shared instructions, read on 2026-09-12, distinguish these workflows:

| Workflow | Locally verifiable entry point | Reference integration status |
| --- | --- | --- |
| Q1 organization/locality and parts of CXL topology studies | HBFSim `experiments.hbf_organization.paper_suite`, importing `hbserve.windows.experiment` and invoking `hbserve run --experiment` or Q1's steady-state runner | Same native context/transaction interface; study configs must explicitly select and bind reference traffic. Peer is not covered. |
| Frozen dispatch/collection | Generated `out/.../worker.py` and `out/.../reporting/start.py` | Not present here; do not execute copied machine-specific commands or assume those workers package new inputs |
| Host-policy/runtime-writeback/lifetime studies | `experiments.hbf_organization.host_policy_comparison` and `experiments.hbf_endurance.batch_gc_lifetime` | Those files/configs were not found in the local workspace; caller compatibility is unverified |

The share is a usage account, not a source/binary identity receipt. Strict
reproduction still needs the original frozen configs, source snapshots and
executable. None of those historical results is relabeled by this candidate.
The compatible local engine tested here is HBFSim `dad246f` with the current
client (upstream `2e228ab`), not the share's unverified frozen binaries.

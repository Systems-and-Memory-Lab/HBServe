# Preparing and reproducing a reference window

The reference runner is not a model-name-to-trace service. This document
separates **reproducing the software path**, **reusing a captured workload**,
and **qualifying a new workload**. None implies the other two.

## Included code and missing inputs

| Step | Included here | Still supplied by the experiment owner |
| --- | --- | --- |
| GPU capture | Raw NVBit readers/converters | GPU capture launcher, compatible NVBit build, model/framework environment, object ownership and captured inputs; no turnkey hardware collector is packaged |
| Compact template compilation | `hbserve.traces._reference.compact_request_template` | Routed transactions, route manifest, object summary and kernel ranges |
| Representative-layer plan | `hbserve.traces._reference.full_model_trace_plan` | Template and exact source scope; any qualified shape/phase generator descriptors |
| Complete model assembly | Consumer accepts supported existing complete plans | Workload-specific prologue, embedding, output-head, nonblock and token-lifecycle assembly recipes are not yet packaged as a generic frontend |
| Continuous modeled GPU cache | `hbserve trace reference`, Python implementation and optional C++ source; two explicit example cache policies | Source plan and all of its referenced artifacts |
| Native layout inspection/binding | `hbserve trace prepare inspect` / `bind` | Native model/experiment and an explicit, semantically correct object map |
| Simulation | Original `hbserve run --experiment` | A compatible separately built HBFSim executable |
| Retained source bundles | [Catalog](../reference_templates/README.md), exact compressed bodies, plans/descriptors, digest-checked export and relocation map | Missing historical validation captures and uncharacterized new workloads |

The `_reference` modules are imported research utilities, not a universal
model registry. Their presence does not certify every accepted input.
Compact retained source bodies and plan dependencies are bundled in the
source catalog. Large full validation captures are not automatically included;
metadata-only and incomplete entries are explicit. A listed source is not
automatically a full-model, native-layout-bound experiment.

## Runnable software-only example

From a checkout or extracted source distribution, install that source using
Python 3.10+ (`python -m pip install -e .`). Then run:

```bash
python examples/reference_native_quickstart.py \
  --phase decode --output-root /tmp/hbserve-reference-demo

hbserve run --experiment /tmp/hbserve-reference-demo/bound/experiment.json \
  --preflight-only --allow-dirty --out /tmp/hbserve-reference-check
```

Use a new output directory each time. The script creates a synthetic template,
two-layer plan, explicit cache policy, small native model/population,
object map, generated post-cache binary, binding and prepared experiment.
It uses no GPU, network, private scripts, or test helpers. A wheel contains
the preparation runtime; use the checkout/source distribution for example
scripts, profiles and optional C++ source.

To exercise physical execution with the separately built compatible engine:

```bash
hbserve run --experiment /tmp/hbserve-reference-demo/bound/experiment.json \
  --simulator /path/to/hbfsim --topologies all-hbm,4h4f,0h8f \
  --allow-dirty --out /tmp/hbserve-reference-simulation
```

The example is deliberately **not a real 7B/32B model, performance benchmark,
or hardware-fidelity result**. `--phase prefill` exercises the other stage
label with the same tiny synthetic access pattern. It is not evidence of
real prefill/decode behavior. `--allow-dirty` is for local development.

The default example uses `examples/reference-cache.json`: 40 MiB, 128 B
lines, 32 B sectors, 16 ways, write-back/write-allocate, no write-miss fetch,
and terminal dirty drain. These are explicit legacy reference assumptions,
not automatically detected GPU parameters. `reference-cache64.json` instead
enables write-miss fetch and aligned 64 B read fills/writebacks. That policy
requires the optional native cache engine, built on Linux with:

```bash
mkdir -p build
c++ -O2 -std=c++17 native/reference_cache.cpp -lcrypto -o build/reference-cache
python examples/reference_native_quickstart.py --phase prefill \
  --cache-engine build/reference-cache --output-root /tmp/hbserve-reference64-demo
```

C++17 and OpenSSL development headers/libraries are required. On macOS,
provide the include/library paths of the installed OpenSSL. The engine is a
GPU-cache transform, not HBFSim. The Python path rejects 64 B mode. Neither
example policy is a hardware-validated universal default.

## Reusing a real source

For included sources, first use `hbserve trace catalog list` / `export` as
documented in the [catalog](../reference_templates/README.md). Pass the exported
`--artifact-map` to generation: source bytes and strict fingerprints remain
unchanged despite server-specific paths inside the original JSON.

For external inputs, keep the exact plan and all referenced template binaries/manifests, graph
bindings and generator descriptors. Source paths must resolve on this
machine. A plan or descriptor may contain absolute paths; moving only the
outer JSON does not make that source bundle portable.

1. Generate once with the source-qualified cache policy:

   ```bash
   hbserve trace reference --plan /data/plan.json \
     --cache-config /data/cache.json --output-root /data/post-cache
   ```

   For 64 B fill/writeback, also pass `--cache-engine`. No flat pre-cache file
   is written. Cache state is continuous across plan segments but starts
   empty per invocation; preceding prefill/token accesses must be in the
   same plan if their residue matters. A final drain is an explicit policy,
   not a recovered hardware writeback timestamp.

   Add `--artifact-map /data/export/artifact-map.json` for a catalog export.
   Every mapped input is size/digest-checked before generation; missing mappings
   never fall back to a similarly named local file. Use `--backend numpy` for
   order-skeleton plans (`python -m pip install numpy`); NumPy is otherwise
   optional. Export paths in `export.json` are relative to that file.

2. Inspect source objects and native regions:

   ```bash
   hbserve trace prepare inspect --experiment /data/native-experiment.json \
     --plan /data/plan.json
   ```

   The output provides both object tables and an **unfilled** mapping list.
   It does not infer bindings by size or name. Prepare `/data/object-map.json`
   with exactly `{"object_bindings": [...]}`; each row has `object_id`,
   `region_id`, and `offset_bytes`, as illustrated by the synthetic example.
   Weight precision/layout, KV token/block ownership and runtime buffers
   must correspond. Valid sizes alone do not prove that correspondence.

3. Generate and validate the bound experiment:

   ```bash
   hbserve trace prepare bind --experiment /data/native-experiment.json \
     --plan /data/plan.json --post-cache-root /data/post-cache \
     --object-bindings /data/object-map.json \
     --initial-state "Exact source cache history and native device initial state" \
     --max-records 1000000 --output-root /data/prepared-reference
   ```

   This calculates hashes, resolves native config paths and runs the native
   preflight. It does not change the source, collect data, infer missing
   accesses, or execute HBFSim. Invalid mappings or count limits produce no
   successful `preparation.json` or final `experiment.json`; partial files
   may remain for diagnosis. The selected window is buffered during
   validation, so choose a bounded window and a deliberate record limit.
   Absolute links are recorded, not all source data copied: rerun preparation
   after transferring the complete input bundle to another machine.

4. Run `/data/prepared-reference/experiment.json` with the original native
   command and chosen topology list. See [native execution](reference-native.md).

## Building a plan from an already compiled template

The included representative-layer utility can be called as a module:

```bash
python -m hbserve.traces._reference.full_model_trace_plan build \
  --template-manifest /data/template.json --template-binary /data/template.bin \
  --model-id EXACT_SOURCE_MODEL --layers 2 --template-layer 0 \
  --phase decode --context-tokens 4 --batch 1 \
  --activation-policy shared_template_arena --output /data/plan.json
```

The numbers above illustrate the **synthetic fixture**, not a recipe for a
different real model. Without an explicit phase-generator descriptor, this
helper repeats the input layer accesses and changes object/kernel bindings:
`--model-id`, `--context-tokens`, `--batch` and `--phase` do not synthesize new
kernel behavior. Merely increasing context metadata leaves the actual
per-layer template traffic unchanged. `--layers` replicates transformer
blocks; it does not capture the missing embedding/output head. Inspect
`coverage.missing`. The default `class_only` activation policy cannot enter
a numerical cache replay; shared/private arenas are named modeling choices.
Older evidence-like labels emitted by this frozen helper are not new
validation results for a new source.

## What can be changed today?

| Change | Current status and required work |
| --- | --- |
| Native coarse model/request configuration | Existing catalog/schema-driven generator remains available; this does not certify detailed-reference accuracy |
| HBM/HBF topology or direct placement for a fixed reference | Reuse the same post-cache export when the source GPU-cache boundary is held fixed and mapping/capacity checks pass; no GPU recollection needed |
| GPU-cache policy | Regenerate post-cache from the same pre-cache plan; old post-cache results are not the new policy |
| Context, batch, prefill/decode or consecutive decode steps | Requires source-qualified templates/descriptors and explicit lifecycle/cache continuity; not a metadata-only edit |
| Layer count in a homogeneous regime | Mechanical block expansion exists; transfer needs validation and complete-model nonblock coverage |
| A new architecture, precision, fused-kernel/attention regime or framework | New source/layout/kernel characterization and holdout validation; no automatic arbitrary-model adapter |
| Very large models / 1M context | Not a general validated reference capability; compact records have u16 object/kernel IDs, u32 per-object offsets and u16 request sizes, plus fine-simulation cost |
| Peer KV migration, dynamic serving with reference, fast coarse-to-fine fitting | Not supplied by this integration |

For a publishable real-model recipe, record model/revision/precision,
framework/GPU/kernel regime, phase/batch/context/steps, source scope,
plan/template/cache hashes, initial/final-state policy, layout binding,
HBFSim revision/configuration, commands and expected correctness counters.
Do not substitute a synthetic fixture for those artifacts or claim a
full-model result from a partial layer/CTA source.

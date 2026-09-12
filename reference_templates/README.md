# Retained reference source catalog

This directory attaches the reference inputs currently available to this
feature branch. It is **not** an arbitrary-model frontend, a collection of
complete hardware post-L2 traces, or a new accuracy validation.

## What is included

| Item | This snapshot |
| --- | ---: |
| Unique compact-template manifests | 159 (162 original locations; duplicates retain aliases) |
| Manifests with exact binary bodies | 95 |
| Metadata-only template entries | 64 |
| Historical generation plans | 42 |
| Plans with every referenced input packaged | 1; K37 source below, not a full-model plan |
| Source-path bindings / unique content blobs | 419 / 371 |
| Compressed source bytes | 73.9 MiB |
| Uncompressed unique source bytes | 627.1 MiB |

[INDEX.md](INDEX.md) lists every plan/template. `catalog.json` records original
paths, SHA-256 identities, sizes, scope, aliases, statuses and missing inputs.
The gzip blobs are content-addressed; each decompressed file is byte-identical
to its retained input. They contain compact address records and metadata, not
model weights. `integrity.json` records the packaged-source integrity check.

Retained bodies include Qwen 7B prefill/decode components, small SGLang captures
and Qwen 32B prefill CTA/phase anchors. The metadata index also references older
1.5B and longer-context validation streams whose bodies are not included.
Model/context coverage is **per entry**, not the union of all names in the
index. A CTA, final norm, sampler or output head is not a full layer or model.

Some older complete-model plans still reference server files that could not
be retrieved after SSH timeouts. There are 58 unresolved source paths; many
are shared by several plans. All 41 incomplete plans remain explicitly
incomplete and are rejected by `catalog export`. Missing large validation
streams are also metadata-only, rather than silently replaced by a smaller
trace. These limitations are source availability, not new accuracy failures.

## Inspect and verify

Clone this feature branch or use its source distribution, then install it:

```bash
python -m pip install -e .
hbserve trace catalog list --catalog reference_templates/catalog.json
hbserve trace catalog list --catalog reference_templates/catalog.json \
  --kind templates --contains qwen7b
hbserve trace catalog verify --catalog reference_templates/catalog.json
```

Verification streams/decompresses all retained blobs and checks their sizes
and digests. It uses no GPU, simulator, network or model expansion. It is not
a hardware-fidelity gate. A wheel contains the runtime commands but **not**
this dataset; pass `--catalog` from a checkout/source archive.

## Quick retained-source check

```bash
python examples/reference_catalog_smoke.py \
  --output-root /tmp/hbserve-retained-cta
```

This consumes the retained Qwen 32B, prefill 8K, K12 CTA-x=0 anchor
`template-d47846e781b6b8e8`: 64 source requests, not full-model inference.
It creates a single-instance source-window plan and applies the explicit
legacy reference-cache example. The expected smoke counters are 64 input
requests / 2 KiB, and 64 post-cache requests / 2 KiB (1 KiB read, 1 KiB final
dirty drain). This small software check is not representative generation cost
for a 32B model, nor proof of the cache policy's accuracy. The separate
[native quickstart](../docs/reference-preparation.md#runnable-software-only-example)
tests the native runner with a clearly synthetic, explicitly bound layout.

## Export a template or existing plan

```bash
hbserve trace catalog export --catalog reference_templates/catalog.json \
  --id template-d47846e781b6b8e8 --output-root /tmp/hbserve-anchor
```

`export.json` identifies the entry point; its paths are relative to the export
directory. `artifact-map.json` maps original absolute source names to local,
hash-checked files. A template export includes its manifest/body, not an
invented full-model plan. See [preparation](../docs/reference-preparation.md)
for representative-layer assembly and its scope limits.

An example of a complete **input closure** is the existing K37 SiLU full-grid
plan, `plan-808a4b07377aee27`. It describes one kernel's full grid, not the
full model. Export is small; fully expanding it is a separate, potentially
expensive operation:

```bash
hbserve trace catalog export --catalog reference_templates/catalog.json \
  --id plan-808a4b07377aee27 --output-root /tmp/hbserve-k37

# Optional full K37 generation, not needed for the packaging check:
hbserve trace reference \
  --plan /tmp/hbserve-k37/artifacts/808a4b07377aee279603bfab7785154f44f55a19fa837fac900fd26499efae38.json \
  --artifact-map /tmp/hbserve-k37/artifact-map.json \
  --cache-config examples/reference-cache.json \
  --output-root /tmp/hbserve-k37-post-cache
```

The relocation map leaves original JSON fingerprints and binaries untouched.
All mapped artifacts are validated before generation; an unmapped input or
changed file fails rather than falling back to the maintainer's filesystem.
Original absolute paths are provenance identifiers: these commands never SSH
to them. Order-skeleton plans require `--backend numpy` and the optional
`python -m pip install '.[reference]'` extra.

After generation, inspect/bind the source objects to a matching native layout,
then use the unchanged `hbserve run --experiment` command. The source-window
provider requires explicit, correct layer/object/layout ownership; catalog
export does not manufacture that binding or certify a new model/context.

## Adding inputs later

Add exact source files as digest-addressed blobs and extend the catalog with
their real scope and required closure. Keep failed and metadata-only entries
explicit. Re-run integrity, relocation, source-generation and native-binding
tests. Hardware accuracy needs separate held-out evidence. Do not rename a
small source as a larger context/model or replace a missing template without
changing and requalifying its evidence contract.

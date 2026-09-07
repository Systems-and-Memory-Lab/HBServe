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
libraries are required. Physical execution needs a compatible external HBFSim
binary with persistent-session support, `--describe-system`, and build-time
source-SHA receipts. Older incompatible engines fail explicitly rather than
falling back to guessed HBF capacity or incomplete provenance.

Each command creates a unique output directory with `experiment.json`,
`result.json`, and `headline.txt`. Receipts record `execution_mode`, model/config
artifacts, layout and trace digests, phase traffic, and topology outcomes.
`--allow-dirty` permits exploratory use from modified or non-Git installations;
it does not make their results paper-ready.

## Interpretation

The logical address trace is identical across compared topologies. Address-only
remappers provide direct HBM/HBF placement or HBM-fronted external backing.
The all-HBM row relaxes capacity and is a timing bound, not an equal-capacity
product. The thermal overlays describe a sustained-serving ceiling start;
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
memory-demand generators, not a claim of prefix-cache lifecycle support in the
closed-loop scheduler. Paper-specific analysis runners and private measurement
receipts are not part of this package.

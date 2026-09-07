#!/usr/bin/env python3
"""Workload-contract resolution for the fixed-footprint window generator.

Validation lives here and nowhere else: `resolve_window_plan` checks every
declared workload field against the population and layout, resolves the
window shape (mixed prefill/decode, decode-only steps, prefill growth), the
MoE routing declaration, and the optional inference-source anchor, and
returns one plain mapping of resolved values. Emission code consumes that
mapping and performs no validation of its own.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from hbserve.windows.memory_trace import (
    MemoryLayout,
    canonical_sha256,
)


TRACE_CONTRACT_SCHEMA = {
    "name": "hbfsim.hbf_fixed_footprint_trace_contract",
    "version": 1,
}

class FixedFootprintTraceError(ValueError):
    """The shared fixed-footprint trace contract is invalid."""


def _fail(message: str) -> NoReturn:
    raise FixedFootprintTraceError(message)


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{description} must be an object")
    return dict(value)


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{description} must be an integer >= {minimum}")
    return value


def _finite(value: Any, description: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        _fail(f"{description} must be finite and >= {minimum}")
    return result


def _text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{description} must be non-empty text")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_receipt(workload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(workload.get("inference_source"), "inference source")
    artifact_text = _text(raw.get("artifact"), "inference source artifact")
    artifact = Path(artifact_text)
    artifact = artifact.resolve()
    if not artifact.is_file() or artifact.is_symlink():
        _fail(f"inference source artifact is not a regular file: {artifact}")
    try:
        document = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read inference source artifact: {error}")
    source = _mapping(document, "inference source document")
    window_name = _text(raw.get("window"), "inference source window")
    policy = _mapping(source.get("window_policy"), "source window policy")
    selectors = _mapping(policy.get("selectors"), "source window selectors")
    selected = _mapping(selectors.get(window_name), "selected source window")
    expected = _mapping(selected.get("expected"), "selected source expectation")
    source_prefill = _integer(
        expected.get("prefill_tokens"), "source prefill tokens", minimum=1
    )
    source_decode = _integer(
        expected.get("decode_tokens"), "source decode tokens", minimum=1
    )
    declared_prefill = _integer(
        raw.get("prefill_tokens"), "declared source prefill tokens", minimum=1
    )
    declared_decode = _integer(
        raw.get("decode_tokens"), "declared source decode tokens", minimum=1
    )
    if (source_prefill, source_decode) != (declared_prefill, declared_decode):
        _fail("declared source token counts disagree with the source artifact")
    source_identity = _mapping(source.get("source"), "inference source identity")
    return {
        "profile": _text(raw.get("profile"), "inference source profile"),
        "window": window_name,
        "artifact": {
            "path": str(artifact),
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        },
        "upstream_repository": source_identity.get("repository"),
        "upstream_revision": source_identity.get("revision"),
        "upstream_trace_sha256": source_identity.get("sha256"),
        "source_prefill_tokens": source_prefill,
        "source_decode_tokens": source_decode,
        "source_prefill_to_decode_ratio": source_prefill / source_decode,
    }



def resolve_window_plan(
    *,
    layout: MemoryLayout,
    population: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the workload declaration and resolve the window plan."""

    if layout.kv_storage_order != "layer_major":
        _fail("fixed-footprint trace requires layer-major KV storage")
    if population.get("layout_sha256") != layout.digest:
        _fail("fixed population and memory layout digests disagree")
    if workload.get("kind") != "fixed_prefill_decode_memory_window":
        _fail("fixed-footprint workload kind is unsupported")

    window_shape = str(workload.get("window_shape", "mixed_prefill_decode"))
    if window_shape not in {
        "mixed_prefill_decode",
        "decode_only_step",
        "prefill_growth",
    }:
        _fail("fixed-footprint window shape is unsupported")
    mixed_window = window_shape == "mixed_prefill_decode"
    growth_window = window_shape == "prefill_growth"
    context_count = _integer(
        workload.get("kv_context_count"), "KV context count", minimum=1
    )
    prefill_contexts = _integer(
        workload.get("prefill_contexts"),
        "prefill contexts",
        minimum=1 if mixed_window else 0,
    )
    prefill_tokens_per_context = _integer(
        workload.get("prefill_tokens_per_context"),
        "prefill tokens per context",
        minimum=1 if mixed_window else 0,
    )
    decode_contexts = _integer(
        workload.get("decode_contexts"),
        "decode contexts",
        minimum=1,
    )
    decode_tokens_per_context = _integer(
        workload.get("decode_tokens_per_context"),
        "decode tokens per context",
        minimum=1,
    )
    random_decode_contexts = _integer(
        workload.get("random_decode_contexts"), "random decode contexts"
    )
    chunk_blocks = _integer(
        workload.get("random_read_chunk_blocks"),
        "random read chunk blocks",
        minimum=1,
    )
    # Optional declared context length: decode reads exactly this many prior
    # tokens instead of the context's whole block-pool partition, so a study
    # can pin a headline context length (for example 1M tokens) while the
    # partition keeps allocator slack like a real serving block pool.
    declared_prior_tokens = workload.get("decode_context_prior_tokens")
    if declared_prior_tokens is not None:
        declared_prior_tokens = _integer(
            declared_prior_tokens,
            "declared decode context prior tokens",
            minimum=1,
        )
    # Multi-step windows execute K consecutive decode iterations: history
    # grows by one token per step, appends advance, and the receipt reports
    # per-step traffic so window stationarity is demonstrated, not asserted.
    decode_steps = _integer(
        workload.get(
            "decode_steps_per_window", 0 if growth_window else 1
        ),
        "decode steps per window",
        minimum=0 if growth_window else 1,
    )
    if decode_steps > 1 and not growth_window and (
        mixed_window or declared_prior_tokens is None
    ):
        _fail(
            "multi-step windows require the decode-only shape and a declared "
            "decode context length"
        )
    # Prefill-growth windows simulate serving from the model-loaded state:
    # declared contexts grow from empty KV to the target length through
    # chunked prefill iterations (each chunk re-reads the weights, reads the
    # history accumulated so far, and writes its own KV), optionally handing
    # off into decode steps over the freshly grown history.
    growth_contexts = 0
    growth_target_tokens = 0
    growth_chunk_tokens = 0
    growth_chunks = 0
    growth_schedule = "concurrent"
    growth_decode_while_filling = False
    if growth_window:
        growth_contexts = _integer(
            workload.get("prefill_growth_contexts"),
            "prefill growth contexts",
            minimum=1,
        )
        # Arrival schedule: "concurrent" batches every growth context into
        # each prefill iteration (they share the weight stream);
        # "sequential" grows one context to its target before the next
        # arrives, so earlier contexts sit idle-resident - their KV is
        # genuinely cold while the active context's KV is hot, and each
        # arrival pays its own weight sweeps.
        growth_schedule = str(
            workload.get("prefill_growth_schedule", "concurrent")
        )
        if growth_schedule not in {"concurrent", "sequential"}:
            _fail(
                "prefill growth schedule must be concurrent or sequential"
            )
        # Continuous-batching semantics: while one arrival prefills, every
        # already-filled context decodes one token per growth iteration in
        # the same forward pass (weight reads are shared with the chunk's
        # sweep). Only meaningful under sequential arrivals - concurrent
        # growth has no filled contexts until the window's decode tail.
        growth_decode_while_filling = bool(
            workload.get("prefill_growth_decode_while_filling", False)
        )
        if growth_decode_while_filling and growth_schedule != "sequential":
            _fail(
                "decode-while-filling requires the sequential growth "
                "schedule"
            )
        growth_target_tokens = _integer(
            workload.get("prefill_target_tokens"),
            "prefill target tokens",
            minimum=layout.block_size_tokens,
        )
        growth_chunk_tokens = _integer(
            workload.get("prefill_chunk_tokens"),
            "prefill chunk tokens",
            minimum=layout.block_size_tokens,
        )
        if (
            growth_chunk_tokens % layout.block_size_tokens
            or growth_target_tokens % growth_chunk_tokens
        ):
            _fail(
                "prefill growth requires whole KV blocks per chunk and whole "
                "chunks per target length"
            )
        growth_chunks = growth_target_tokens // growth_chunk_tokens
        if decode_contexts != growth_contexts:
            _fail(
                "prefill growth declares its decode handoff over the grown "
                "contexts; decode_contexts must equal the growth cohort"
            )
        if declared_prior_tokens is not None:
            _fail(
                "prefill growth derives the decode history from the grown "
                "target; decode_context_prior_tokens must be absent"
            )
        if prefill_contexts or prefill_tokens_per_context:
            _fail(
                "prefill growth replaces the mixed prefill cohort; the "
                "chunked growth loop is the prefill"
            )
        declared_prior_tokens = growth_target_tokens
    locality_seed = _integer(workload.get("locality_seed"), "locality seed")
    if not mixed_window and (
        prefill_contexts or prefill_tokens_per_context
    ):
        _fail("a decode-only step window cannot schedule a prefill cohort")
    active_contexts = prefill_contexts + decode_contexts
    if active_contexts > context_count:
        _fail("prefill and decode cohorts exceed the fixed context population")
    if random_decode_contexts > decode_contexts:
        _fail("random decode contexts exceed decode contexts")
    if prefill_tokens_per_context > layout.block_size_tokens:
        _fail("prefill chunk exceeds one canonical KV block")
    if decode_tokens_per_context != 1:
        _fail("current long-context decode window requires one token per context")

    model_family = str(population.get("model_family", "dense"))
    moe_plan: dict[str, Any] | None = None
    moe_routing_receipt: dict[str, Any] | None = None
    zipf_exponent = 0.0
    routing_seed = 0
    routing_distribution = ""
    routing_rank_weights: tuple[float, ...] | None = None
    step_expert_reuse_probability = 0.0
    if model_family == "moe":
        moe_plan = _mapping(population.get("moe"), "population MoE plan")
        routing = _mapping(
            workload.get("moe_routing"), "MoE routing workload"
        )
        routing_distribution = _text(
            routing.get("distribution"), "MoE routing distribution"
        )
        if routing_distribution == (
            "zipf_without_replacement_over_ranked_experts"
        ):
            zipf_exponent = _finite(
                routing.get("zipf_exponent"), "MoE routing zipf exponent"
            )
        elif routing_distribution == (
            "empirical_rank_frequencies_without_replacement"
        ):
            raw_weights = routing.get("rank_weights")
            if not isinstance(raw_weights, Sequence) or isinstance(
                raw_weights, (str, bytes)
            ):
                _fail("MoE empirical routing requires a rank_weights array")
            routing_rank_weights = tuple(
                _finite(value, "MoE routing rank weight")
                for value in raw_weights
            )
            if not routing_rank_weights or min(routing_rank_weights) <= 0.0:
                _fail("MoE routing rank weights must be positive")
        else:
            _fail("MoE routing distribution is unsupported")
        routing_seed = _integer(routing.get("seed"), "MoE routing seed")
        _text(routing.get("basis"), "MoE routing basis")
        reuse_value = routing.get("step_expert_reuse_probability", 0.0)
        step_expert_reuse_probability = _finite(
            reuse_value, "MoE step expert reuse probability"
        )
        if step_expert_reuse_probability >= 1.0:
            _fail("MoE step expert reuse probability must be below 1.0")
    elif "moe_routing" in workload:
        _fail("moe_routing applies only to MoE model populations")

    if mixed_window:
        inference_source = _source_receipt(workload)
        source_ratio = float(
            inference_source["source_prefill_to_decode_ratio"]
        )
        trace_prefill_tokens = prefill_contexts * prefill_tokens_per_context
        trace_decode_tokens = decode_contexts * decode_tokens_per_context
        trace_ratio = trace_prefill_tokens / trace_decode_tokens
        ratio_error = abs(trace_ratio - source_ratio) / source_ratio
        maximum_error = _finite(
            _mapping(workload.get("inference_source"), "inference source").get(
                "maximum_relative_ratio_error"
            ),
            "maximum prefill/decode ratio error",
        )
        if ratio_error > maximum_error:
            _fail(
                "trace prefill/decode token ratio is outside the declared "
                "source tolerance"
            )
        inference_source.update(
            {
                "trace_prefill_tokens": trace_prefill_tokens,
                "trace_decode_tokens": trace_decode_tokens,
                "trace_prefill_to_decode_ratio": trace_ratio,
                "relative_ratio_error": ratio_error,
                "maximum_relative_ratio_error": maximum_error,
                "use": (
                    "request-shape ratio only; addresses and timing remain "
                    "simulator inputs rather than hardware measurements"
                ),
            }
        )
    else:
        if "inference_source" in workload:
            _fail(
                "declared-regime windows take no prefill/decode source "
                "anchor: the window shape itself is the controlled variable"
            )
        inference_source = {
            "profile": f"declared_{window_shape}",
            "window": window_shape,
            "anchoring": (
                "none;_the_serving_regime_is_the_declared_variable_of_"
                "this_window_shape"
            ),
        }

    contract = {
        "schema": TRACE_CONTRACT_SCHEMA,
        "layout_sha256": layout.digest,
        "population_sha256": canonical_sha256(dict(population)),
        "workload": dict(workload),
        "window_shape": window_shape,
        "measurement_start": (
            "installed_model_and_valid_long_context_decode_KV"
        ),
        "modeled_traffic": [
            "weight_reads",
            "attention_KV_reads",
            "KV_appends",
            "embedding_and_output_reads",
            "block_table_reads_and_allocations",
        ]
        + (["batch_activated_routed_expert_reads"] if moe_plan else []),
        "excluded_traffic": [
            "model_install",
            "prior_decode_context_construction",
            "uncalibrated_activation_or_scratch_traffic",
            "compute_time",
        ],
    }
    contract_sha256 = canonical_sha256(contract)
    return {
        "window_shape": window_shape,
        "mixed_window": mixed_window,
        "growth_window": growth_window,
        "context_count": context_count,
        "prefill_contexts": prefill_contexts,
        "prefill_tokens_per_context": prefill_tokens_per_context,
        "decode_contexts": decode_contexts,
        "decode_tokens_per_context": decode_tokens_per_context,
        "random_decode_contexts": random_decode_contexts,
        "chunk_blocks": chunk_blocks,
        "declared_prior_tokens": declared_prior_tokens,
        "decode_steps": decode_steps,
        "growth_contexts": growth_contexts,
        "growth_target_tokens": growth_target_tokens,
        "growth_chunk_tokens": growth_chunk_tokens,
        "growth_chunks": growth_chunks,
        "growth_schedule": growth_schedule,
        "growth_decode_while_filling": growth_decode_while_filling,
        "locality_seed": locality_seed,
        "model_family": model_family,
        "moe_plan": moe_plan,
        "zipf_exponent": zipf_exponent,
        "routing_seed": routing_seed,
        "routing_distribution": routing_distribution,
        "routing_rank_weights": routing_rank_weights,
        "step_expert_reuse_probability": step_expert_reuse_probability,
        "inference_source": inference_source,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "active_contexts": active_contexts,
    }

#!/usr/bin/env python3
"""Public-dimension model-memory derivation for the workload model catalog.

The paper-facing methodology is one pipeline for every model in the catalog:

1. A descriptor transcribes the publicly documented architecture dimensions
   (layer count, hidden size, attention geometry, FFN/expert geometry,
   vocabulary) and a declared precision profile, with its sources named.
2. This module derives the complete byte ledger from those dimensions with
   explicit formulas — attention/FFN/expert/embedding parameter counts,
   quantization-scale overhead, 4 KiB-aligned expert extents, and the
   KV bytes per token per layer that the attention kind implies (GQA:
   ``2 * kv_heads * head_dim * kv_bytes``; MLA: ``(kv_lora_rank +
   qk_rope_head_dim) * kv_bytes``).  Derived totals must reproduce the
   model's published headline figures (total and activated parameters),
   which is the catalog's cross-check that the transcription is faithful.
3. The derived capacity inputs feed the explicit-population builder and the
   fixed-footprint trace generator unchanged: weights become immutable
   streamed regions (dense sublayer plus per-expert extents for MoE
   layers), and KV becomes a layer-major block arena whose declared
   per-context history length (for example 1M tokens) drives the window's
   reads and appends.

These descriptors are deliberately scoped to architecture-derived storage,
traffic, and FLOP ledgers. Runtime latency claims require a separately
declared and validated execution backend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, NoReturn


PUBLIC_MODEL_SCHEMA_NAME = "hbserve.public_model"
PUBLIC_MODEL_SCHEMA_VERSION = 1
KV_BLOCK_SIZE_TOKENS = 16
BLOCK_TABLE_ENTRY_BYTES = 4
# FP8 block-wise quantization scale granularity (one scale per 128x128 tile).
QUANT_BLOCK_ELEMENTS = 128 * 128


class PublicModelDescriptorError(ValueError):
    """The public model-memory descriptor contract is invalid."""


def _fail(message: str) -> NoReturn:
    raise PublicModelDescriptorError(message)


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{description} must be an object")
    return dict(value)


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{description} must be an integer >= {minimum}")
    return value


def _text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{description} must be non-empty text")
    return value


def _boolean(value: Any, description: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{description} must be a boolean")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def derive_public_model_capacity_inputs(
    model_descriptor_path: Path,
) -> dict[str, Any]:
    """Derive explicit-population capacity inputs from a public descriptor."""

    model_descriptor_path = model_descriptor_path.resolve()
    try:
        descriptor = json.loads(
            model_descriptor_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read public model descriptor: {error}")
    return derive_public_model_ledger(
        descriptor,
        descriptor_artifact={
            "path": str(model_descriptor_path),
            "bytes": model_descriptor_path.stat().st_size,
            "sha256": _sha256_file(model_descriptor_path),
        },
    )


def derive_public_model_ledger(
    descriptor: Any,
    *,
    descriptor_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the byte and FLOP ledger from an in-memory public descriptor.

    ``descriptor_artifact`` names the source (path, bytes, sha256) the caller
    read the document from; converters that translate another descriptor
    schema into this one pass the original artifact so provenance stays with
    the file that was actually read.
    """

    document = _mapping(descriptor, "public model descriptor")
    schema = _mapping(document.get("schema"), "public descriptor schema")
    if (
        schema.get("name") != PUBLIC_MODEL_SCHEMA_NAME
        or schema.get("version") != PUBLIC_MODEL_SCHEMA_VERSION
    ):
        _fail("public model descriptor schema is unsupported")

    model = _mapping(document.get("model"), "public descriptor model")
    architecture_kind = model.get("architecture")
    if architecture_kind not in {
        "dense_decoder_transformer",
        "moe_decoder_transformer",
    }:
        _fail("public descriptor architecture kind is unsupported")
    name = _text(model.get("name"), "public model name")
    source = _mapping(model.get("source"), "public model source")
    for field in (
        "model_repository",
        "public_reference",
        "config_access",
        "dimension_transcription",
    ):
        _text(source.get(field), f"public model source {field}")

    architecture = _mapping(
        document.get("architecture"), "public descriptor architecture"
    )
    num_layers = _integer(
        architecture.get("num_layers"), "public num_layers", minimum=1
    )
    hidden = _integer(
        architecture.get("hidden_size"), "public hidden_size", minimum=1
    )
    vocab = _integer(
        architecture.get("vocab_size"), "public vocab_size", minimum=1
    )
    heads = _integer(
        architecture.get("num_attention_heads"),
        "public num_attention_heads",
        minimum=1,
    )
    tie_embeddings = _boolean(
        architecture.get("tie_word_embeddings"),
        "public tie_word_embeddings",
    )
    if tie_embeddings:
        _fail(
            "tied word embeddings are not supported: the layout streams a "
            "distinct output-head object"
        )

    precision = _mapping(document.get("precision"), "public precision")
    matrix_weight_bytes = _integer(
        precision.get("matrix_weight_bytes"),
        "public matrix weight bytes",
        minimum=1,
    )
    non_matrix_bytes = _integer(
        precision.get("non_matrix_weight_bytes"),
        "public non-matrix weight bytes",
        minimum=1,
    )
    scale_bytes = _integer(precision.get("scale_bytes"), "public scale bytes")
    kv_bytes = _integer(
        precision.get("kv_bytes"), "public KV bytes", minimum=1
    )
    profile_id = _text(precision.get("profile_id"), "public precision profile")

    addressing = _mapping(document.get("addressing"), "public addressing")
    alignment = _integer(
        addressing.get("object_alignment_bytes"),
        "public object alignment",
        minimum=1,
    )
    traffic = _mapping(document.get("traffic_model"), "public traffic model")
    if (
        _integer(
            traffic.get("block_table_entry_bytes"),
            "public block-table entry bytes",
            minimum=1,
        )
        != BLOCK_TABLE_ENTRY_BYTES
    ):
        _fail("public block-table entry bytes must match the shared planner")

    def matrix(parameters: int) -> int:
        """Quantized matrix storage: payload plus block-wise scales."""

        if parameters % QUANT_BLOCK_ELEMENTS:
            scales = parameters // QUANT_BLOCK_ELEMENTS + 1
        else:
            scales = parameters // QUANT_BLOCK_ELEMENTS
        return parameters * matrix_weight_bytes + scales * scale_bytes

    # --- Attention geometry ------------------------------------------------
    attention = _mapping(
        architecture.get("attention"), "public attention geometry"
    )
    attention_kind = attention.get("kind")
    if attention_kind == "gqa":
        kv_heads = _integer(
            attention.get("num_key_value_heads"),
            "public num_key_value_heads",
            minimum=1,
        )
        head_dim = _integer(
            attention.get("head_dim"), "public head_dim", minimum=1
        )
        qk_head_norms = _boolean(
            attention.get("qk_head_norms"), "public qk_head_norms"
        )
        attention_parameters = (
            hidden * heads * head_dim
            + 2 * hidden * kv_heads * head_dim
            + heads * head_dim * hidden
        )
        attention_norm_parameters = (
            2 * hidden + (2 * head_dim if qk_head_norms else 0)
        )
        kv_bytes_per_token_per_layer = 2 * kv_heads * head_dim * kv_bytes
        # QK^T and PV: 2 FLOPs each per head per query/context pair.
        attention_flops_per_context_token = 4 * heads * head_dim
    elif attention_kind == "mla":
        q_lora_rank = _integer(
            attention.get("q_lora_rank"), "public q_lora_rank", minimum=1
        )
        kv_lora_rank = _integer(
            attention.get("kv_lora_rank"), "public kv_lora_rank", minimum=1
        )
        qk_nope_head_dim = _integer(
            attention.get("qk_nope_head_dim"),
            "public qk_nope_head_dim",
            minimum=1,
        )
        qk_rope_head_dim = _integer(
            attention.get("qk_rope_head_dim"),
            "public qk_rope_head_dim",
            minimum=1,
        )
        v_head_dim = _integer(
            attention.get("v_head_dim"), "public v_head_dim", minimum=1
        )
        # MLA projections: q_a, q_b, kv_a (with rope MQA), kv_b, o.
        attention_parameters = (
            hidden * q_lora_rank
            + q_lora_rank * heads * (qk_nope_head_dim + qk_rope_head_dim)
            + hidden * (kv_lora_rank + qk_rope_head_dim)
            + kv_lora_rank * heads * (qk_nope_head_dim + v_head_dim)
            + heads * v_head_dim * hidden
        )
        # Input/post norms plus the two MLA low-rank norms.
        attention_norm_parameters = 2 * hidden + q_lora_rank + kv_lora_rank
        kv_bytes_per_token_per_layer = (
            kv_lora_rank + qk_rope_head_dim
        ) * kv_bytes
        # Scores over the nope+rope query width, values over v_head_dim.
        attention_flops_per_context_token = 2 * heads * (
            qk_nope_head_dim + qk_rope_head_dim
        ) + 2 * heads * v_head_dim
    else:
        _fail("public attention kind must be gqa or mla")

    # --- FFN / expert geometry --------------------------------------------
    ffn = _mapping(architecture.get("ffn"), "public FFN geometry")
    moe_config = ffn.get("moe")
    if (moe_config is None) != (
        architecture_kind == "dense_decoder_transformer"
    ):
        _fail("MoE FFN geometry must match the declared architecture kind")

    embedding_bytes = vocab * hidden * non_matrix_bytes
    final_norm_bytes = hidden * non_matrix_bytes
    output_head_bytes = matrix(vocab * hidden)
    output_head_parameters = vocab * hidden

    layer_bytes: list[int] = []
    dense_sublayer_bytes_by_layer: list[int] = []
    moe_block: dict[str, Any] | None = None
    attention_bytes = matrix(attention_parameters)
    norm_bytes = attention_norm_parameters * non_matrix_bytes

    if moe_config is None:
        dense_intermediate = _integer(
            ffn.get("dense_intermediate_size"),
            "public dense_intermediate_size",
            minimum=1,
        )
        dense_ffn_parameters = 3 * hidden * dense_intermediate
        dense_ffn_bytes = matrix(dense_ffn_parameters)
        uniform_layer = _align_up(
            attention_bytes + norm_bytes + dense_ffn_bytes, alignment
        )
        layer_bytes = [uniform_layer] * num_layers
        dense_sublayer_bytes_by_layer = list(layer_bytes)
        linear_flops_by_layer = [
            2 * (attention_parameters + dense_ffn_parameters)
        ] * num_layers
        router_bytes = 0
        shared_expert_bytes = 0
        expert_payload_bytes = 0
        expert_stride_bytes = 0
        first_moe_layer = num_layers
        total_parameters = (
            vocab * hidden
            + output_head_parameters
            + hidden
            + num_layers
            * (
                attention_parameters
                + attention_norm_parameters
                + dense_ffn_parameters
            )
        )
        activated_parameters = total_parameters
    else:
        moe_geometry = _mapping(moe_config, "public MoE geometry")
        first_moe_layer = _integer(
            moe_geometry.get("first_moe_layer"), "public first_moe_layer"
        )
        routed_experts = _integer(
            moe_geometry.get("routed_experts_per_layer"),
            "public routed_experts_per_layer",
            minimum=2,
        )
        activated = _integer(
            moe_geometry.get("activated_routed_experts_per_token"),
            "public activated routed experts per token",
            minimum=1,
        )
        shared_experts = _integer(
            moe_geometry.get("shared_experts_per_layer"),
            "public shared experts",
        )
        expert_intermediate = _integer(
            moe_geometry.get("expert_intermediate_size"),
            "public expert_intermediate_size",
            minimum=1,
        )
        if first_moe_layer >= num_layers:
            _fail("public first_moe_layer leaves no MoE layers")
        if activated > routed_experts:
            _fail("public MoE activates more routed experts than exist")
        for flag in (
            "group_limited_routing_modeled",
            "multi_token_prediction_module_modeled",
        ):
            if _boolean(
                moe_geometry.get(flag), f"public MoE honesty flag {flag}"
            ):
                _fail(f"public descriptor declares unsupported modeling: {flag}")
        dense_ffn_parameters = 0
        dense_ffn_bytes = 0
        if first_moe_layer > 0:
            dense_intermediate = _integer(
                ffn.get("dense_intermediate_size"),
                "public dense_intermediate_size",
                minimum=1,
            )
            dense_ffn_parameters = 3 * hidden * dense_intermediate
            dense_ffn_bytes = matrix(dense_ffn_parameters)
        expert_parameters = 3 * hidden * expert_intermediate
        router_parameters = hidden * routed_experts
        expert_payload_bytes = matrix(expert_parameters)
        expert_stride_bytes = _align_up(expert_payload_bytes, alignment)
        router_bytes = (
            matrix(router_parameters) + routed_experts * non_matrix_bytes
        )
        shared_expert_bytes = shared_experts * expert_payload_bytes
        linear_flops_by_layer = []
        for layer in range(num_layers):
            if layer < first_moe_layer:
                dense_sublayer = (
                    attention_bytes + norm_bytes + dense_ffn_bytes
                )
                routed_bytes = 0
                linear_flops_by_layer.append(
                    2 * (attention_parameters + dense_ffn_parameters)
                )
            else:
                dense_sublayer = (
                    attention_bytes
                    + norm_bytes
                    + router_bytes
                    + shared_expert_bytes
                )
                routed_bytes = routed_experts * expert_stride_bytes
                linear_flops_by_layer.append(
                    2
                    * (
                        attention_parameters
                        + router_parameters
                        + (activated + shared_experts) * expert_parameters
                    )
                )
            dense_sublayer = _align_up(dense_sublayer, alignment)
            dense_sublayer_bytes_by_layer.append(dense_sublayer)
            layer_bytes.append(dense_sublayer + routed_bytes)
        moe_layer_count = num_layers - first_moe_layer
        total_parameters = (
            vocab * hidden
            + output_head_parameters
            + hidden
            + num_layers * (attention_parameters + attention_norm_parameters)
            + first_moe_layer * dense_ffn_parameters
            + moe_layer_count
            * (
                router_parameters
                + routed_experts
                + (routed_experts + shared_experts) * expert_parameters
            )
        )
        activated_parameters = (
            vocab * hidden
            + output_head_parameters
            + hidden
            + num_layers * (attention_parameters + attention_norm_parameters)
            + first_moe_layer * dense_ffn_parameters
            + moe_layer_count
            * (
                router_parameters
                + routed_experts
                + (activated + shared_experts) * expert_parameters
            )
        )
        moe_block = {
            "first_moe_layer": first_moe_layer,
            "moe_layer_count": moe_layer_count,
            "routed_experts_per_layer": routed_experts,
            "activated_routed_experts_per_token": activated,
            "shared_experts_per_layer": shared_experts,
            "expert_payload_bytes": expert_payload_bytes,
            "expert_stride_bytes": expert_stride_bytes,
            "expert_alignment_padding_bytes": (
                expert_stride_bytes - expert_payload_bytes
            ),
            "dense_sublayer_bytes_by_layer": dense_sublayer_bytes_by_layer,
            "attention_bytes_per_layer": attention_bytes,
            "router_bytes_per_moe_layer": router_bytes,
            "shared_expert_bytes_per_moe_layer": shared_expert_bytes,
            "kv_bytes_per_token_per_layer": kv_bytes_per_token_per_layer,
            "group_limited_routing_modeled": False,
            "multi_token_prediction_module_modeled": False,
        }

    resident_bytes = (
        embedding_bytes
        + final_norm_bytes
        + output_head_bytes
        + sum(layer_bytes)
    )
    kv_page_bytes_per_layer = (
        KV_BLOCK_SIZE_TOKENS * kv_bytes_per_token_per_layer
    )

    result: dict[str, Any] = {
        "model_name": name,
        "model_family": "dense" if moe_block is None else "moe",
        "attention_kind": attention_kind,
        "precision_profile": profile_id,
        "model_descriptor": dict(descriptor_artifact),
        "source": dict(source),
        "total_parameters": total_parameters,
        "activated_parameters_per_token": activated_parameters,
        "kv_bytes_per_token_per_layer": kv_bytes_per_token_per_layer,
        "immutable_weight_backing_bytes": resident_bytes,
        "active_weight_buffer_bytes_per_slot": max(
            embedding_bytes,
            output_head_bytes,
            final_norm_bytes,
            max(layer_bytes),
        ),
        "weight_streaming_objects": {
            "embedding": {"count": 1, "bytes_per_object": embedding_bytes},
            "transformer_layer": {
                "count": num_layers,
                "bytes_per_object": max(layer_bytes),
                "bytes_per_object_by_layer": layer_bytes,
            },
            "final_norm": {"count": 1, "bytes_per_object": final_norm_bytes},
            "output_head": {
                "count": 1,
                "bytes_per_object": output_head_bytes,
            },
        },
        "kv_block_size_tokens": KV_BLOCK_SIZE_TOKENS,
        "kv_page_bytes_per_layer": kv_page_bytes_per_layer,
        "num_layers": num_layers,
        "block_table_entry_bytes": BLOCK_TABLE_ENTRY_BYTES,
        # Per-object byte ledger for HBServe's object-level frontend.
        "components": {
            "object_alignment_bytes": alignment,
            "vocab_size": vocab,
            "hidden_size": hidden,
            "embedding_bytes": embedding_bytes,
            "final_norm_bytes": final_norm_bytes,
            "output_head_bytes": output_head_bytes,
            "attention_bytes_per_layer": attention_bytes,
            "norm_bytes_per_layer": norm_bytes,
            "dense_ffn_bytes_per_layer": dense_ffn_bytes,
            "router_bytes_per_moe_layer": router_bytes,
            "shared_expert_bytes_per_moe_layer": shared_expert_bytes,
            "expert_payload_bytes": expert_payload_bytes,
            "expert_stride_bytes": expert_stride_bytes,
            "first_moe_layer": first_moe_layer,
            "kv_bytes_per_token_per_layer": kv_bytes_per_token_per_layer,
        },
        # FLOP ledger: 2 x activated matrix parameters per token per layer,
        # attention 4 x heads x head_dim (GQA) or the absorbed MLA widths per
        # token per context token, and 2 x vocab x hidden for the LM head.
        "compute": {
            "linear_flops_per_token_by_layer": linear_flops_by_layer,
            "attention_flops_per_token_per_context_token": (
                attention_flops_per_context_token
            ),
            "lm_head_flops_per_token": 2 * output_head_parameters,
        },
    }
    if moe_block is not None:
        result["moe"] = moe_block
    return result

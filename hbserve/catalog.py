#!/usr/bin/env python3
"""Convert public model descriptors into ``hbserve.model`` JSON.

HBServe owns one standalone descriptor schema, ``hbserve.public_model`` v1.
It derives dense and MoE memory/FLOP ledgers directly from public architecture
dimensions and an explicit precision profile. Tensor precision and
quantization metadata are preserved rather than reinterpreted during
conversion.

Every weight object is sized to the descriptor's ``object_alignment_bytes``
(4 KiB in the catalog), exactly as the population ledger lays layers out: a
loader allocates tensors page-aligned, and a whole-object read of an
HBF-resident object is then a whole number of flash pages.  The embedding
table keeps its exact row-addressed size.

Usage: ``python -m hbserve model DESCRIPTOR --output model.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbserve.contracts import (  # noqa: E402
    LayerSpec,
    MODEL_SCHEMA,
    HBServeError,
    ModelSpec,
)
from hbserve.io import load_json_object, write_json_atomic  # noqa: E402


PUBLIC_SCHEMA = {"name": "hbserve.public_model", "version": 1}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HBServeError(f"{description} must be an object")
    return dict(value)


def model_from_ledger(
    ledger: Mapping[str, Any],
    *,
    provenance_source: str,
    provenance_sha256: str,
) -> ModelSpec:
    """Build a ``ModelSpec`` from a canonical per-object capacity ledger."""

    components = _mapping(ledger.get("components"), "ledger components")
    compute = _mapping(ledger.get("compute"), "ledger compute")
    num_layers = int(ledger["num_layers"])
    moe = ledger.get("moe")
    alignment = int(components["object_alignment_bytes"])
    if alignment <= 0 or alignment & (alignment - 1):
        raise HBServeError("ledger object alignment must be a power of two")

    def aligned(value: int) -> int:
        return (int(value) + alignment - 1) // alignment * alignment

    attention = aligned(
        int(components["attention_bytes_per_layer"])
        + int(components["norm_bytes_per_layer"])
    )
    first_moe_layer = int(components["first_moe_layer"])
    linear_flops = compute["linear_flops_per_token_by_layer"]
    tied_embeddings = components.get("tie_word_embeddings", False)
    layers: list[LayerSpec] = []
    for layer in range(num_layers):
        if moe is None or layer < first_moe_layer:
            layers.append(
                LayerSpec(
                    attention_weight_bytes=attention,
                    ffn_weight_bytes=aligned(components["dense_ffn_bytes_per_layer"]),
                    router_weight_bytes=0,
                    shared_expert_weight_bytes=0,
                    expert_weight_bytes=(),
                    top_k=0,
                    kv_bytes_per_token=int(
                        components["kv_bytes_per_token_per_layer"]
                    ),
                    flops_per_token=int(linear_flops[layer]),
                    attention_flops_per_context_token=int(
                        compute["attention_flops_per_token_per_context_token"]
                    ),
                )
            )
        else:
            routed = int(moe["routed_experts_per_layer"])
            layers.append(
                LayerSpec(
                    attention_weight_bytes=attention,
                    ffn_weight_bytes=0,
                    router_weight_bytes=aligned(
                        components["router_bytes_per_moe_layer"]
                    ),
                    shared_expert_weight_bytes=aligned(
                        components["shared_expert_bytes_per_moe_layer"]
                    ),
                    expert_weight_bytes=(aligned(components["expert_stride_bytes"]),)
                    * routed,
                    top_k=int(moe["activated_routed_experts_per_token"]),
                    pre_routing_flops_per_token=int(
                        compute["pre_routing_flops_per_token_by_layer"][layer]
                    ),
                    kv_bytes_per_token=int(
                        components["kv_bytes_per_token_per_layer"]
                    ),
                    flops_per_token=int(linear_flops[layer]),
                    attention_flops_per_context_token=int(
                        compute["attention_flops_per_token_per_context_token"]
                    ),
                )
            )
    # Catalog names may be repository paths ("org/model"); object IDs use
    # "/" as their delimiter, so the model ID keeps only identifier characters.
    model_id = "".join(
        character if character.isascii() and (character.isalnum() or character in "_-.:") else "_"
        for character in str(ledger["model_name"])
    )
    model = ModelSpec(
        model_id=model_id,
        provenance={
            "kind": "published_descriptor",
            "source": provenance_source,
            "sha256": provenance_sha256,
        },
        vocab_size=int(components["vocab_size"]),
        embedding_bytes=int(components["embedding_bytes"]),
        final_norm_bytes=aligned(components["final_norm_bytes"]),
        lm_head_bytes=(
            int(components["embedding_bytes"])
            if tied_embeddings else aligned(components["output_head_bytes"])
        ),
        tie_word_embeddings=tied_embeddings,
        layers=tuple(layers),
        lm_head_flops_per_token=int(compute["lm_head_flops_per_token"]),
    )
    # The population ledger aligns each layer as one region (attention +
    # norms + FFN, or the dense MoE sublayer); this ledger aligns the objects
    # inside a layer separately, so the footprints may differ by at most a
    # few alignment units per layer plus the head, norm, and embedding.
    resident = int(ledger["immutable_weight_backing_bytes"])
    if abs(resident - model.weight_footprint_bytes) > (3 * num_layers + 3) * alignment:
        raise HBServeError(
            "hbserve model footprint diverged from the catalog derivation"
        )
    return model


def convert_descriptor(path: Path) -> ModelSpec:
    """Convert one public HBServe descriptor to a :class:`ModelSpec`."""

    from hbserve.public_model import derive_public_model_ledger

    document = load_json_object(path, "model catalog descriptor")
    schema = document.get("schema")
    artifact = _artifact(path)
    if schema != PUBLIC_SCHEMA:
        raise HBServeError(
            f"{path} is neither a hbserve.model nor a "
            f"{PUBLIC_SCHEMA['name']} v{PUBLIC_SCHEMA['version']} descriptor"
        )
    try:
        ledger = derive_public_model_ledger(
            document,
            descriptor_artifact=artifact,
        )
    except ValueError as error:
        raise HBServeError(f"catalog derivation failed for {path}: {error}") from error
    schema_label = f"{schema['name']}#v{schema['version']}"
    return model_from_ledger(
        ledger,
        provenance_source=f"{artifact['path']}#{schema_label}",
        provenance_sha256=str(artifact["sha256"]),
    )


def load_model_any(path: Path) -> ModelSpec:
    """Load a ``hbserve.model`` JSON or convert a catalog descriptor."""

    document = load_json_object(path, "model")
    if document.get("schema") == MODEL_SCHEMA:
        return ModelSpec.from_dict(document)
    return convert_descriptor(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hbserve model",
        description=(
            "Convert an hbserve.public_model descriptor into an "
            "hbserve.model JSON with byte and FLOP ledgers"
        ),
    )
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        model = convert_descriptor(args.descriptor)
        write_json_atomic(args.output, model.canonical())
    except (OSError, HBServeError) as error:
        print(f"hbserve model conversion failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "model_id": model.model_id,
                "layers": model.num_layers,
                "weight_footprint_bytes": model.weight_footprint_bytes,
                "kv_bytes_per_token": model.kv_bytes_per_token,
                "flops_per_token": sum(
                    layer.flops_per_token for layer in model.layers
                )
                + model.lm_head_flops_per_token,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

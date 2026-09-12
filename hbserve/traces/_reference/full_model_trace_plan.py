#!/usr/bin/env python3
"""Build and query a lazy full-model trace plan from one request template.

The plan stores the representative layer once and creates only per-layer
object/kernel bindings.  It can expose exact object-relative weight/KV
requests and one of three explicit activation contracts:

* ``class_only``: no numerical activation address is invented;
* ``shared_template_arena``: all layers reuse the representative activation
  layout (modeled endpoint); or
* ``private_per_layer``: each layer gets a disjoint copy of that layout
  (modeled endpoint).

Materializing billions of JSON events is never the default.  ``expand_slice``
provides bounded on-demand records, while downstream streaming transforms can
iterate the same fixed-width template directly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from hbserve.traces._reference.compact_request_template import (
    OPERATION_NAME,
    RECORD_BYTES,
    RECORD_STRUCT,
    SCHEMA as TEMPLATE_SCHEMA,
    SASS_FLAG_SCHEMA,
    known_request_flags,
    load_json,
    require,
    sha256_file,
    sha256_value,
)
from hbserve.traces._reference.phase_aware_cta_generator import (
    PreparedSegmentGenerator,
    prepare_segment_generator,
    segment_generation_census,
)


PLAN_SCHEMA = {"name": "hbfsim.lazy_full_model_trace_plan", "version": 1}
KERNEL_GRAPH_SCHEMA = {
    "name": "hbfsim.full_model_kernel_graph_binding",
    "version": 1,
}
ACTIVATION_POLICIES = (
    "class_only",
    "shared_template_arena",
    "private_per_layer",
)
TRANSIENT_KINDS = {"activation", "anonymous"}
PAGE_BYTES = 4096
LAYER_NAME = re.compile(r"^model\.layers\.(?:\{LAYER\}|[0-9]+)\.(.+)$")


def align_up(value: int, alignment: int = PAGE_BYTES) -> int:
    return (value + alignment - 1) // alignment * alignment


def layer_object_name(source_name: str, kind: str, layer: int) -> str:
    if kind == "weight":
        match = LAYER_NAME.match(source_name)
        require(match is not None, f"weight lacks a layer-relative name: {source_name}")
        return f"model.layers.{layer}.{match.group(1)}"
    if kind == "kv_cache":
        return f"model.layers.{layer}.{source_name}"
    require(kind in TRANSIENT_KINDS, f"unsupported layer object kind: {kind}")
    return f"model.layers.{layer}.{kind}_template.{source_name}"


class ObjectAllocator:
    def __init__(self) -> None:
        self.cursor = 0
        self.objects: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}

    def allocate(
        self,
        *,
        object_id: str,
        kind: str,
        byte_count: int,
        evidence: str,
        source_template_object_index: int,
        layer: int | None,
    ) -> dict[str, Any]:
        require(object_id not in self.by_id, f"duplicate target object {object_id}")
        self.cursor = align_up(self.cursor)
        item = {
            "target_object_index": len(self.objects),
            "object_id": object_id,
            "kind": kind,
            "logical_address": self.cursor,
            "bytes": byte_count,
            "address_evidence": evidence,
            "source_template_object_index": source_template_object_index,
            "layer": layer,
        }
        self.cursor += align_up(byte_count)
        self.objects.append(item)
        self.by_id[object_id] = item
        return item


def template_kind_totals(template: dict[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {}
    for item in template["objects"]:
        kind = str(item["kind"])
        counter = result.setdefault(kind, Counter())
        compiled = item.get("compiled") or {}
        counter["requests"] += int(compiled.get("requests", 0))
        counter["bytes"] += int(compiled.get("bytes", 0))
        counter["r_requests"] += int(compiled.get("r_requests", 0))
        counter["r_bytes"] += int(compiled.get("r_bytes", 0))
        counter["w_requests"] += int(compiled.get("w_requests", 0))
        counter["w_bytes"] += int(compiled.get("w_bytes", 0))
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def validate_template_manifest(template: dict[str, Any], binary_path: Path) -> None:
    require(
        template.get("schema") in (TEMPLATE_SCHEMA, SASS_FLAG_SCHEMA),
        "unsupported template schema",
    )
    require(template.get("status") == "PASS", "template did not pass compilation")
    requests = int(template.get("requests", 0))
    require(requests > 0, "template has no requests")
    require(
        int(template.get("binary_bytes", -1)) == requests * RECORD_BYTES,
        "template manifest has an invalid fixed-width size",
    )
    require(binary_path.is_file(), f"missing template binary: {binary_path}")
    require(
        binary_path.stat().st_size == int(template["binary_bytes"]),
        "template binary size disagrees with its manifest",
    )
    objects = template.get("objects")
    require(isinstance(objects, list) and objects, "template has no objects")
    require(
        [int(item["template_object_index"]) for item in objects]
        == list(range(len(objects))),
        "template object indices are not dense",
    )


def validate_kernel_graph_binding(
    binding: dict[str, Any], *, layers: int, block_kernels: int
) -> None:
    require(binding.get("schema") == KERNEL_GRAPH_SCHEMA,
            "unsupported kernel-graph binding schema")
    require(str(binding.get("status", "")).startswith("PASS_"),
            "kernel-graph binding did not pass")
    metadata = binding.get("binding") or {}
    require(int(metadata.get("layers", -1)) == layers,
            "kernel-graph layer count disagrees with the plan")
    require(int(metadata.get("block_kernels", -1)) == block_kernels,
            "kernel-graph block width disagrees with the template")
    blocks = (binding.get("coverage") or {}).get("blocks")
    require(isinstance(blocks, list) and len(blocks) == layers,
            "kernel-graph binding lacks one span per transformer layer")
    require(
        [int(item["layer"]) for item in blocks] == list(range(layers)),
        "kernel-graph layer spans are not dense and ordered",
    )


def load_phase_segment_generator(
    *,
    descriptor_path: Path,
    template: dict[str, Any],
    template_binary_path: Path,
) -> tuple[dict[str, Any], PreparedSegmentGenerator, dict[str, Any]]:
    descriptor_path = descriptor_path.resolve()
    descriptor = load_json(descriptor_path)
    prepared = prepare_segment_generator(
        descriptor=descriptor,
        template_manifest=template,
        template_binary_path=template_binary_path,
    )
    census = segment_generation_census(
        prepared=prepared,
        source_path=template_binary_path,
    )
    require(int(census["totals"].get("requests", 0)) > 0,
            "phase-aware generator emits no requests")
    require(int(census["totals"].get("bytes", 0)) > 0,
            "phase-aware generator emits no bytes")
    bound = {
        **descriptor,
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": sha256_file(descriptor_path),
    }
    return bound, prepared, census


def generated_kind_totals(
    source_objects: list[dict[str, Any]], census: dict[str, Any]
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {}
    by_object = census.get("by_object") or {}
    for source in source_objects:
        index = int(source["template_object_index"])
        metrics = by_object.get(str(index), {})
        counter = result.setdefault(str(source["kind"]), Counter())
        for name, value in metrics.items():
            counter[str(name)] += int(value)
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def build_plan(
    *,
    template_manifest_path: Path,
    template_binary_path: Path,
    model_id: str,
    layers: int,
    phase: str,
    context_tokens: int,
    batch: int,
    activation_policy: str,
    template_layer: int,
    kernel_graph_binding_path: Path | None = None,
    phase_generator_descriptor_path: Path | None = None,
) -> dict[str, Any]:
    template_manifest_path = template_manifest_path.resolve()
    template_binary_path = template_binary_path.resolve()
    template = load_json(template_manifest_path)
    validate_template_manifest(template, template_binary_path)
    require(model_id, "model id cannot be empty")
    require(layers > 0, "layer count must be positive")
    require(phase in {"prefill", "decode"}, "phase must be prefill or decode")
    require(context_tokens > 0, "context tokens must be positive")
    require(batch > 0, "batch must be positive")
    require(
        activation_policy in ACTIVATION_POLICIES,
        f"unsupported activation policy {activation_policy!r}",
    )
    require(0 <= template_layer < layers, "template layer is outside the model")

    allocator = ObjectAllocator()
    source_objects = template["objects"]
    shared_transient: dict[int, dict[str, Any]] = {}
    segments = []
    source_template_requests = int(template["requests"])
    source_template_bytes = int(template["request_bytes"])
    template_kernels = len(template["kernels"])
    phase_generator = None
    prepared_phase_generator = None
    phase_generator_census = None
    phase_generator_descriptor_sha256 = None
    if phase_generator_descriptor_path is not None:
        (
            phase_generator,
            prepared_phase_generator,
            phase_generator_census,
        ) = load_phase_segment_generator(
            descriptor_path=phase_generator_descriptor_path,
            template=template,
            template_binary_path=template_binary_path,
        )
        phase_generator_descriptor_sha256 = phase_generator["descriptor_sha256"]
        template_requests = int(phase_generator_census["totals"]["requests"])
        template_bytes = int(phase_generator_census["totals"]["bytes"])
    else:
        template_requests = source_template_requests
        template_bytes = source_template_bytes
    kernel_graph_binding = None
    kernel_graph_sha256 = None
    if kernel_graph_binding_path is not None:
        kernel_graph_binding_path = kernel_graph_binding_path.resolve()
        kernel_graph_binding = load_json(kernel_graph_binding_path)
        validate_kernel_graph_binding(
            kernel_graph_binding, layers=layers, block_kernels=template_kernels
        )
        kernel_graph_sha256 = sha256_file(kernel_graph_binding_path)
    request_cursor = 0
    kernel_cursor = 0

    for layer in range(layers):
        bindings = []
        for source in source_objects:
            source_index = int(source["template_object_index"])
            kind = str(source["kind"])
            byte_count = (
                int(prepared_phase_generator.generator.target_extents[source_index])
                if prepared_phase_generator is not None
                else int(source["bytes"])
            )
            source_name = str(source["source_name"])
            if kind in {"weight", "kv_cache"}:
                target = allocator.allocate(
                    object_id=layer_object_name(source_name, kind, layer),
                    kind=kind,
                    byte_count=byte_count,
                    evidence="template-expanded exact object-relative binding",
                    source_template_object_index=source_index,
                    layer=layer,
                )
                binding = {
                    "source_template_object_index": source_index,
                    "mode": "numeric",
                    "target_object_index": target["target_object_index"],
                    "target_object_id": target["object_id"],
                    "target_logical_address": target["logical_address"],
                    "object_offset_evidence": "exact held-out cross-layer gate",
                }
            elif kind in TRANSIENT_KINDS and activation_policy == "class_only":
                binding = {
                    "source_template_object_index": source_index,
                    "mode": "symbolic_class_only",
                    "target_object_index": None,
                    "target_object_id": None,
                    "target_logical_address": None,
                    "object_offset_evidence": "unresolved",
                }
            elif kind in TRANSIENT_KINDS and activation_policy == "shared_template_arena":
                target = shared_transient.get(source_index)
                if target is None:
                    target = allocator.allocate(
                        object_id=f"shared.{kind}_template.{source_name}",
                        kind=kind,
                        byte_count=byte_count,
                        evidence="modeled shared representative-layer transient arena",
                        source_template_object_index=source_index,
                        layer=None,
                    )
                    shared_transient[source_index] = target
                binding = {
                    "source_template_object_index": source_index,
                    "mode": "numeric_modeled",
                    "target_object_index": target["target_object_index"],
                    "target_object_id": target["object_id"],
                    "target_logical_address": target["logical_address"],
                    "object_offset_evidence": "modeled representative-layer layout",
                }
            elif kind in TRANSIENT_KINDS and activation_policy == "private_per_layer":
                target = allocator.allocate(
                    object_id=layer_object_name(source_name, kind, layer),
                    kind=kind,
                    byte_count=byte_count,
                    evidence="modeled private representative-layer transient arena",
                    source_template_object_index=source_index,
                    layer=layer,
                )
                binding = {
                    "source_template_object_index": source_index,
                    "mode": "numeric_modeled",
                    "target_object_index": target["target_object_index"],
                    "target_object_id": target["object_id"],
                    "target_logical_address": target["logical_address"],
                    "object_offset_evidence": "modeled representative-layer layout",
                }
            else:
                raise ValueError(f"unsupported source object kind {kind!r}")
            bindings.append(binding)

        segment = {
                "segment_id": f"transformer_layer_{layer:04d}",
                "kind": "representative_layer_instance",
                "layer": layer,
                "template_request_begin": 0,
                "template_request_end_exclusive": source_template_requests,
                "expanded_request_begin": request_cursor,
                "expanded_request_end_exclusive": request_cursor + template_requests,
                "expanded_kernel_ordinal_begin": kernel_cursor,
                "expanded_kernel_ordinal_end_exclusive": kernel_cursor + template_kernels,
                "object_bindings": bindings,
            }
        if phase_generator is not None:
            segment["generator"] = phase_generator
        if kernel_graph_binding is not None:
            graph_block = kernel_graph_binding["coverage"]["blocks"][layer]
            # Once a complete observed graph is bound, kernel ordinals are
            # full-model ordinals rather than a transformer-only namespace.
            # Keeping the old zero-based block ordinal would silently attach
            # the first layer's requests to prefix kernels and shift every
            # cadence/cache boundary in a complete-model replay.
            segment["expanded_kernel_ordinal_begin"] = int(
                graph_block["kernel_begin"]
            )
            segment["expanded_kernel_ordinal_end_exclusive"] = int(
                graph_block["kernel_end_exclusive"]
            )
            segment["observed_kernel_begin"] = int(graph_block["kernel_begin"])
            segment["observed_kernel_end_exclusive"] = int(
                graph_block["kernel_end_exclusive"]
            )
            segment["observed_timing"] = {
                key: graph_block[key]
                for key in (
                    "first_start_ns",
                    "last_end_ns",
                    "envelope_ns",
                    "summed_kernel_duration_ns",
                )
            }
        segments.append(segment)
        request_cursor += template_requests
        kernel_cursor += template_kernels

    by_kind = (
        generated_kind_totals(source_objects, phase_generator_census)
        if phase_generator_census is not None
        else template_kind_totals(template)
    )
    expanded_by_kind = {
        kind: {metric: value * layers for metric, value in totals.items()}
        for kind, totals in by_kind.items()
    }
    plan_core = {
        "workload": {
            "model_id": model_id,
            "layers": layers,
            "phase": phase,
            "context_tokens": context_tokens,
            "batch": batch,
            "template_layer": template_layer,
        },
        "activation_policy": activation_policy,
        "template_binary_sha256": template["binary_sha256"],
        "objects": allocator.objects,
        "segments": segments,
        "kernel_graph_binding_sha256": kernel_graph_sha256,
        "phase_generator_descriptor_sha256": phase_generator_descriptor_sha256,
    }
    coverage = "transformer_blocks_complete_nonblock_segments_missing"
    missing = [
        "model prologue/embedding request template",
        "final normalization/output-head request template",
    ]
    nonblock_segments = []
    if kernel_graph_binding is None:
        missing.append("full-model observed kernel-graph binding")
    else:
        for name in ("prefix", "suffix"):
            span = kernel_graph_binding["coverage"][name]
            nonblock_segments.append(
                {
                    "segment_id": f"nonblock_{name}",
                    "kind": f"observed_{name}",
                    "observed_kernel_begin": int(span["kernel_begin"]),
                    "observed_kernel_end_exclusive": int(
                        span["kernel_end_exclusive"]
                    ),
                    "observed_kernels": int(span["kernels"]),
                    "observed_timing": {
                        key: span[key]
                        for key in (
                            "first_start_ns",
                            "last_end_ns",
                            "envelope_ns",
                            "summed_kernel_duration_ns",
                        )
                    },
                    "request_template_status": "missing",
                }
            )
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "PASS_TRANSFORMER_BLOCK_PLAN",
        "classification": "lazy object-relative full-transformer-block request plan",
        "coverage": {
            "status": coverage,
            "transformer_layers": layers,
            "transformer_layer_requests": request_cursor,
            "transformer_layer_request_bytes": template_bytes * layers,
            "transformer_layer_kernels": kernel_cursor,
            "observed_full_graph_bound": kernel_graph_binding is not None,
            "missing": missing,
        },
        "workload": plan_core["workload"],
        "template": {
            "manifest": str(template_manifest_path),
            "manifest_sha256": sha256_file(template_manifest_path),
            "binary": str(template_binary_path),
            "binary_sha256": template["binary_sha256"],
            "requests": source_template_requests,
            "request_bytes": source_template_bytes,
            "kernels": template_kernels,
            "objects": len(source_objects),
            "generated_requests_per_layer": (
                template_requests if phase_generator is not None else None
            ),
            "generated_request_bytes_per_layer": (
                template_bytes if phase_generator is not None else None
            ),
        },
        "phase_generator": (
            {
                "descriptor": str(phase_generator_descriptor_path.resolve()),
                "descriptor_sha256": phase_generator_descriptor_sha256,
                "census": phase_generator_census,
            }
            if phase_generator_descriptor_path is not None
            else None
        ),
        "activation_contract": {
            "policy": activation_policy,
            "covered_transient_kinds": sorted(TRANSIENT_KINDS),
            "exact_cross_layer_identity_claimed": False,
            "interpretation": (
                "activation and anonymous-workspace addresses remain symbolic and cannot enter a numerical cache replay"
                if activation_policy == "class_only"
                else "representative-layer transient offsets are a named modeled endpoint"
            ),
        },
        "objects": allocator.objects,
        "logical_population_bytes": align_up(allocator.cursor),
        "segments": segments,
        "nonblock_segments": nonblock_segments,
        "template_by_kind": by_kind,
        "expanded_by_kind": expanded_by_kind,
        "plan_fingerprint_sha256": sha256_value(plan_core),
        "provenance": {
            "weight_kv_object_offsets": "template-expanded; held-out layers 14 and 27 exact",
            "transient_addresses": (
                "unresolved symbolic class"
                if activation_policy == "class_only"
                else "modeled sensitivity endpoint"
            ),
            "kernel_order": (
                "directly observed full-model NSYS graph"
                if kernel_graph_binding is not None
                else "template-local only; full-model graph not yet bound"
            ),
            "issue_time": (
                "observed per-kernel envelopes; per-load/store issue times remain modeled"
                if kernel_graph_binding is not None
                else "unknown until an observed full-model graph/cadence is bound"
            ),
        },
        "not_claimed": [
            "complete model trace before non-block segments are added",
            "exact activation allocation or cache conflicts",
            "generality beyond the bound model/phase/context/batch/software shape",
            "production request issue timestamps",
            "production cross-CTA scheduling order",
        ],
    }
    if kernel_graph_binding is not None:
        plan["kernel_graph"] = {
            "binding": str(kernel_graph_binding_path),
            "binding_sha256": kernel_graph_sha256,
            "status": kernel_graph_binding["status"],
            "full_graph_kernels": kernel_graph_binding["coverage"][
                "full_graph_kernels"
            ],
            "prefix_kernels": kernel_graph_binding["coverage"]["prefix"][
                "kernels"
            ],
            "suffix_kernels": kernel_graph_binding["coverage"]["suffix"][
                "kernels"
            ],
        }
    return plan


def plan_lookup(plan: dict[str, Any], layer: int) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    segments = [item for item in plan["segments"] if int(item["layer"]) == layer]
    require(len(segments) == 1, f"plan has no unique segment for layer {layer}")
    bindings = {
        int(item["source_template_object_index"]): item
        for item in segments[0]["object_bindings"]
    }
    return segments[0], bindings


def expand_slice(
    *,
    plan: dict[str, Any],
    template_binary_path: Path,
    layer: int,
    begin: int,
    count: int,
    include_kinds: set[str] | None = None,
) -> list[dict[str, Any]]:
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported plan schema")
    template_binary_path = template_binary_path.resolve()
    segment, bindings = plan_lookup(plan, layer)
    template_requests = int(plan["template"]["requests"])
    require(begin >= 0 and count >= 0 and begin + count <= template_requests,
            "requested slice escapes the template")
    source_objects = load_json(Path(plan["template"]["manifest"]))["objects"]
    source_by_index = {
        int(item["template_object_index"]): item for item in source_objects
    }
    target_objects = {
        int(item["target_object_index"]): item for item in plan["objects"]
    }

    with template_binary_path.open("rb") as stream:
        stream.seek(begin * RECORD_BYTES)
        payload = stream.read(count * RECORD_BYTES)
    require(len(payload) == count * RECORD_BYTES, "template slice is truncated")
    result = []
    for local_index, record in enumerate(RECORD_STRUCT.iter_unpack(payload)):
        object_index, object_offset, kernel_ordinal, byte_count, operation, flags = record
        require(known_request_flags(flags), "unsupported compact-template flags")
        source = source_by_index[object_index]
        kind = str(source["kind"])
        if include_kinds is not None and kind not in include_kinds:
            continue
        binding = bindings[object_index]
        target_index = binding["target_object_index"]
        target = target_objects[int(target_index)] if target_index is not None else None
        result.append(
            {
                "record_type": "expanded_request",
                "expanded_sequence_index": int(segment["expanded_request_begin"]) + begin + local_index,
                "layer": layer,
                "kernel_ordinal": int(segment["expanded_kernel_ordinal_begin"]) + kernel_ordinal,
                "template_kernel_ordinal": kernel_ordinal,
                "op": OPERATION_NAME[operation],
                "bytes": byte_count,
                "kind": kind,
                "target_object_id": target["object_id"] if target is not None else None,
                "target_logical_address": (
                    int(target["logical_address"]) + object_offset
                    if target is not None
                    else None
                ),
                "template_object_index": object_index,
                "template_object_offset": object_offset,
                "compact_flags": flags,
                "address_provenance": binding["object_offset_evidence"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--template-manifest", required=True, type=Path)
    build.add_argument("--template-binary", required=True, type=Path)
    build.add_argument("--model-id", required=True)
    build.add_argument("--layers", required=True, type=int)
    build.add_argument("--phase", required=True, choices=("prefill", "decode"))
    build.add_argument("--context-tokens", required=True, type=int)
    build.add_argument("--batch", required=True, type=int)
    build.add_argument("--template-layer", default=0, type=int)
    build.add_argument("--kernel-graph-binding", type=Path)
    build.add_argument("--phase-generator-descriptor", type=Path)
    build.add_argument("--activation-policy", choices=ACTIVATION_POLICIES,
                       default="class_only")
    build.add_argument("--output", required=True, type=Path)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--plan", required=True, type=Path)
    sample.add_argument("--template-binary", required=True, type=Path)
    sample.add_argument("--layer", required=True, type=int)
    sample.add_argument("--begin", default=0, type=int)
    sample.add_argument("--count", default=16, type=int)
    sample.add_argument("--include-kind", action="append")
    args = parser.parse_args()

    if args.command == "build":
        result = build_plan(
            template_manifest_path=args.template_manifest,
            template_binary_path=args.template_binary,
            model_id=args.model_id,
            layers=args.layers,
            phase=args.phase,
            context_tokens=args.context_tokens,
            batch=args.batch,
            activation_policy=args.activation_policy,
            template_layer=args.template_layer,
            kernel_graph_binding_path=args.kernel_graph_binding,
            phase_generator_descriptor_path=args.phase_generator_descriptor,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "coverage": result["coverage"]["status"],
                    "requests": result["coverage"]["transformer_layer_requests"],
                    "plan_fingerprint_sha256": result["plan_fingerprint_sha256"],
                },
                sort_keys=True,
            )
        )
        return

    plan = load_json(args.plan)
    result = expand_slice(
        plan=plan,
        template_binary_path=args.template_binary,
        layer=args.layer,
        begin=args.begin,
        count=args.count,
        include_kinds=set(args.include_kind) if args.include_kind else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

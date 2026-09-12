#!/usr/bin/env python3
"""Prepare a tiny SYNTHETIC reference/native example without a GPU or test imports."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from hbserve.traces.common import load_json, new_output, save
from hbserve.traces.prepare import bind_inputs, inspect_inputs, resolve_experiment_paths
from hbserve.traces.reference import generate

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quickstart_source_fixture", ROOT / "examples/trace_fixture.py")
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def prepare_example(output: Path, *, phase: str = "decode", cache_engine: Path | None = None) -> dict:
    root = new_output(output)
    model = load_json(ROOT / "models/llama31-8b-w8-kv-bf16.json")
    model["model"]["name"] = "synthetic_reference_two_layer_fixture"
    model["model"]["source"] = {
        "model_repository": "examples/reference_native_quickstart.py",
        "public_reference": "hand-sized synthetic fixture, not Llama or a hardware capture",
        "config_access": "source checkout or extracted source distribution",
        "dimension_transcription": "2 layers, hidden 128, vocabulary 256, FFN 256",
    }
    model["intended_use"] = {"role": "synthetic_plumbing_test", "paper_claim_eligible_by_itself": False}
    model["architecture"].update({
        "num_layers": 2, "hidden_size": 128, "vocab_size": 256, "num_attention_heads": 4,
        "attention": {"kind": "gqa", "num_key_value_heads": 1, "head_dim": 32, "qk_head_norms": False},
        "ffn": {"dense_intermediate_size": 256},
    })
    save(root / "model.json", model)
    original = ROOT / "configs/windows/miniquick-decode.json"
    experiment = resolve_experiment_paths(load_json(original), original.parent)
    experiment["experiment_id"] = "synthetic-reference-quickstart"
    experiment["purpose"] = "Software contract example only; stage labels do not simulate a real model."
    experiment["population"].update({"model_descriptor": str(root / "model.json"),
                                     "target_population_bytes": 4 * 1024**2,
                                     "runtime_overhead_bytes": 64 * 1024})
    experiment["workload"].update({"decode_context_prior_tokens": 16,
                                   "decode_steps_per_window": 1, "random_read_chunk_blocks": 1})
    experiment["topologies"] = [t for t in experiment["topologies"] if t["id"] in {"all-hbm", "4h4f", "0h8f"}]
    original = root / "native-experiment.json"
    save(original, experiment)
    plan = FIXTURE.create_fixture(root / "source", phase=phase)
    cache_name = "reference-cache64.json" if cache_engine else "reference-cache.json"
    cache_config = load_json(ROOT / "examples" / cache_name)
    save(root / "cache.json", cache_config)
    generated = generate(plan=plan, cache_config=cache_config, output=root / "generated", cache_engine=cache_engine)
    inventory = inspect_inputs(experiment=original, plan=plan)
    save(root / "inventory.json", inventory)
    # This map is only for the known fixture. Real models must supply an
    # independently checked object map; size or object kind is not enough.
    regions = {r["id"]: r for r in inventory["native_regions"]}
    kv = next(r for r in regions.values() if r["placement_class"] == "kv")
    runtime = regions["metadata/runtime"]
    rows = []
    for obj in inventory["source_objects"]:
        if obj["kind"] == "weight":
            region, offset = f"weights/layer/{obj['layer']}", 0
        elif obj["kind"] == "kv_cache":
            region, offset = kv["id"], obj["layer"] * 4096
        else:
            region, offset = runtime["id"], 0
        rows.append({"object_id": obj["object_id"], "region_id": region, "offset_bytes": offset})
    mapping = root / "object-map.json"
    save(mapping, {"object_bindings": rows})
    prepared = bind_inputs(experiment=original, plan=plan, post_cache_root=root / "generated",
                           object_bindings=mapping, max_records=100, output=root / "bound",
                           initial_state="Synthetic test accesses; source GPU cache starts empty and drains at exit; native device state follows the selected topology.")
    result = {"status": "PASS_SYNTHETIC_REFERENCE_QUICKSTART", "experiment": prepared["experiment"],
              "phase_label": phase, "post_cache_counts": generated["cache"]["counts"],
              "hardware_fidelity_validated": False, "simulator_executed": False,
              "warning": "Same tiny synthetic accesses for both phase labels; not real prefill/decode evidence."}
    save(root / "quickstart.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--cache-engine", type=Path, help="optional C++ cache engine; selects explicit 64B fill/writeback")
    arguments = parser.parse_args()
    print(json.dumps(prepare_example(arguments.output_root, phase=arguments.phase,
                                     cache_engine=arguments.cache_engine), indent=2, sort_keys=True))

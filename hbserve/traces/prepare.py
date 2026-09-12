"""Inspect source/layout contracts and prepare an explicitly bound native window.

This tool does not collect GPU data, infer object ownership, resize a source,
or validate hardware fidelity. It only automates reproducible input plumbing.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Sequence

from .common import load_json, new_output, require, save, sha256_file
from .plan_contract import uint, validate_plan
from hbserve.windows.experiment import EXPERIMENT_SCHEMA, build_preflight
from hbserve.windows.population import build_explicit_fixed_population
from hbserve.windows.reference import BINDING_SCHEMA, KIND


def absolute_path(value: Any, base: Path) -> str:
    require(isinstance(value, str) and value, "expected a nonempty artifact path")
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def resolve_experiment_paths(document: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve the native window's input paths before writing it elsewhere."""
    result = deepcopy(document)
    result["population"]["model_descriptor"] = absolute_path(
        result["population"]["model_descriptor"], base)
    thermal = result["thermal"]
    thermal["start_state_overlay"] = absolute_path(thermal["start_state_overlay"], base)
    for point in thermal.get("boundary_temperature_axis", {}).get("points", []):
        if point.get("overlay") is not None:
            point["overlay"] = absolute_path(point["overlay"], base)

    def resolve_profile(mapping: dict[str, Any]) -> None:
        placement = mapping.get("direct_placement", {})
        if "profile" in placement:
            placement["profile"] = absolute_path(placement["profile"], base)

    resolve_profile(result.get("mapping", {}))
    for topology in result["topologies"]:
        topology["system_configs"] = [absolute_path(p, base) for p in topology["system_configs"]]
        resolve_profile(topology.get("mapping", {}))
    return result


def inspect_inputs(*, experiment: Path, plan: Path) -> dict[str, Any]:
    """Read only the native population, not a potentially huge coarse trace."""
    experiment = experiment.resolve()
    document = load_json(experiment)
    require(document.get("schema") == EXPERIMENT_SCHEMA, "unsupported native experiment schema")
    population = document["population"]
    layout, _ = build_explicit_fixed_population(
        model_descriptor_path=Path(absolute_path(population["model_descriptor"], experiment.parent)),
        target_population_bytes=uint(population["target_population_bytes"], "population bytes", minimum=1),
        runtime_overhead_bytes=uint(population["runtime_overhead_bytes"], "runtime overhead", minimum=1),
    )
    source, objects, digest = validate_plan(plan)
    return {
        "status": "INSPECTED_NOT_BOUND_OR_FIDELITY_VALIDATED",
        "experiment_sha256": sha256_file(experiment), "plan_sha256": digest,
        "layout_sha256": layout.digest, "native_model": layout.model_name,
        "native_layers": layout.num_layers, "source_workload": source.get("workload"),
        "source_coverage": source.get("coverage"), "source_objects": objects,
        "native_regions": [region.canonical() for region in layout.regions],
        "object_bindings_to_complete": [
            {"object_id": obj["object_id"], "region_id": None, "offset_bytes": None}
            for obj in objects
        ],
        "warning": "Match real object ownership and layout; matching size alone is not semantic equivalence.",
    }


def bind_inputs(*, experiment: Path, plan: Path, post_cache_root: Path,
                object_bindings: Path, initial_state: str, output: Path,
                max_records: int) -> dict[str, Any]:
    require(isinstance(initial_state, str) and initial_state.strip(), "initial state must be explicit")
    uint(max_records, "max records", minimum=1)
    inventory = inspect_inputs(experiment=experiment, plan=plan)
    mapping = load_json(object_bindings)
    require(set(mapping) == {"object_bindings"}, "object-map file must contain only object_bindings")
    source = post_cache_root.resolve()
    cache = load_json(source / "post-cache.manifest.json")
    document = resolve_experiment_paths(load_json(experiment), experiment.resolve().parent)
    require(all(t.get("integration_mode") != "peer_hbm_hbf" for t in document["topologies"]),
            "reference peer KV migration requires token/block ownership; do not guess or drop the topology")
    binding = {
        "schema": BINDING_SCHEMA, "plan": str(plan.resolve()), "post_cache_root": str(source),
        "plan_sha256": inventory["plan_sha256"], "post_cache_sha256": cache["output_sha256"],
        "layout_sha256": inventory["layout_sha256"], "source_workload": inventory["source_workload"],
        "schedule": "serial-kernels", "object_bindings": mapping["object_bindings"],
        "max_records": max_records,
    }
    root = new_output(output)
    save(root / "binding.json", binding)
    document["experiment_id"] = str(document["experiment_id"]) + "-reference"
    document["purpose"] = "Bound reference source window; use source coverage, not the original coarse workload scope."
    document["workload"] = {
        "kind": KIND, "reference_source": "binding.json", "locality_seed": 0,
        "compute_time": "not_modeled", "initial_state": initial_state,
        "same_trace_for_every_topology": True,
    }
    partial = root / "experiment.partial.json"
    save(partial, document)
    # Native validation checks source bytes/digests, all object mappings, and
    # topology contracts. No simulation is launched. Failures leave diagnostics
    # but never publish experiment.json or a successful preparation receipt.
    checked = build_preflight(partial)
    result = {
        "status": "PASS_REFERENCE_WINDOW_PREPARATION",
        "experiment": str(root / "experiment.json"),
        "source_experiment_sha256": inventory["experiment_sha256"],
        "object_bindings_sha256": sha256_file(object_bindings),
        "plan_sha256": inventory["plan_sha256"], "post_cache_sha256": binding["post_cache_sha256"],
        "layout_sha256": inventory["layout_sha256"], "source_coverage": inventory["source_coverage"],
        "trace_sha256": checked["trace"]["trace_sha256"], "traffic": checked["trace"]["traffic"],
        "source_or_cache_changed": False, "simulator_executed": False,
        "hardware_fidelity_validated": False,
        "portability": "Absolute source/config paths; rerun preparation on another machine with its artifacts.",
    }
    partial.rename(root / "experiment.json")
    save(root / "preparation.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hbserve trace prepare", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect", help="list source objects and native regions; never guess ownership")
    bind = sub.add_parser("bind", help="check an explicit map and write a native reference experiment")
    for command in (inspect, bind):
        command.add_argument("--experiment", type=Path, required=True)
        command.add_argument("--plan", type=Path, required=True)
    bind.add_argument("--post-cache-root", type=Path, required=True)
    bind.add_argument("--object-bindings", type=Path, required=True)
    bind.add_argument("--initial-state", required=True)
    bind.add_argument("--max-records", type=int, required=True)
    bind.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "inspect":
        result = inspect_inputs(experiment=args.experiment, plan=args.plan)
    else:
        result = bind_inputs(experiment=args.experiment, plan=args.plan, post_cache_root=args.post_cache_root,
                             object_bindings=args.object_bindings, initial_state=args.initial_state,
                             max_records=args.max_records, output=args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fixed-window execution behind the single HBServe run command."""

from __future__ import annotations

import argparse
import sys

from hbfsim_client.fixed_footprint_metrics import FixedFootprintMetricsError
from hbfsim_client.provenance import gate_provenance, stamp_result
from hbfsim_client.simulation_session import SimulationSessionError
from hbserve.contracts import HBServeError
from hbserve.io import create_run_directory, load_json_object, write_json_atomic
from hbserve.public_model import PublicModelDescriptorError
from hbserve.windows.experiment import (
    FixedFootprintExperimentError,
    build_preflight,
    run_reference_experiment,
)
from hbserve.windows.fixed_footprint_trace import FixedFootprintTraceError
from hbserve.windows.population import ExplicitPopulationError
from hbserve.windows.remap import RemapError


def run_window(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    topology_ids = None
    if args.topologies is not None:
        topology_ids = tuple(item.strip() for item in args.topologies.split(","))
        if not all(topology_ids) or len(set(topology_ids)) != len(topology_ids):
            parser.error("--topologies must contain non-empty, unique IDs")
    provenance = gate_provenance(
        parser,
        None if args.preflight_only else args.simulator,
        allow_dirty=args.allow_dirty,
        config_paths=(args.experiment,),
    )
    try:
        document = load_json_object(args.experiment, "fixed-window experiment")
        result = (
            build_preflight(args.experiment)
            if args.preflight_only
            else run_reference_experiment(
                experiment_path=args.experiment,
                simulator_path=args.simulator,
                topology_ids=topology_ids,
            )
        )
        preflight = result if args.preflight_only else result["preflight"]
        if preflight["experiment"] != provenance["configs"][0]:
            raise HBServeError("experiment identity changed after pre-run provenance")
        stamp_result(result, provenance)
        output_directory = create_run_directory(args.out)
        write_json_atomic(output_directory / "experiment.json", document)
        result_path = output_directory / "result.json"
        write_json_atomic(result_path, result)
        trace = result["trace"]
        lines = [
            f"hbserve run: fixed_window; experiment {result['experiment_id']}",
            "timing model: memory-only fixed work; "
            "no request admission feedback or TTFT/TPOT",
            f"trace: {trace['trace_sha256']}; phases: {trace['phase_count']}",
        ]
        if args.preflight_only:
            lines.append(
                f"preflight only: {len(preflight['topologies'])} topology contracts; "
                "no physical timing executed"
            )
        else:
            for row in result["reference_topology_results"]:
                final_drain_ms = row["metrics"]["final_drain_time_ns"] / 1e6
                lines.append(
                    f"{row['id']}: final drain {final_drain_ms:.3f} ms"
                )
        lines.append(f"output: {result_path}")
        (output_directory / "headline.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    except (
        OSError,
        HBServeError,
        PublicModelDescriptorError,
        ExplicitPopulationError,
        FixedFootprintTraceError,
        FixedFootprintExperimentError,
        FixedFootprintMetricsError,
        RemapError,
        SimulationSessionError,
    ) as error:
        print(f"hbserve fixed-window run failed: {error}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0

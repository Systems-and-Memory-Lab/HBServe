#!/usr/bin/env python3
"""Run reference generation/replay from an isolated installed wheel."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile

import hbserve
import hbfsim_client
from hbserve.__main__ import main
from hbserve.traces import ROUTES
from hbserve.traces.common import load_json
from hbserve.traces.reference import generate
from hbserve.traces.replay import ReferenceInput, preflight

ROOT = Path(__file__).resolve().parents[1]
CACHE = dict(capacity_bytes=1024, line_bytes=128, sector_bytes=32, associativity=2,
             write_policy="write-back", write_allocate=True, write_miss_fetch=True, final_drain=True)


def check(native_experiment=None):
    for package in (hbserve, hbfsim_client):
        if Path(package.__file__).resolve().is_relative_to(ROOT / package.__name__):
            raise RuntimeError("smoke imported a source checkout instead of an installed wheel")
    assert set(ROUTES) == {"reference"}
    assert importlib.util.find_spec("hbserve.traces.simple") is None
    spec = importlib.util.spec_from_file_location("installed_fixture", ROOT / "examples/trace_fixture.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    engine = os.environ.get("HBFSIM_TRACE_TEST_ENGINE")
    runs = []
    with tempfile.TemporaryDirectory(prefix="installed-reference-") as directory:
        root = Path(directory)
        for stage in ("prefill", "decode"):
            plan = fixture.create_fixture(root / (stage + "-plan"), phase=stage)
            input_root = root / (stage + "-generated")
            result = generate(plan=plan, cache_config=CACHE, output=input_root)
            for placement in ("weight-hbf", "kv-hbf"):
                source = ReferenceInput(input_root, plan_path=plan, policy=placement,
                                        page_bytes=4096, max_phase_records=100)
                assert preflight(source)["memory_requests"] == result["cache"]["counts"]["output_requests"]
                if engine:
                    output = root / (stage + "-" + placement)
                    with redirect_stdout(io.StringIO()):
                        assert main(["trace", "replay", "--plan", str(plan), "--input-root", str(input_root),
                                     "--schedule", "serial-kernels", "--placement", placement,
                                     "--simulator", engine, "--system-config", str(ROOT / "configs/4hbm-4hbf.cfg"),
                                     "--system-config", str(ROOT / "configs/simulation-session-mini.cfg"),
                                     "--output-root", str(output)]) == 0
                    replay = load_json(output / "result.json")
                    assert replay["counts"]["R_bytes"] + replay["counts"]["W_bytes"] == result["cache"]["counts"]["output_bytes"]
                    assert replay["memory_finish_ns"] == replay["session"]["final_measurement"]["completed_frontier_ns"]
                runs.append({"stage": stage, "placement": placement, "native_executed": bool(engine)})
        native_result = "not supplied"
        if native_experiment:
            output = root / "native-window"
            args = ["run", "--experiment", str(native_experiment), "--allow-dirty", "--out", str(output)]
            args += ["--simulator", engine, "--topologies", "all-hbm,0h8f"] if engine else ["--preflight-only"]
            with redirect_stdout(io.StringIO()):
                assert main(args) == 0
            result_file, = output.glob("*/result.json")
            assert load_json(result_file)["trace"]["generator"] == "reference"
            native_result = "executed" if engine else "preflight only"
    return {"status": "PASS_INSTALLED_REFERENCE", "source_root": str(ROOT),
            "installed_hbserve": hbserve.__file__, "cases": runs,
            "native_window": native_result,
            "evidence": "synthetic two-layer stage-labeled fixtures; not hardware or continuous-two-token validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-experiment", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(check(arguments.native_experiment), sort_keys=True))

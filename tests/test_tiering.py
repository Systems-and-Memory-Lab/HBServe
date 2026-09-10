"""Matched HBF tiering through the public fixed-window experiment path."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_windows import tiny_experiment
from hbserve.windows.experiment import (
    build_preflight, execute_topology, load_experiment_context, new_remapper,
)


def tiered_experiment(directory):
    path = tiny_experiment(directory)
    experiment = json.loads(path.read_text())
    base = next(row for row in experiment["topologies"] if row["id"] == "6h2f")
    overlay = directory / "cache.cfg"
    overlay.write_text("hbm-capacity-bytes=65536\n")
    experiment["topologies"] = []
    for policy in ("address_only_lru", "decayed_lfu", "threshold_promotion", "class_aware"):
        row = deepcopy(base)
        row.update({"id": "6h2f-" + policy, "integration_mode": "hbm_fronted_hbf"})
        row["system_configs"].append(str(overlay))
        row["mapping"] = {"hbf_tiering": {
            "policy": policy, "migration_granularity_bytes": 4096,
            "transfer_chunk_bytes": 4096, "read_ahead_window_bytes": 8192,
        }}
        experiment["topologies"].append(row)
    path.write_text(json.dumps(experiment))
    return path


class TieringTests(unittest.TestCase):
    simulator = None

    def test_physical_all_hbm_is_distinct_from_capacity_relaxed_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = tiny_experiment(Path(directory))
            experiment = json.loads(path.read_text())
            anchor = next(row for row in experiment["topologies"] if row["id"] == "all-hbm")
            physical = deepcopy(anchor)
            overlay = Path(directory) / "physical-hbm.cfg"
            overlay.write_text("hbm-capacity-bytes=65536\n")
            physical.update(id="physical-hbm", integration_mode="direct_hbm_hbf",
                            expected_outcome="capacity_oom", expected_outcome_reason="finite HBM capacity")
            physical["system_configs"].append(str(overlay))
            experiment["topologies"] = [anchor, physical]
            path.write_text(json.dumps(experiment))
            rows = build_preflight(path)["topologies"]
            self.assertTrue(rows[0]["population_fits_execution_path"])
            self.assertFalse(rows[1]["population_fits_execution_path"])
            self.assertEqual(rows[1]["capacity_oom_validation"]["failure_kind"], "hbm_population_exceeds_physical_capacity")

    def test_policies_share_one_hbf_backing_and_capacity_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = tiered_experiment(Path(directory))
            context = load_experiment_context(path)
            receipt = build_preflight(path)
            for topology, row in zip(context.topologies, receipt["topologies"]):
                remapper = new_remapper(topology, context_layout=context.layout,
                                        experiment=context.experiment, trace_sha256=context.trace.digest)
                self.assertEqual(remapper.backing_kind, "hbf")
                self.assertEqual(row["initial_placement"]["hbf_payload_bytes"], context.layout.address_space_bytes)
                self.assertEqual(row["initial_placement"]["external_backing_bytes"], 0)
                placement = row["initial_placement"]
                self.assertEqual(placement["hbm_cache_capacity_bytes"] +
                                 placement["hbm_stream_staging_bytes"], 65536)
                self.assertEqual(row["initial_placement"]["policy"], remapper.policy)
                self.assertEqual(bool(remapper._kv_units), remapper.policy == "class_aware")

    def test_all_policies_execute_reads_fills_and_writeback(self):
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        with tempfile.TemporaryDirectory() as directory:
            context = load_experiment_context(tiered_experiment(Path(directory)))
            for topology in context.topologies:
                with self.subTest(policy=topology.id):
                    result = execute_topology(context, topology, simulator_path=self.simulator)
                    self.assertEqual(result["trace_sha256"], context.trace.digest)
                    self.assertGreater(result["metrics"]["final_drain_time_ns"], 0)
                    media = result["metrics"]["media"]
                    self.assertGreater(media["hbf"]["transactions"]["read"]["logical_bytes"], 0)
                    self.assertGreater(media["hbf"]["transactions"]["write"]["logical_bytes"], 0)
                    self.assertGreater(media["hbm"]["transactions"]["write"]["logical_bytes"], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path)
    arguments, remaining = parser.parse_known_args()
    TieringTests.simulator = arguments.simulator
    unittest.main(argv=[sys.argv[0], *remaining])

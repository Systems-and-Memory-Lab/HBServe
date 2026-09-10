"""Static placement conserves addresses and never trains on measurement."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from hbserve.contracts import HBServeError
from hbserve.io import load_json_object, write_json_atomic
from hbserve.windows.experiment import load_experiment_context
from hbserve.windows.memory_trace import canonical_sha256
from hbserve.windows.placement import access_profile, placement_order
from hbserve.windows.remap import DirectAttachedRemapper, RemapError
from test_windows import tiny_experiment


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = tiny_experiment(self.directory)
        self.context = load_experiment_context(self.path)
        self.geometry = self.context.topologies[-1].system_config.hbf_geometry

    def remapper(self, capacity, priority):
        return DirectAttachedRemapper(
            address_space_bytes=25 * 4096, hbm_capacity_bytes=capacity,
            hbf_geometry=self.geometry, hbm_stacks=2, hbf_stacks=8,
            placement_granularity_bytes=8192, hbm_priority_units=priority,
        )

    def test_priority_mapping_is_dense_and_conserves_partial_tail(self):
        for priority in (None, list(reversed(range(13))), list(range(13))):
            remapper = self.remapper(5 * 8192, priority)
            addresses = {"HBM": set(), "HBF_LOGICAL": set()}
            hbf_bytes = 0
            for address in range(0, 25 * 4096, 4096):
                target, physical, size = remapper._physical_segments(address, 4096)[0]
                self.assertEqual(size, 4096)
                self.assertNotIn(physical, addresses[target])
                addresses[target].add(physical)
                hbf_bytes += 4096 if target == "HBF_LOGICAL" else 0
            for selected in addresses.values():
                self.assertEqual(selected, set(range(0, len(selected) * 4096, 4096)))
            self.assertEqual(hbf_bytes, remapper.resident_hbf_bytes(0, 25 * 4096))
            self.assertEqual(hbf_bytes, remapper.hbf_resident_payload_bytes)
            self.assertLessEqual(remapper.hbm_resident_payload_bytes, 5 * 8192)

    def test_capacity_increase_keeps_existing_priority_units(self):
        priority = [12, 1, 10, 3, 8, 5, 6, 7, 4, 9, 2, 11, 0]
        small, large = self.remapper(3 * 8192, priority), self.remapper(7 * 8192, priority)
        selected = lambda remapper: {unit for unit in range(13) if remapper._unit_target_and_rank(unit)[0] == "HBM"}
        self.assertTrue(selected(small) < selected(large))
        with self.assertRaises(RemapError):
            self.remapper(8192, [0] * 13)

    def test_object_policies_use_layout_not_measurement(self):
        layout = self.context.layout
        granularity = 4096
        weights, _ = placement_order(layout, {"policy": "weights_first"}, granularity, "measurement")
        kv, detail = placement_order(layout, {"policy": "kv_first"}, granularity, "measurement")
        weight_unit = next(region.begin // granularity for region in layout.regions if region.placement_class == "immutable_weight")
        kv_unit = next(region.begin // granularity for region in layout.regions if region.placement_class == "kv")
        self.assertLess(weights.index(weight_unit), weights.index(kv_unit))
        self.assertLess(kv.index(kv_unit), kv.index(weight_unit))
        self.assertFalse(detail["measurement_trace_used_for_placement"])

    def test_profile_requires_independent_training_and_matching_layout(self):
        profile = access_profile(self.context, 4096, role="training")
        self.assertEqual(sum(unit["read_bytes"] for unit in profile["units"]), self.context.trace.read_bytes)
        self.assertEqual(sum(unit["write_bytes"] for unit in profile["units"]), self.context.trace.write_bytes)
        self.assertLessEqual(profile["unique_accessed_bytes"], profile["accessed_placement_bytes"])
        path = self.directory / "profile.json"
        write_json_atomic(path, profile)
        config = {"policy": "profiled_hotset", "profile": str(path)}
        training_logical = profile["logical_trace_sha256"]
        with self.assertRaisesRegex(HBServeError, "own training"):
            placement_order(self.context.layout, config, 4096, self.context.trace.digest,
                            logical_trace_sha256=training_logical)
        experiment = load_json_object(self.path, "experiment")
        experiment["workload"]["locality_seed"] += 1
        measured_path = self.directory / "measurement.json"
        write_json_atomic(measured_path, experiment)
        measured = load_experiment_context(measured_path)
        measured_logical = canonical_sha256([phase.trace_group.digest for phase in measured.trace.phases])
        order, detail = placement_order(measured.layout, config, 4096, measured.trace.digest,
                                        logical_trace_sha256=measured_logical)
        written = {unit["unit"] for unit in profile["units"] if unit["write_bytes"]}
        self.assertEqual(set(order[:len(written)]), written)
        self.assertFalse(detail["training_cost_included_in_measurement"])
        with self.assertRaisesRegex(HBServeError, "granularity"):
            placement_order(measured.layout, config, 8192, measured.trace.digest,
                            logical_trace_sha256=measured_logical)
        profile["role"] = "measurement"
        wrong_profile = self.directory / "wrong-profile.json"
        write_json_atomic(wrong_profile, profile)
        config["profile"] = str(wrong_profile)
        with self.assertRaisesRegex(HBServeError, "role"):
            placement_order(measured.layout, config, 4096, measured.trace.digest,
                            logical_trace_sha256=measured_logical)

    def test_profile_rejects_the_same_logical_trace_with_changed_prose(self):
        profile = access_profile(self.context, 4096, role="training")
        profile_path = self.directory / "training-profile.json"
        write_json_atomic(profile_path, profile)
        experiment = load_json_object(self.path, "experiment")
        experiment["workload"]["context_length_basis"] = "same work with a changed description"
        measurement_path = self.directory / "measurement.json"
        write_json_atomic(measurement_path, experiment)
        measured = load_experiment_context(measurement_path)
        measured_logical = canonical_sha256([phase.trace_group.digest for phase in measured.trace.phases])
        self.assertNotEqual(measured.trace.digest, self.context.trace.digest)
        self.assertEqual(measured_logical, profile["logical_trace_sha256"])
        config = {"policy": "profiled_hotset", "profile": str(profile_path)}
        with self.assertRaisesRegex(HBServeError, "own training"):
            placement_order(measured.layout, config, 4096, measured.trace.digest,
                            logical_trace_sha256=measured_logical)
        with self.assertRaisesRegex(HBServeError, "logical trace digest"):
            placement_order(measured.layout, config, 4096, measured.trace.digest)


if __name__ == "__main__":
    unittest.main()

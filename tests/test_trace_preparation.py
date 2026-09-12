"""Preparation contracts; no GPU or real-model fidelity claims."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hbserve.__main__ import main
from hbserve.traces.common import load_json, save, sha256_file
from hbserve.traces.prepare import bind_inputs, inspect_inputs, resolve_experiment_paths
from hbserve.windows.experiment import build_preflight
from test_trace_native_window import fixture

ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def source(self, root):
        _, _, binding = fixture(root)
        mapping = root / "explicit-map.json"
        save(mapping, {"object_bindings": binding["object_bindings"]})
        return dict(experiment=root / "experiment.json", plan=root / "source/plan.json",
                    post_cache_root=root / "generated", object_bindings=mapping,
                    initial_state="synthetic explicit cache/native device state", max_records=100,
                    output=root / "prepared")

    def test_inspect_does_not_generate_coarse_requests_or_guess_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.source(Path(directory))
            with patch("hbserve.windows.experiment.build_fixed_footprint_trace", side_effect=AssertionError("unneeded generation")):
                result = inspect_inputs(experiment=kwargs["experiment"], plan=kwargs["plan"])
            self.assertTrue(all(row["region_id"] is None for row in result["object_bindings_to_complete"]))
            self.assertEqual(result["source_workload"]["context_tokens"], 4)

    def test_binding_roundtrip_keeps_source_and_original_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.source(Path(directory))
            paths = [kwargs["experiment"], kwargs["plan"], kwargs["post_cache_root"] / "post-cache.bin"]
            before = {p: sha256_file(p) for p in paths}
            result = bind_inputs(**kwargs)
            self.assertEqual(result["status"], "PASS_REFERENCE_WINDOW_PREPARATION")
            self.assertFalse(result["hardware_fidelity_validated"])
            checked = build_preflight(Path(result["experiment"]))
            self.assertEqual(checked["trace"]["trace_sha256"], result["trace_sha256"])
            self.assertEqual(before, {p: sha256_file(p) for p in paths})
            self.assertFalse((kwargs["output"] / "experiment.partial.json").exists())
            with self.assertRaises(FileExistsError):
                bind_inputs(**kwargs)

    def test_bad_maps_and_record_limit_never_publish_success(self):
        for error in ("missing-object", "record-limit"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                kwargs = self.source(Path(directory))
                if error == "record-limit":
                    kwargs["max_records"] = 1
                else:
                    mapping = load_json(kwargs["object_bindings"])
                    mapping["object_bindings"].pop()
                    alternate = Path(directory) / "bad-map.json"
                    save(alternate, mapping)
                    kwargs["object_bindings"] = alternate
                with self.assertRaises(ValueError):
                    bind_inputs(**kwargs)
                self.assertFalse((kwargs["output"] / "experiment.json").exists())
                self.assertFalse((kwargs["output"] / "preparation.json").exists())

    def test_path_relocation_including_placement_profiles_is_explicit(self):
        original = {"population": {"model_descriptor": "model.json"},
                    "thermal": {"start_state_overlay": "cold.cfg", "boundary_temperature_axis": {
                        "points": [{"overlay": None}, {"overlay": "warm.cfg"}]}},
                    "mapping": {"direct_placement": {"profile": "trained.json"}},
                    "topologies": [{"system_configs": ["device.cfg"], "mapping": {
                        "direct_placement": {"profile": "other.json"}}}]}
        before = deepcopy(original)
        resolved = resolve_experiment_paths(original, Path("/data/source"))
        self.assertEqual(original, before)
        self.assertEqual(resolved["population"]["model_descriptor"], "/data/source/model.json")
        self.assertEqual(resolved["mapping"]["direct_placement"]["profile"], "/data/source/trained.json")
        self.assertEqual(resolved["topologies"][0]["mapping"]["direct_placement"]["profile"], "/data/source/other.json")
        self.assertIsNone(resolved["thermal"]["boundary_temperature_axis"]["points"][0]["overlay"])

    def test_public_prepare_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.source(Path(directory))
            common = ["--experiment", str(kwargs["experiment"]), "--plan", str(kwargs["plan"])]
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["trace", "prepare", "inspect", *common]), 0)
            self.assertIn("native_regions", json.loads(out.getvalue()))
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["trace", "prepare", "bind", *common,
                    "--post-cache-root", str(kwargs["post_cache_root"]),
                    "--object-bindings", str(kwargs["object_bindings"]),
                    "--initial-state", kwargs["initial_state"], "--max-records", "100",
                    "--output-root", str(kwargs["output"])]), 0)
            self.assertEqual(json.loads(out.getvalue())["status"], "PASS_REFERENCE_WINDOW_PREPARATION")

    def test_self_contained_quickstart_for_both_stage_labels(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                command = [sys.executable, "-B", str(ROOT / "examples/reference_native_quickstart.py"),
                           "--phase", phase, "--output-root", str(Path(directory) / "demo")]
                executed = subprocess.run(command, cwd=directory, timeout=30, capture_output=True, text=True,
                    env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
                self.assertEqual(executed.returncode, 0, executed.stderr)
                result = json.loads(executed.stdout)
                self.assertFalse(result["hardware_fidelity_validated"])
                self.assertGreater(result["post_cache_counts"]["output_w_bytes"], 0)
                checked = build_preflight(Path(result["experiment"]))
                self.assertEqual(checked["trace"]["source"]["workload"]["phase"], phase)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Standalone workload migration and fixed-window execution regressions."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbserve.__main__ import main
from hbserve.contracts import HBServeError
from hbserve.io import create_run_directory, load_json_object, write_json_atomic
from hbserve.public_model import (
    PublicModelDescriptorError,
    derive_public_model_capacity_inputs,
    derive_public_model_ledger,
)
from hbserve.windows.experiment import load_experiment_context


def logical_trace_digest(trace) -> str:
    digest = hashlib.sha256()
    for phase in trace.phases:
        document = {
            "phase": phase.id,
            "stage": phase.stage,
            "layer": phase.layer,
            "transactions": [asdict(transaction) for transaction in phase.trace_group.transactions],
        }
        digest.update(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def tiny_experiment(directory: Path) -> Path:
    descriptor = load_json_object(ROOT / "models/llama31-8b-w8-kv-bf16.json", "template")
    descriptor["model"]["name"] = "ci_tiny_window"
    descriptor["model"]["source"] = {
        "model_repository": "tests/test_windows.py",
        "public_reference": "hand-sized fixture, not a public model",
        "config_access": "in-test fixture",
        "dimension_transcription": "2 layers, hidden 128, vocab 256, FFN 256",
    }
    descriptor["architecture"].update({
        "num_layers": 2,
        "hidden_size": 128,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "attention": {
            "kind": "gqa", "num_key_value_heads": 1,
            "head_dim": 32, "qk_head_norms": False,
        },
        "ffn": {"dense_intermediate_size": 256},
    })
    write_json_atomic(directory / "model.json", descriptor)
    template = ROOT / "configs/windows/miniquick-decode.json"

    def resolve_profiles(value):
        if isinstance(value, dict):
            return {key: resolve_profiles(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve_profiles(item) for item in value]
        if isinstance(value, str) and value.endswith(".cfg"):
            return str((template.parent / value).resolve())
        return value

    experiment = resolve_profiles(load_json_object(template, "experiment"))
    experiment["experiment_id"] = "ci_tiny_window"
    experiment["population"].update({
        "model_descriptor": "model.json",
        "target_population_bytes": 4 * 1024**2,
        "runtime_overhead_bytes": 64 * 1024,
    })
    experiment["workload"].update({
        "decode_context_prior_tokens": 16,
        "decode_steps_per_window": 1,
        "random_read_chunk_blocks": 1,
        "context_length_basis": "one-block CI fixture",
        "initial_state": "four one-block contexts installed before timing",
    })
    experiment_path = directory / "experiment.json"
    write_json_atomic(experiment_path, experiment)
    return experiment_path


class WindowTests(unittest.TestCase):
    def test_migrated_windows_preserve_source_transactions(self) -> None:
        """Freeze logical transactions from HBFSim 70c3fd5, not source-file hashes."""

        references = {
            "miniquick-serving": (
                9565, 3649608300312, 26868469760,
                "61ee3e95009ad7d8a2804bc0a0e4cf675e5be9c754acc4b5e31dccc7854c12d3",
            ),
            "miniquick-decode": (
                653, 385454378144, 5242880,
                "e4e5cbb6409c581722c585834b327832eba0831c70db056ee2ab4f5bfa5635fa",
            ),
            "full-scale-serving": (
                11415, 53736474428180, 294751049728,
                "9580c1f1618824c45a5853731b1a7f4991720ba0a33bb4515767d2e12cd67ee1",
            ),
        }
        for name, expected in references.items():
            with self.subTest(window=name):
                context = load_experiment_context(ROOT / "configs/windows" / f"{name}.json")
                trace = context.trace
                actual = (len(trace.phases), trace.read_bytes, trace.write_bytes, logical_trace_digest(trace))
                self.assertEqual(actual, expected)
                self.assertEqual(len(context.topologies), 8 if name == "miniquick-serving" else 7)

    def test_70b_ledger_preserves_per_channel_scales_and_matrix_embeddings(self) -> None:
        ledger = derive_public_model_capacity_inputs(ROOT / "models/llama31-70b-w8a16-kv-bf16.json")
        self.assertEqual(ledger["immutable_weight_backing_bytes"], 70568973312)
        self.assertEqual(ledger["components"]["embedding_bytes"], 128256 * (8192 + 2))
        self.assertEqual(ledger["kv_page_bytes_per_layer"], 16 * 2 * 8 * 128 * 2)
        attention_parameters = 2 * 8192**2 + 2 * 8192 * 1024
        self.assertEqual(
            ledger["components"]["attention_bytes_per_layer"],
            attention_parameters + 2 * (8192 + 1024 + 1024 + 8192),
        )

    def test_precision_contract_does_not_silently_substitute_a_scheme(self) -> None:
        descriptor = load_json_object(ROOT / "models/llama31-70b-w8a16-kv-bf16.json", "model")
        for field, value in (
            ("quantization_scheme", "unmodeled"),
            ("quantization_scheme", "none"),
            ("scale_bytes", 0),
            ("embedding_storage", "unmodeled"),
            ("zero_point_bytes", 2),
        ):
            with self.subTest(field=field, value=value):
                changed = json.loads(json.dumps(descriptor))
                changed["precision"][field] = value
                with self.assertRaises(PublicModelDescriptorError):
                    derive_public_model_ledger(changed, descriptor_artifact={"source": "test"})

    def test_mode_specific_options_are_not_silently_ignored(self) -> None:
        cases = (
            ["--requests", "requests.json", "--experiment", "window.json"],
            ["--experiment", "window.json", "--timing", "roofline"],
            ["--experiment", "window.json", "--preflight-only", "--topologies", "all-hbm"],
            ["--requests", "requests.json", "--allow-dirty"],
            ["--requests", "requests.json"],
            ["--experiment", "window.json"],
            ["--requests", "requests.json", "--model", "model.json", "--system", "system.cfg",
             "--placement", "all-hbm", "--run-config", "run.json", "--efficiency", "0.5"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    main(["run", *arguments, "--out", "unused"])
                self.assertEqual(failure.exception.code, 2)

    def test_output_directories_are_unique_and_json_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = create_run_directory(Path(directory))
            second = create_run_directory(Path(directory))
            self.assertNotEqual(first, second)
            destination = first / "result.json"
            write_json_atomic(destination, {"run": 1})
            with self.assertRaises(HBServeError):
                write_json_atomic(destination, {"run": 2})
            self.assertEqual(load_json_object(destination, "result"), {"run": 1})

    def test_preflight_is_standalone_and_resolves_paths_from_the_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            experiment = tiny_experiment(root)
            self.assertEqual(main([
                "run", "--experiment", str(experiment), "--preflight-only",
                "--allow-dirty", "--out", str(root / "out"),
            ]), 0)
            result_path, = (root / "out").glob("*/result.json")
            result = load_json_object(result_path, "preflight")
            self.assertEqual(result["execution_mode"], "fixed_window")
            self.assertEqual(result["invariants"]["window_shape"], "decode_only_step")
            self.assertIsNone(result["source"]["simulator"])
            self.assertNotIn("reference_topology_results", result)


class PhysicalWindowTests(unittest.TestCase):
    simulator: Path | None = None

    def test_selected_topology_executes_the_same_trace(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            experiment = tiny_experiment(root)
            self.assertEqual(main([
                "run", "--experiment", str(experiment), "--simulator", str(self.simulator),
                "--topologies", "all-hbm,0h8f,8h0f-dram", "--allow-dirty", "--out", str(root / "out"),
            ]), 0)
            result_path, = (root / "out").glob("*/result.json")
            result = load_json_object(result_path, "run")
            self.assertEqual(result["execution_mode"], "fixed_window")
            rows = result["reference_topology_results"]
            self.assertEqual([row["id"] for row in rows], ["all-hbm", "0h8f", "8h0f-dram"])
            for row in rows:
                self.assertEqual(row["trace_sha256"], result["trace"]["trace_sha256"])
                self.assertGreater(row["metrics"]["final_drain_time_ns"], 0)
            self.assertNotIn("hbserve", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path)
    arguments, remaining = parser.parse_known_args()
    PhysicalWindowTests.simulator = arguments.simulator
    unittest.main(argv=[sys.argv[0], *remaining])

"""Reference/native interface regressions. All traces here are synthetic."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from hbserve.__main__ import main
from hbserve.traces.common import load_json, save, sha256_file
from hbserve.traces.reference import generate
from hbserve.traces._reference.compact_request_template import RECORD_STRUCT
from hbserve.windows.experiment import load_experiment_context, build_preflight, new_remapper
from hbserve.windows.placement import access_profile
from hbserve.windows.reference import KIND, BINDING_SCHEMA
from hbserve.windows.window_contract import FixedFootprintTraceError

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


WINDOW = module("window_fixture", ROOT / "tests/test_windows.py")
FIXTURE = module("reference_fixture", ROOT / "examples/trace_fixture.py")
CACHE = dict(capacity_bytes=1024, line_bytes=128, sector_bytes=32, associativity=2,
             write_policy="write-back", write_allocate=True, write_miss_fetch=True, final_drain=True)


def fixture(root, phase="decode"):
    original = WINDOW.tiny_experiment(root)
    context = load_experiment_context(original)
    plan_path = FIXTURE.create_fixture(root / "source", phase=phase)
    plan = load_json(plan_path)
    generated = generate(plan=plan_path, cache_config=CACHE, output=root / "generated")
    bindings = []
    for obj in plan["objects"]:
        if obj["kind"] == "weight":
            region = next(r for r in context.layout.regions if r.placement_class == "immutable_weight"
                          and r.group == obj["layer"] and r.bytes >= obj["bytes"])
            offset = 0
        elif obj["kind"] == "kv_cache":
            region = context.layout.region(context.layout.kv_region_id)
            span = obj["layer"] * context.layout.num_logical_kv_blocks * context.layout.kv_page_bytes_per_layer
            offset = (span + 4095) // 4096 * 4096
        else:
            region = context.layout.region(context.layout.metadata_region_id)
            offset = 0
        bindings.append({"object_id": obj["object_id"], "region_id": region.id, "offset_bytes": offset})
    binding = dict(schema=BINDING_SCHEMA, plan="source/plan.json", post_cache_root="generated",
                   plan_sha256=sha256_file(plan_path), post_cache_sha256=generated["cache"]["output_sha256"],
                   layout_sha256=context.layout.digest, source_workload=plan["workload"],
                   schedule="serial-kernels", object_bindings=bindings, max_records=100)
    save(root / "binding.json", binding)
    document = load_json(original)
    document["workload"] = dict(kind=KIND, reference_source="binding.json", locality_seed=0,
                               compute_time="not_modeled", same_trace_for_every_topology=True,
                               initial_state="synthetic fixture; source GPU-cache state preserved; native device initial state")
    output = root / "reference-experiment.json"
    save(output, document)
    return output, generated, binding


class NativeReferenceTests(unittest.TestCase):
    def test_prefill_and_decode_keep_exact_requests_and_join_every_kernel(self):
        for stage in ("prefill", "decode"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, generated, _ = fixture(root, stage)
                context = load_experiment_context(path)
                trace = context.trace
                raw = list(RECORD_STRUCT.iter_unpack((root / "generated/post-cache.bin").read_bytes()))
                requests = [tx for p in trace.phases for tx in p.trace_group.memory_transactions]
                self.assertEqual(len(requests), len(raw))
                self.assertEqual([tx.bytes for tx in requests], [r[3] for r in raw])
                self.assertEqual([tx.op for tx in requests], ["R" if r[4] == 0 else "W" for r in raw])
                self.assertEqual([tx.addr % 4096 for tx in requests], [r[1] % 4096 for r in raw])
                self.assertEqual(trace.read_bytes + trace.write_bytes, generated["cache"]["counts"]["output_bytes"])
                for phase in trace.phases:
                    self.assertEqual(phase.stage, stage)
                    batch = phase.trace_group
                    self.assertEqual(set(batch.transactions[-1].dependencies), {tx.id for tx in batch.memory_transactions})
                profile = access_profile(context, 4096, role="measurement")
                self.assertEqual(profile["read_bytes"], trace.read_bytes)
                self.assertEqual(profile["write_bytes"], trace.write_bytes)
                self.assertEqual(trace.receipt()["source"]["coverage"], load_json(root / "source/plan.json")["coverage"])

    def test_native_preflight_and_cli_consume_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = fixture(root)
            result = build_preflight(path)
            self.assertEqual(result["trace"]["generator"], "reference")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", "--experiment", str(path), "--preflight-only",
                                       "--allow-dirty", "--out", str(root / "out")]), 0)
            result_file, = (root / "out").glob("*/result.json")
            self.assertEqual(load_json(result_file)["trace"]["generator"], "reference")

    def test_malformed_binding_or_conflicting_workload_fails_closed(self):
        for mutation, message in (
            (lambda b: b.update(layout_sha256="0" * 64), "different native layout"),
            (lambda b: b.update(plan_sha256="0" * 64), "plan digest"),
            (lambda b: b.update(post_cache_sha256="0" * 64), "post-cache digest"),
            (lambda b: b.update(source_workload={}), "scope mismatch"),
            (lambda b: b.update(max_records=1), "max_records"),
            (lambda b: b.update(object_bindings=b["object_bindings"][:-1]), "every source object"),
            (lambda b: b["object_bindings"][0].update(offset_bytes=1), "within-page"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, _, binding = fixture(root)
                mutation(binding)
                (root / "binding.json").write_text(json.dumps(binding))
                with self.assertRaisesRegex(FixedFootprintTraceError, message):
                    load_experiment_context(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = fixture(root)
            document = load_json(path)
            document["workload"]["decode_steps_per_window"] = 10
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(FixedFootprintTraceError, "not silently ignored"):
                load_experiment_context(path)

    def test_simple_not_registered_or_packaged(self):
        from hbserve.traces import ROUTES
        self.assertEqual(set(ROUTES), {"reference"})
        for name in ("simple.py", "activations.py", "_baseline", "_client"):
            self.assertFalse((ROOT / "hbserve/traces" / name).exists())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["trace", "simple"]), 2)


@unittest.skipUnless(os.environ.get("HBFSIM_TRACE_TEST_ENGINE"), "optional native HBFSim")
class NativeReferenceExecutionTests(unittest.TestCase):
    def test_native_mixed_placement_policies_are_not_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = fixture(root)
            document = load_json(path)
            mixed = next(row for row in document["topologies"] if row["id"] == "4h4f")
            overlay = root / "small-hbm.cfg"
            overlay.write_text("hbm-capacity-bytes=131072\n")
            document["mapping"]["placement_granularity_bytes"] = 4096
            document["topologies"] = []
            for policy in ("weights_first", "kv_first"):
                row = deepcopy(mixed)
                row["id"] = policy
                row["system_configs"].append(str(overlay))
                row["mapping"] = {"direct_placement": {"policy": policy}}
                document["topologies"].append(row)
            path.write_text(json.dumps(document))
            context = load_experiment_context(path, simulator_path=Path(os.environ["HBFSIM_TRACE_TEST_ENGINE"]))
            targets = {}
            for topology in context.topologies:
                mapper = new_remapper(topology, context_layout=context.layout,
                                      experiment=context.experiment, trace_sha256=context.trace.digest)
                weight_targets = []
                for phase in context.trace.phases:
                    mapped = mapper.remap(phase.trace_group)
                    weight_targets.extend(tx.target for tx in mapped.transactions
                                          if tx.op == "R" and not tx.id.endswith("complete"))
                targets[topology.id] = weight_targets
            self.assertNotEqual(targets["weights_first"], targets["kv_first"])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", "--experiment", str(path), "--simulator",
                                       os.environ["HBFSIM_TRACE_TEST_ENGINE"], "--allow-dirty",
                                       "--out", str(root / "mixed-out")]), 0)
            result_file, = (root / "mixed-out").glob("*/result.json")
            result = load_json(result_file)
            self.assertEqual([r["id"] for r in result["reference_topology_results"]], ["weights_first", "kv_first"])
            self.assertEqual(len({r["trace_sha256"] for r in result["reference_topology_results"]}), 1)

    def test_actual_native_run_prefill_decode_and_three_topologies(self):
        for stage in ("prefill", "decode"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, generated, _ = fixture(root, stage)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["run", "--experiment", str(path), "--simulator",
                                           os.environ["HBFSIM_TRACE_TEST_ENGINE"], "--topologies",
                                           "all-hbm,0h8f,8h0f-dram", "--allow-dirty", "--out", str(root / "out")]), 0)
                result_file, = (root / "out").glob("*/result.json")
                result = load_json(result_file)
                self.assertEqual(result["trace"]["traffic"]["bytes"], generated["cache"]["counts"]["output_bytes"])
                for row in result["reference_topology_results"]:
                    self.assertEqual(row["trace_sha256"], result["trace"]["trace_sha256"])
                    self.assertGreater(row["metrics"]["final_drain_time_ns"], 0)
                    final = row["simulation_session"]["final_measurement"]
                    self.assertGreaterEqual(final["drained_frontier_ns"], final["completed_frontier_ns"])


if __name__ == "__main__":
    unittest.main()

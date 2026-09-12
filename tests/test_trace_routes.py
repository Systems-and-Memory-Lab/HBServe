"""Publication contracts: explicit routes, streaming, failure safety and parity."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hbserve.__main__ import main as cli
from hbserve.traces import ROUTES
from hbserve.traces.common import load_json, sha256_file
from hbserve.traces.reference import generate as reference_generate
from hbserve.traces._reference.compact_request_template import RECORD_STRUCT
from hbserve.traces._reference.stream_lazy_trace_plan import stream_plan
from hbserve.traces._reference.cache_compact_trace import transform_compact

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trace_fixture", ROOT / "examples/trace_fixture.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
CACHE = dict(capacity_bytes=1024, line_bytes=128, sector_bytes=32,
             associativity=2, write_policy="write-back", write_allocate=True,
             write_miss_fetch=True, final_drain=True)


class TraceRouteTests(unittest.TestCase):
    def test_source_receipt_matches_packaged_files(self):
        receipt = load_json(ROOT / "hbserve/traces/SOURCES.json")
        self.assertFalse(receipt["native_simulator_source_changed"])
        for row in receipt["files"]:
            self.assertEqual(sha256_file(ROOT / row["path"]), row["packaged_sha256"], row["path"])

    def test_explicit_routes_only_and_existing_commands(self):
        self.assertEqual(set(ROUTES), {"reference"})
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for args in (["trace"], ["trace", "list"], ["capabilities"], ["--version"]):
                self.assertEqual(cli(args), 0)
            self.assertEqual(cli(["trace", "compact"]), 2)
            self.assertEqual(cli(["trace", "list", "--unknown"]), 2)

    def test_pipeline_matches_separate_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = fixture.create_fixture(root / "input")
            producer = stream_plan(plan_path=plan, output_path=root / "issued.bin",
                                   output_manifest_path=root / "issued.json", backend="python")
            separate = transform_compact(plan_path=plan, input_path=root / "issued.bin",
                                         input_stream=None, output_path=root / "separate.bin",
                                         output_stream=None, manifest_path=root / "separate.json",
                                         cache_mode="reference-lru", **CACHE)
            joined = reference_generate(plan=plan, cache_config=CACHE, output=root / "joined", chunk_records=1)
            self.assertEqual((root / "joined/post-cache.bin").read_bytes(), (root / "separate.bin").read_bytes())
            self.assertEqual(joined["cache"]["counts"], separate["counts"])
            self.assertEqual(joined["cache"]["input_sha256"], producer["output_sha256"])
            self.assertEqual(joined["cache"]["cache"]["final_state"], separate["cache"]["final_state"])
            self.assertFalse(joined["precache_flat_file_written"])
            self.assertFalse((root / "joined/issued.bin").exists())
            self.assertGreater(joined["cache"]["counts"]["output_w_bytes"], 0)

    def test_stdout_is_binary_only_and_matches_saved_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = fixture.create_fixture(root / "input")
            saved = reference_generate(plan=plan, cache_config=CACHE, output=root / "saved")
            config = root / "config.json"
            config.write_text(json.dumps(CACHE))
            run = subprocess.run([sys.executable, "-B", "-m", "hbserve", "trace", "reference",
                                  "--plan", str(plan), "--cache-config", str(config),
                                  "--output-root", str(root / "stream"), "--stdout"],
                                 capture_output=True, timeout=20, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout, (root / "saved/post-cache.bin").read_bytes())
            self.assertEqual(load_json(root / "stream/post-cache.manifest.json")["output_sha256"],
                             saved["cache"]["output_sha256"])
            self.assertEqual(json.loads(run.stderr)["generator"], "reference")

    def test_producer_failure_never_publishes_final_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = fixture.create_fixture(root / "input")
            (root / "input/template.bin").write_bytes(b"corrupted")
            with self.assertRaises((RuntimeError, ValueError)):
                reference_generate(plan=plan, cache_config=CACHE, output=root / "failed")
            self.assertFalse((root / "failed/post-cache.bin").exists())
            self.assertFalse((root / "failed/post-cache.manifest.json").exists())
            self.assertFalse((root / "failed/result.json").exists())

    def test_validation_and_existing_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = fixture.create_fixture(root / "input")
            for label, config in (("unknown", {**CACHE, "unknown": 1}),
                                  ("geometry", {**CACHE, "capacity_bytes": 1000}),
                                  ("bool", {**CACHE, "sector_bytes": True}),
                                  ("native64", {**CACHE, "read_fill_bytes": 64})):
                with self.subTest(label=label), self.assertRaises(ValueError):
                    reference_generate(plan=plan, cache_config=config, output=root / label)
                self.assertFalse((root / label).exists())
            with self.assertRaises(FileExistsError):
                reference_generate(plan=plan, cache_config=CACHE, output=root)

    @unittest.skipUnless(os.environ.get("HBF_FAST_CACHE_ENGINE"), "optional C++ cache engine")
    def test_native_sector_parity_and_64b_fill_writeback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = fixture.create_fixture(root / "input")
            engine = Path(os.environ["HBF_FAST_CACHE_ENGINE"]).resolve()
            py = reference_generate(plan=plan, cache_config=CACHE, output=root / "python")
            native = reference_generate(plan=plan, cache_config=CACHE, output=root / "native", cache_engine=engine)
            for key in ("output_sha256", "input_sha256", "counts", "cache"):
                self.assertEqual(py["cache"][key], native["cache"][key], key)
            config = {**CACHE, "read_fill_bytes": 64, "writeback_bytes": 64}
            grouped = reference_generate(plan=plan, cache_config=config, output=root / "grouped", cache_engine=engine)
            rows = list(RECORD_STRUCT.iter_unpack((root / "grouped/post-cache.bin").read_bytes()))
            self.assertTrue(rows)
            self.assertTrue(all(row[3] == 64 and row[1] % 64 == 0 for row in rows))
            self.assertTrue(any(row[4] == 1 for row in rows))
            self.assertEqual(grouped["cache"]["read_fill"]["bytes"], 64)
            self.assertEqual(grouped["cache"]["writeback"]["bytes"], 64)


if __name__ == "__main__":
    unittest.main()

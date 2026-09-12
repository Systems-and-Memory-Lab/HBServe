"""Portable template export preserves source bytes and fails closed."""

from __future__ import annotations

from contextlib import redirect_stdout
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from hbserve.__main__ import main as cli
from hbserve.traces.artifacts import artifact_context, resolve_input
from hbserve.traces.catalog import export_entry, list_entries
from hbserve.traces.common import load_json
from hbserve.traces.reference import generate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("catalog_fixture", ROOT / "examples/trace_fixture.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
CACHE = dict(capacity_bytes=1024, line_bytes=128, sector_bytes=32,
             associativity=2, write_policy="write-back", write_allocate=True,
             write_miss_fetch=True, final_drain=True)


def make_catalog(root: Path) -> tuple[Path, Path]:
    plan = fixture.create_fixture(root / "original")
    directory = root / "catalog"
    (directory / "blobs").mkdir(parents=True)
    artifacts = {}
    for path in sorted(plan.parent.iterdir()):
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        compressed = gzip.compress(body, mtime=0)
        blob = "blobs/" + digest + ".gz"
        (directory / blob).write_bytes(compressed)
        artifacts[str(path)] = dict(sha256=digest, bytes=len(body), blob=blob,
                                   compressed_bytes=len(compressed),
                                   encoding="json" if path.suffix == ".json" else "binary")
    document = {
        "schema": {"name": "hbserve.reference_source_catalog", "version": 1},
        "artifacts": artifacts,
        "recipes": [{"id": "fixture", "label": "synthetic only", "source": str(plan),
                     "artifacts": list(artifacts), "missing": [], "status": "test_fixture"}],
        "templates": [{"id": "body", "label": "synthetic template",
                       "source": str(plan.parent / "template.json"),
                       "binary_source": str(plan.parent / "template.bin"), "status": "test_fixture"}],
    }
    path = directory / "catalog.json"
    path.write_text(json.dumps(document))
    return path, plan


class CatalogTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "reference_templates/catalog.json").is_file(), "source dataset is optional in a wheel")
    def test_retained_plan_prepares_and_streams_after_relocation(self):
        from hbserve.traces._reference.stream_lazy_trace_plan import stream_plan
        class BoundedSampleComplete(Exception):
            pass
        class FirstChunk(io.BytesIO):
            def write(self, data):
                super().write(data)
                raise BoundedSampleComplete()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = export_entry(catalog=ROOT / "reference_templates/catalog.json",
                                  entry_id="plan-808a4b07377aee27", output=root / "source")
            sample = FirstChunk()
            with artifact_context(root / "source/artifact-map.json"):
                with self.assertRaises(BoundedSampleComplete):
                    stream_plan(plan_path=root / "source" / result["entrypoint"], output_path=None,
                                output_stream=sample, output_manifest_path=root / "not-complete.json",
                                backend="python", chunk_records=32)
            self.assertEqual(len(sample.getvalue()), 32 * 12)
            self.assertFalse((root / "not-complete.json").exists())

    def test_export_relocation_and_producer_thread_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, plan = make_catalog(root)
            baseline = generate(plan=plan, cache_config=CACHE, output=root / "baseline")
            result = export_entry(catalog=catalog, entry_id="fixture", output=root / "export")
            (root / "original").rename(root / "original-unavailable")
            (root / "export").rename(root / "relocated")
            moved = root / "relocated"
            regenerated = generate(plan=moved / result["entrypoint"], cache_config=CACHE,
                                   artifact_map=moved / "artifact-map.json", output=root / "regenerated")
            self.assertEqual(baseline["producer"]["output_sha256"], regenerated["producer"]["output_sha256"])
            self.assertEqual(baseline["cache"]["output_sha256"], regenerated["cache"]["output_sha256"])
            self.assertEqual(baseline["cache"]["counts"], regenerated["cache"]["counts"])
            self.assertFalse(regenerated["source_resolution"]["source_bytes_rewritten"])
            self.assertEqual((moved / result["entrypoint"]).read_bytes(),
                             (root / "original-unavailable/plan.json").read_bytes())

    def test_missing_binding_does_not_fall_back_to_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, plan = make_catalog(root)
            result = export_entry(catalog=catalog, entry_id="fixture", output=root / "export")
            map_path = root / "export/artifact-map.json"
            document = load_json(map_path)
            document["artifacts"].pop(str(plan.parent / "template.bin"))
            map_path.write_text(json.dumps(document))
            with artifact_context(map_path):
                with self.assertRaisesRegex(ValueError, "not in the explicit"):
                    resolve_input(str(plan.parent / "template.bin"))
            with self.assertRaises(RuntimeError):
                generate(plan=root / "export" / result["entrypoint"], cache_config=CACHE,
                         artifact_map=map_path, output=root / "failed")
            self.assertFalse((root / "failed/result.json").exists())
            self.assertFalse((root / "failed/post-cache.manifest.json").exists())
            self.assertEqual(resolve_input(plan), plan.resolve())

    def test_corrupt_mapped_artifact_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, _ = make_catalog(root)
            result = export_entry(catalog=catalog, entry_id="fixture", output=root / "export")
            body, = (root / "export/artifacts").glob("*.bin")
            body.write_bytes(bytes(body.stat().st_size))
            with self.assertRaisesRegex(ValueError, "digest differs"):
                generate(plan=root / "export" / result["entrypoint"], cache_config=CACHE,
                         artifact_map=root / "export/artifact-map.json", output=root / "failed")
            self.assertFalse((root / "failed").exists())

    def test_parent_and_symlink_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, plan = make_catalog(root)
            export_entry(catalog=catalog, entry_id="fixture", output=root / "export")
            map_path = root / "export/artifact-map.json"
            document = load_json(map_path)
            row = document["artifacts"][str(plan)]
            for escape in (str(plan), "../original/plan.json", "outside/plan.json"):
                if escape.startswith("outside"):
                    (root / "export/outside").symlink_to(root / "original", target_is_directory=True)
                row["path"] = escape
                map_path.write_text(json.dumps(document))
                with self.subTest(escape=escape), self.assertRaises(ValueError):
                    with artifact_context(map_path):
                        pass

    def test_catalog_cli_templates_and_repeated_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, _ = make_catalog(root)
            self.assertEqual(list_entries(catalog, contains="no-match")["count"], 0)
            with redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(cli(["trace", "catalog", "list", "--catalog", str(catalog)]), 0)
            self.assertEqual(json.loads(stream.getvalue())["count"], 1)
            result = export_entry(catalog=catalog, entry_id="body", output=root / "export")
            self.assertEqual(result["unique_files"], 2)
            self.assertEqual(result["kind"], "templates")
            with self.assertRaises(FileExistsError):
                export_entry(catalog=catalog, entry_id="body", output=root / "export")

    def test_incomplete_or_corrupt_catalog_never_publishes_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, plan = make_catalog(root)
            document = load_json(catalog)
            document["recipes"][0]["missing"] = ["unavailable.bin"]
            catalog.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "missing inputs"):
                export_entry(catalog=catalog, entry_id="fixture", output=root / "incomplete")
            self.assertFalse((root / "incomplete").exists())
            document["recipes"][0]["missing"] = []
            document["artifacts"][str(plan)]["bytes"] = 1
            catalog.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "declared size"):
                export_entry(catalog=catalog, entry_id="fixture", output=root / "corrupt")
            self.assertFalse((root / "corrupt/artifact-map.json").exists())
            self.assertFalse((root / "corrupt/export.json").exists())


if __name__ == "__main__":
    unittest.main()

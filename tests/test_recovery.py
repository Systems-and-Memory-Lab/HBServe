"""Nonnative checkpoint integrity and recovery link-payload accounting tests."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hbfsim_client.transaction_protocol import HbfGeometry
from hbserve.contracts import HBServeError, canonical_sha256
from hbserve.recovery import CHECKPOINT_SCHEMA, link_payload_by_stack, verify_checkpoint


def seal_receipt(record):
    record["checkpoint_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "checkpoint_sha256"}
    )


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.image_path = Path(temporary.name) / "image.hbfstate"
        self.image_bytes = b"synthetic artifact for hash verification only"
        self.image_path.write_bytes(self.image_bytes)
        body = {"epoch": 0, "versions": [{"object_id": "request/example/kv", "bytes": 4096}]}
        self.record = {
            "schema": CHECKPOINT_SCHEMA,
            "body": body,
            "body_sha256": canonical_sha256(body),
            "publication": "published",
            "image": {
                "path": str(self.image_path),
                "bytes": len(self.image_bytes),
                "sha256": hashlib.sha256(self.image_bytes).hexdigest(),
            },
        }
        seal_receipt(self.record)

    def test_matching_digests_and_artifact_are_accepted(self):
        for publication in ("published", "staged_unpublished"):
            with self.subTest(publication=publication):
                record = deepcopy(self.record)
                record["publication"] = publication
                seal_receipt(record)
                self.assertIsNone(verify_checkpoint(record))

    def test_tampered_receipt_is_rejected(self):
        self.record["body"]["epoch"] += 1
        with self.assertRaisesRegex(HBServeError, "receipt digest mismatch"):
            verify_checkpoint(self.record)

    def test_tampered_manifest_is_rejected_after_receipt_is_resealed(self):
        self.record["body"]["versions"][0]["bytes"] += 1
        seal_receipt(self.record)
        with self.assertRaisesRegex(HBServeError, "manifest digest mismatch"):
            verify_checkpoint(self.record)

    def test_image_content_and_declared_size_mismatches_are_rejected(self):
        for mismatch in ("content", "size"):
            with self.subTest(mismatch=mismatch):
                record = deepcopy(self.record)
                self.image_path.write_bytes(self.image_bytes)
                if mismatch == "content":
                    self.image_path.write_bytes(b"x" * len(self.image_bytes))
                else:
                    record["image"]["bytes"] += 1
                    seal_receipt(record)
                with self.assertRaisesRegex(HBServeError, "image artifact mismatch"):
                    verify_checkpoint(record)


class LinkPayloadTests(unittest.TestCase):
    def test_partial_pages_conserve_exact_payload_per_stack(self):
        for stacks in (1, 2, 3, 4):
            geometry = HbfGeometry(
                stacks=stacks, channels_per_stack=1, dies_per_channel=1,
                planes_per_die=1, blocks_per_plane=32, pages_per_block=8,
                page_size_bytes=4096, mapping_entries_per_page=2,
            )
            page_bytes = geometry.page_size_bytes
            group_bytes = stacks * geometry.mapping_entries_per_page * page_bytes
            cases = (
                (0, 1),
                (123, 456),
                (page_bytes - 11, 29),
                (page_bytes + 7, 2 * page_bytes - 19),
                (page_bytes, 3 * page_bytes),
                (group_bytes - 13, group_bytes + 31),
            )
            for address, byte_count in cases:
                with self.subTest(stacks=stacks, address=address, bytes=byte_count):
                    end = address + byte_count
                    expected = [0] * stacks
                    for logical_page in range(address // page_bytes, (end - 1) // page_bytes + 1):
                        overlap = min(end, (logical_page + 1) * page_bytes) - max(address, logical_page * page_bytes)
                        expected[geometry.stack_for_logical_page(logical_page)] += overlap
                    actual = link_payload_by_stack(address, byte_count, geometry)
                    self.assertEqual(actual, tuple(expected))
                    self.assertEqual(sum(actual), byte_count)


if __name__ == "__main__":
    unittest.main()

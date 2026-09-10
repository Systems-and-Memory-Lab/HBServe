"""Static object OOM and exact peer migration through the native device model."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hbfsim_client.simulation_session import ResolvedSystemConfig, SimulationSession
from hbfsim_client.transaction_protocol import HbfGeometry, TransactionProtocolError
from hbserve.windows.memory_trace import CanonicalTraceBatch, LogicalTransaction, MemoryLayout, MemoryRegion
from hbserve.windows.peer import PeerCapacityError, PeerKvMigrationRemapper


def layout():
    return MemoryLayout(
        model_descriptor_sha256="a" * 64, model_name="peer-test", vocab_size=16,
        alignment_bytes=4096, address_space_bytes=32768, num_layers=1,
        block_size_tokens=2, bytes_per_token_per_layer=4096,
        kv_block_stride_bytes=8192, num_logical_kv_blocks=2,
        kv_region_id="kv", metadata_region_id="metadata",
        regions=(MemoryRegion("weights", 0, 8192, "immutable_weight"),
                 MemoryRegion("kv", 8192, 16384, "kv"),
                 MemoryRegion("metadata", 24576, 8192, "metadata")),
    )


def access_batch(batch_id, accesses, *, memory_layout=None):
    memory_layout = memory_layout or layout()
    transactions = tuple(LogicalTransaction(f"b{batch_id}/m{index}", operation, address,
                                             byte_count, 0.0)
                         for index, (operation, address, byte_count) in enumerate(accesses))
    return CanonicalTraceBatch(
        batch_id=batch_id, transactions=transactions, layout=memory_layout,
        routing={transaction.id: {"region_id": memory_layout.region_for_range(transaction.addr, transaction.bytes).id,
                                  "group": None} for transaction in transactions},
        audit_labels={}, contract_sha256="b" * 64,
    )


def geometry():
    return HbfGeometry(stacks=2, channels_per_stack=1, dies_per_channel=1,
                       planes_per_die=2, blocks_per_plane=32, pages_per_block=8,
                       page_size_bytes=4096)


def remapper(policy="capacity_migration", **overrides):
    arguments = dict(address_space_bytes=32768, hbm_capacity_bytes=8192,
                     migration_granularity_bytes=4096, transfer_chunk_bytes=4096,
                     hbf_geometry=geometry(), hbf_logical_capacity_bytes=1 << 20,
                     kv_range=(8192, 24576), hbm_stacks=4, hbf_stacks=2,
                     policy=policy, hbf_mapping_mode="cached")
    arguments.update(overrides)
    return PeerKvMigrationRemapper(**arguments)


class PeerTests(unittest.TestCase):
    simulator = None
    output = None

    def test_static_is_identical_until_capacity_requires_a_move(self):
        trace = (access_batch(0, [("R", 0, 4096), ("W", 8192, 4096), ("W", 12288, 4096)]),
                 access_batch(1, [("R", 8192, 4096), ("W", 24576, 4096)]))
        observations = []
        for policy in ("static", "capacity_migration"):
            mapper = remapper(policy)
            preflight = mapper.preflight(trace)
            self.assertEqual(preflight["peak_live_kv_units"], 2)
            self.assertEqual(len(mapper._units), 0)
            observations.append(tuple(mapper.remap(batch).transactions for batch in trace))
            self.assertEqual(mapper._cumulative["demotions"], 0)
        self.assertEqual(*observations)

    def test_static_oom_is_actual_live_capacity_not_virtual_reservation(self):
        static = remapper("static")
        self.assertEqual(static.preflight([access_batch(0, [("W", 8192, 1)])])["peak_live_kv_units"], 1)
        trace = [access_batch(0, [("W", 8192, 4096), ("W", 12288, 4096), ("W", 16384, 4096)])]
        with self.assertRaises(PeerCapacityError) as raised:
            static.preflight(trace)
        self.assertEqual(raised.exception.code, "static_kv_hbm_capacity_exceeded")
        self.assertEqual(raised.exception.receipt["required_hbm_bytes"], 12288)
        with self.assertRaises(PeerCapacityError):
            remapper("static").remap(trace[0])
        self.assertEqual(remapper().preflight(trace)["peak_demoted_home_bytes"], 4096)

    def test_global_physical_oom_is_not_repaired_by_migration(self):
        mapper = remapper(hbm_capacity_bytes=4096, hbf_logical_capacity_bytes=20480)
        with self.assertRaises(PeerCapacityError) as raised:
            mapper.preflight([access_batch(0, [("W", 8192, 4096), ("W", 12288, 4096), ("W", 16384, 4096)])])
        self.assertEqual(raised.exception.code, "peer_total_kv_capacity_exceeded")
        with self.assertRaises(PeerCapacityError) as raised:
            remapper(hbf_logical_capacity_bytes=12288)
        self.assertEqual(raised.exception.code, "hbf_static_image_capacity_exceeded")

    def test_compact_initial_image_has_no_future_kv_population(self):
        mapper = remapper()
        self.assertEqual(mapper.static_flash_bytes, 16384)
        self.assertEqual(mapper.initial_hbf_logical_pages, 4)
        self.assertEqual(mapper.flash_logical_bytes, 24576)
        mapped = mapper.remap(access_batch(0, [("R", 24576, 4096)]))
        self.assertEqual(mapped.transactions[0].addr, 8192)
        image = mapped.receipt["policy"]["initial_hbf_logical_image"]
        self.assertEqual(image["kv_initially_populated_bytes"], 0)
        self.assertFalse(image["future_kv_homes_preinstalled"])

    def test_page_aligned_boundary_slivers_are_not_preinstalled(self):
        mapper = remapper(kv_range=(4096, 28672), migration_granularity_bytes=8192)
        self.assertEqual(mapper.kv_units, 3)
        self.assertEqual(mapper.initial_hbf_logical_pages, 2)
        self.assertEqual(mapper.static_flash_bytes, 8192)
        with self.assertRaisesRegex(TransactionProtocolError, "page aligned"):
            remapper(kv_range=(4097, 28672))

    def test_decode_only_and_unwritten_holes_are_trace_errors_not_oom(self):
        for trace in ([access_batch(0, [("R", 8192, 4096)])],
                      [access_batch(0, [("W", 8704, 256)]), access_batch(1, [("R", 8192, 256)])]):
            with self.assertRaisesRegex(TransactionProtocolError, "read before write") as raised:
                remapper().preflight(trace)
            self.assertNotIsInstance(raised.exception, PeerCapacityError)
        mapper = remapper()
        mapper.remap(access_batch(0, [("W", 8704, 256)]))
        with self.assertRaisesRegex(TransactionProtocolError, "read before write"):
            mapper.remap(access_batch(1, [("R", 8192, 1024)]))
        with self.assertRaisesRegex(TransactionProtocolError, "empty initial"):
            remapper(initial_kv_state="preinstalled")
        with self.assertRaisesRegex(TransactionProtocolError, "writable"):
            remapper(hbf_mapping_mode="direct")

    def test_sparse_unit_moves_only_written_ranges_and_waits_before_slot_reuse(self):
        mapper = remapper(hbm_capacity_bytes=4096)
        first = mapper.remap(access_batch(0, [("W", 8704, 256), ("W", 10240, 512)]))
        mapped = mapper.remap(access_batch(1, [("W", 12288, 4096)]))
        reads = [transaction for transaction in mapped.transactions if transaction.target == "HBM" and transaction.op == "R"]
        self.assertEqual([(transaction.addr, transaction.bytes) for transaction in reads], [(512, 256), (2048, 512)])
        self.assertTrue(all(first.transactions[-1].id in transaction.dependencies for transaction in reads))
        self.assertEqual(sum(transaction.bytes for transaction in mapped.transactions if transaction.target == "D2D_HBM_TO_HBF"), 768)
        reused = next(transaction for transaction in mapped.transactions if transaction.target == "HBM" and transaction.op == "W")
        join = next(transaction for transaction in mapped.transactions if transaction.id in reused.dependencies)
        writes = [transaction.id for transaction in mapped.transactions if transaction.target == "HBF_LOGICAL" and transaction.op == "W"]
        self.assertEqual(set(join.dependencies), set(writes))
        self.assertEqual(mapped.receipt["policy"]["batch_counters"]["demoted_bytes"], 768)
        later = mapper.remap(access_batch(2, [("R", 8704, 256), ("W", 8192, 512), ("R", 8192, 768)]))
        self.assertTrue(all(transaction.target == "HBF_LOGICAL" for transaction in later.transactions))
        self.assertEqual(later.receipt["policy"]["batch_counters"]["demoted_bytes"], 0)
        self.assertEqual(later.receipt["policy"]["cumulative_counters"]["demoted_bytes"], 768)

    def test_lru_is_address_touch_order_and_home_remains_exclusive(self):
        mapper = remapper()
        mapper.remap(access_batch(0, [("W", 8192, 4096), ("W", 12288, 4096), ("R", 8192, 4096)]))
        mapped = mapper.remap(access_batch(1, [("W", 16384, 4096)]))
        self.assertEqual(next(transaction.addr for transaction in mapped.transactions
                              if transaction.target == "HBM" and transaction.op == "R"), 4096)
        self.assertIsNone(mapper._units[1].slot)
        self.assertEqual(mapper._units[1].home, 0)
        later = mapper.remap(access_batch(2, [("W", 12288, 4096), ("R", 8192, 4096)]))
        self.assertEqual([transaction.target for transaction in later.transactions], ["HBF_LOGICAL", "HBM"])
        self.assertEqual(later.receipt["policy"]["batch_counters"]["hbf_kv_write_bytes"], 4096)
        self.assertIsNone(mapper.finalize())
        self.assertFalse(mapper.post_serving_flush_required)

    def test_hbm_can_cover_kv_whose_virtual_space_exceeds_flash(self):
        capacity = geometry().capacity_bytes
        mapper = remapper(address_space_bytes=capacity * 2,
                          hbm_capacity_bytes=capacity * 3 // 2,
                          hbf_logical_capacity_bytes=capacity * 3 // 4,
                          kv_range=(4096, capacity * 2))
        self.assertLess(mapper.flash_logical_bytes, capacity)
        self.assertEqual(mapper.initial_hbf_logical_pages, 1)

    def test_native_media_accounting_and_retained_dependencies(self):
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        observations = []
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "peer.cfg"
            overlay.write_text("\n".join((
                "hbm-capacity-bytes=16384", "hbf-stacks=2", "hbf-channels=1", "hbf-dies-per-channel=1",
                "hbf-planes-per-die=2", "hbf-blocks-per-plane=32", "hbf-pages-per-block=8",
                "hbf-page-size=4096", "hbf-gc-low-watermark-pages=8", "hbf-gc-hard-watermark-pages=0",
                "hbf-gc-reserved-free-blocks-per-plane=2", "hbf-write-buffer-pages=16",
                "hbf-write-buffer-flush-threshold-pages=8", "hbf-mapping-mode=cached",
                "hbf-ctrl-dram-bytes=262144", "hbf-thermal-enable=false", "",
            )))
            config = ResolvedSystemConfig.load((ROOT / "configs/4hbm-4hbf.cfg", overlay)).resolve(self.simulator)
            trace = [access_batch(0, [("R", 0, 4096), ("W", 8192, 4096), ("W", 12288, 4096)]),
                     access_batch(1, [("W", 16384, 4096), ("W", 20480, 4096)])]
            trace.extend(access_batch(batch_id, [("R", 24576, 4096)]) for batch_id in range(2, 14))
            trace.append(access_batch(14, [("R", 8192, 4096), ("W", 8192, 4096), ("R", 20480, 4096)]))
            sparse_trace = [access_batch(0, [("W", 8704, 256), ("W", 10240, 512)]),
                            access_batch(1, [("W", 12288, 4096)]),
                            access_batch(2, [("R", 8704, 256), ("W", 8192, 512), ("R", 8192, 768)])]
            cases = (("static", 16384, trace, 0, 0), ("capacity_migration", 16384, trace, 0, 0),
                     ("capacity_migration", 8192, trace, 2, 8192),
                     ("capacity_migration", 4096, sparse_trace, 1, 768))
            for policy, hbm_bytes, selected_trace, expected_demotions, expected_copy_bytes in cases:
                mapper = remapper(policy, hbm_capacity_bytes=hbm_bytes,
                                  hbf_geometry=config.hbf_geometry,
                                  hbf_logical_capacity_bytes=config.logical_hbf_capacity_bytes)
                preflight = mapper.preflight(selected_trace)
                session = SimulationSession(simulator_path=self.simulator, system_config=config,
                                            enable_hbm=True, enable_hbf=True, hbm_capacity_bytes=hbm_bytes,
                                            initial_hbf_logical_first_lpn=mapper.initial_hbf_logical_first_lpn,
                                            initial_hbf_logical_pages=mapper.initial_hbf_logical_pages)
                completions = []
                mapped_receipts = []
                try:
                    for batch in selected_trace:
                        mapped = mapper.remap(batch)
                        completions.append(session.submit(mapped))
                        mapped_receipts.append(mapped.receipt)
                    mapper.finalize()
                finally:
                    session.close()
                source = session.source_receipt()
                observations.append({"policy": policy, "hbm_bytes": hbm_bytes,
                                     "preflight": preflight, "source": source,
                                     "completions": completions, "remap_receipts": mapped_receipts})
                self.assertEqual(source["execution_options"]["initial_hbf_logical_image"]["pages"], 4)
                self.assertEqual(mapped_receipts[-1]["policy"]["cumulative_counters"]["demotions"],
                                 expected_demotions)
                self.assertTrue(all(receipt["invariants"]["address_rw_bytes_conserved"] for receipt in mapped_receipts))
                totals = source["final_measurement"]["device_workload_totals"]
                self.assertEqual(totals["base_die_link"]["write_bytes"], expected_copy_bytes)
                if expected_demotions:
                    self.assertGreater(totals["hbf"]["physical_write_bytes"], 0)
                    self.assertGreater(totals["hbf"]["mapping_update_ops"], 0)
                    self.assertGreater(totals["hbf"]["mapping_program_payload_bytes"], 0)
                    self.assertGreater(source["final_measurement"]["end_of_session_drain"]["drain_physical_bytes"], 0)
            self.assertEqual([entry["finish_ns"] for entry in observations[0]["completions"]],
                             [entry["finish_ns"] for entry in observations[1]["completions"]])
            if self.output is not None:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                self.output.write_text(json.dumps({"result": "pass", "experiments": observations}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path)
    parser.add_argument("--output", type=Path)
    arguments, remaining = parser.parse_known_args()
    PeerTests.simulator = arguments.simulator
    PeerTests.output = arguments.output
    unittest.main(argv=[sys.argv[0], *remaining])

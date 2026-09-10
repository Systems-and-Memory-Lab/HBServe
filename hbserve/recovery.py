"""Quiescent, versioned HBF recovery for HBServe's resident-KV experiments.

The native simulator persists media/FTL state, not tensor values. This module
binds that image to HBServe objects and token frontiers, charges copy-on-write
KV snapshots and application-log I/O, and rebuilds volatile allocation state
using the normal compiler and placement. It does not persist the HBM prefix
cache or claim arbitrary-instruction power-failure correctness.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from hbfsim_client.transaction_protocol import (
    HbfGeometry,
    Transaction,
    TransactionBatch,
    hbf_link_bytes_by_stack,
)
from hbserve.compiler import HBServeCompiler
from hbserve.contracts import (
    BatchSlice,
    CanonicalServingBatch,
    HBServeError,
    ScheduledBatch,
    canonical_sha256,
)
from hbserve.hbfsim import HbfSimExecutor
from hbserve.engine import BatchExecution
from hbserve.io import write_json_atomic
from hbserve.placement import HBServePlacement


CHECKPOINT_SCHEMA = {"name": "hbserve.recovery_checkpoint", "version": 1}
FAILURE_DOMAIN = {
    "kind": "node_volatile_loss_at_quiescent_protocol_boundary",
    "lost": [
        "HBM_contents", "host_scheduler_and_allocators", "HBM_prefix_cache",
        "volatile_translation_and_data_caches", "thermal_runtime_state",
    ],
    "retained": ["HBF_media_mapping_allocator_and_wear", "external_model_source"],
    "payload_values_modeled": False,
    "mid_program_or_torn_root_write_modeled": False,
    "power_loss_protection_assumed": False,
    "native_image_import_latency_modeled": False,
    "native_FTL_journal_replay_modeled": False,
    "mapping_cache_refills_physically_timed": True,
    "application_log_io_and_replay_timed": True,
    "persistent_HBF_prefix_cache": False,
}


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _encoded(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def link_payload_by_stack(address: int, byte_count: int, geometry: HbfGeometry) -> tuple[int, ...]:
    """Account exact payload, including the partial pages of a KV snapshot."""

    if address < 0 or byte_count <= 0:
        raise HBServeError("recovery link range must be positive and nonnegative-addressed")
    page = geometry.page_size_bytes
    begin = address // page * page
    end = _align_up(address + byte_count, page)
    counts = list(hbf_link_bytes_by_stack(begin, end - begin, geometry))
    counts[geometry.stack_for_logical_page(address // page)] -= address - begin
    counts[geometry.stack_for_logical_page((address + byte_count - 1) // page)] -= end - address - byte_count
    return tuple(counts)


def _artifact_matches(artifact: Mapping[str, Any]) -> bool:
    candidate = Path(str(artifact.get("path", "")))
    if candidate.is_symlink() or not candidate.is_file():
        return False
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return candidate.stat().st_size == artifact.get("bytes") and digest.hexdigest() == artifact.get("sha256")


def verify_checkpoint(record: Mapping[str, Any]) -> None:
    if record.get("schema") != CHECKPOINT_SCHEMA:
        raise HBServeError("unsupported HBServe recovery checkpoint")
    if canonical_sha256({key: value for key, value in record.items() if key != "checkpoint_sha256"}) != record.get("checkpoint_sha256"):
        raise HBServeError("recovery checkpoint receipt digest mismatch")
    body = record.get("body")
    if not isinstance(body, Mapping) or canonical_sha256(body) != record.get("body_sha256"):
        raise HBServeError("recovery object/version manifest digest mismatch")
    if record.get("publication") not in {"published", "staged_unpublished"}:
        raise HBServeError("unknown recovery publication boundary")
    if not _artifact_matches(record.get("image", {})):
        raise HBServeError("recovery native image artifact mismatch")


def read_checkpoint(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    verify_checkpoint(record)
    return record


class RecoveryExecutor(HbfSimExecutor):
    """A bounded COW checkpoint policy atop the ordinary HBServe executor.

    All weights reside in HBF and active KV remains in HBM. Cold-KV migration,
    model caching and prefix sharing are rejected rather than silently omitted
    from a snapshot. Each checkpoint copies all live KV into new HBF extents;
    the append-only arena deliberately fails on exhaustion instead of implying
    unimplemented snapshot reclamation or an unbounded production policy.
    """

    def __init__(
        self,
        *,
        simulator_path: Path,
        system_config_paths: Sequence[Path],
        placement: HBServePlacement,
        compiler: HBServeCompiler,
        recovery_identity: Mapping[str, str],
        image_record: Mapping[str, Any] | None = None,
        controller_boot_ns: float = 10_000.0,
        log_replay_ns_per_record: float = 100.0,
    ) -> None:
        spec = placement.spec
        if (
            set(spec.model_weight_tiers.values()) != {"hbf"}
            or spec.object_tier_overrides
            or spec.kv_placement.cold is not None
            or spec.prefix_cache_bytes
            or spec.hbm_model_cache_bytes
            or spec.initial_cached_models
        ):
            raise HBServeError("recovery requires HBF weights, resident HBM KV, and no prefix/model cache or migration")
        if spec.hbm_runtime_reserve_bytes < spec.hbf_page_size_bytes:
            raise HBServeError("recovery needs one HBM scratch page in the runtime reserve")
        if any(request.token_ids is None for request in compiler.request_trace.requests):
            raise HBServeError("versioned recovery requires explicit processed input token IDs")
        if set(recovery_identity) != {"tokenizer", "position_encoding", "kv_representation"} or any(
            not isinstance(value, str) or not value for value in recovery_identity.values()
        ):
            raise HBServeError("recovery identity must declare tokenizer, position_encoding and kv_representation")
        for value in (controller_boot_ns, log_replay_ns_per_record):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise HBServeError("recovery timing sensitivities must be finite and nonnegative")
        if {key: model.digest for key, model in placement.models.items()} != {
            key: model.digest for key, model in compiler.models.items()
        }:
            raise HBServeError("recovery compiler and placement model identities differ")
        self.compiler = compiler
        self.identity = {
            "model_sha256": {key: model.digest for key, model in compiler.models.items()},
            "request_trace_sha256": compiler.request_trace.digest,
            "placement_sha256": spec.digest,
            "timing": compiler.timing.canonical(),
            "prefetch_depth": compiler.prefetch_depth,
            "representation": dict(recovery_identity),
        }
        if image_record is not None:
            verify_checkpoint(image_record)
            if image_record["body"]["identity"] != self.identity:
                raise HBServeError("recovery model/request/representation/placement identity mismatch")
        self.controller_boot_ns = float(controller_boot_ns)
        self.log_replay_ns_per_record = float(log_replay_ns_per_record)
        self._segments: dict[str, list[dict[str, int]]] = {}
        self._progress: dict[str, dict[str, Any]] = {}
        self._initial_emitted_tokens: dict[str, int] = {}
        self._schedules: list[dict[str, Any]] = []
        self._published: dict[str, Any] | None = None
        self._image_record = None if image_record is None else deepcopy(dict(image_record))
        self._epoch = 0 if image_record is None else int(image_record["body"]["epoch"]) + 1
        self._lifecycle_batch_id = 1_000_000_000
        self.lifecycle: list[dict[str, Any]] = []
        self.page_bytes = spec.hbf_page_size_bytes
        self.root_addr = placement.hbf_allocation_bytes
        self._cursor = self.root_addr + self.page_bytes
        if image_record is not None:
            self._cursor = int(image_record["arena_next_addr"])
        self._scratch = spec.hbm_capacity_bytes - spec.hbm_runtime_reserve_bytes
        super().__init__(
            simulator_path=simulator_path,
            system_config_paths=system_config_paths,
            placement=placement,
            initial_hbf_persistent_image=(None if image_record is None else Path(image_record["image"]["path"])),
            preinstall_hbf_weights=False,
            enable_external_recovery=True,
            on_batch_completed=self._record_completed,
        )
        if self._cursor > spec.hbf_capacity_bytes:
            self.close()
            raise HBServeError("recovery arena exceeds HBF payload capacity")

    @property
    def progress(self) -> dict[str, Any]:
        return deepcopy(self._progress)

    @property
    def schedules(self) -> list[dict[str, Any]]:
        return deepcopy(self._schedules)

    def _record_completed(
        self, batch: CanonicalServingBatch, mapped: TransactionBatch,
        completion: Mapping[str, Any],
    ) -> None:
        if any(mapped.receipt["policy_overhead"].values()):
            raise HBServeError("recovery cannot snapshot unmodeled placement movement")
        physical = [item for item in mapped.transactions if item.target != "BARRIER"]
        cursor = 0
        for operation in batch.memory_operations:
            consumed = 0
            while consumed < operation.bytes:
                if cursor >= len(physical):
                    raise HBServeError("recovery projection ended before its canonical operation")
                item = physical[cursor]
                cursor += 1
                if item.op != operation.op or consumed + item.bytes > operation.bytes:
                    raise HBServeError("recovery projection changed operation boundaries")
                if operation.op == "W":
                    if item.target != "HBM" or not operation.object_id.startswith("request/"):
                        raise HBServeError("recovery only supports append-only active KV writes")
                    segments = self._segments.setdefault(operation.object_id, [])
                    offset = operation.offset + consumed
                    expected = 0 if not segments else segments[-1]["offset"] + segments[-1]["bytes"]
                    if offset != expected:
                        raise HBServeError("recovery KV writes are not a contiguous token frontier")
                    if segments and segments[-1]["hbm_addr"] + segments[-1]["bytes"] == item.addr:
                        segments[-1]["bytes"] += item.bytes
                    else:
                        segments.append({"offset": offset, "hbm_addr": item.addr, "bytes": item.bytes})
                consumed += item.bytes
        if cursor != len(physical):
            raise HBServeError("recovery projection contains unexplained physical operations")
        for item in batch.schedule.slices:
            request = self.compiler.requests[item.request_id]
            previous = self._progress.get(item.request_id, {
                "processed_tokens": 0,
                "emitted_tokens": self._initial_emitted_tokens.get(item.request_id, 0),
            })
            if previous["processed_tokens"] != item.token_begin:
                raise HBServeError("recovery completion skips a token frontier")
            emitted = previous["emitted_tokens"] + int(item.emits_output)
            token_identity_end = min(item.token_end + int(emitted > 0), request.processed_input_tokens)
            self._progress[item.request_id] = {
                "processed_tokens": item.token_end,
                "emitted_tokens": emitted,
                "token_history_sha256": canonical_sha256(list(request.token_ids[:token_identity_end])),
                "next_input_token_present": item.token_end < request.processed_input_tokens,
            }
        self._schedules.append({
            "schedule": batch.schedule.canonical(),
            "canonical_sha256": batch.digest,
        })

    def _submit(self, name: str, transactions: list[Transaction]) -> dict[str, Any]:
        if not transactions:
            raise HBServeError("empty recovery lifecycle transaction batch")
        batch = TransactionBatch(
            batch_id=self._lifecycle_batch_id,
            logical_trace_sha256=canonical_sha256({"name": name, "identity": self.identity}),
            routing_sidecar_sha256=canonical_sha256({"recovery": name, "page_bytes": self.page_bytes}),
            transactions=tuple(transactions),
            completions=False,
            receipt={"scope": "HBServe recovery lifecycle", "name": name},
        )
        self._lifecycle_batch_id += 1
        completion = self.submit_transactions(batch)
        record = {"name": name, "completion": completion, "mapped_sha256": batch.transaction_trace_sha256}
        self.lifecycle.append(record)
        return record

    def _emit(
        self, transactions: list[Transaction], target: str, op: str | None,
        addr: int, byte_count: int, dependencies: Sequence[str] = (),
        *, stack: int | None = None, duration_ns: float = 0.0,
    ) -> str:
        identifier = f"recovery/{self._lifecycle_batch_id}/{len(transactions)}"
        transactions.append(Transaction(
            id=identifier, target=target, op=op, addr=addr, bytes=byte_count,
            issue_ns=0.0, dependencies=tuple(dependencies), stack=stack,
            duration_ns=duration_ns,
        ))
        return identifier

    def _copy(
        self, transactions: list[Transaction], source: str, source_addr: int,
        destination: str, destination_addr: int, byte_count: int,
        dependencies: Sequence[str] = (),
    ) -> str:
        read = self._emit(transactions, source, "R", source_addr, byte_count, dependencies)
        hbf_addr = destination_addr if destination == "HBF_LOGICAL" else source_addr
        link_target = "D2D_HBM_TO_HBF" if destination == "HBF_LOGICAL" else "D2D_HBF_TO_HBM"
        link_op = "W" if destination == "HBF_LOGICAL" else "R"
        links = [
            self._emit(transactions, link_target, link_op, hbf_addr, stack_bytes, (read,), stack=stack)
            for stack, stack_bytes in enumerate(link_payload_by_stack(hbf_addr, byte_count, self.system_config.hbf_geometry))
            if stack_bytes
        ]
        return self._emit(transactions, destination, "W", destination_addr, byte_count, links)

    def _metadata_write(self, name: str, addr: int, byte_count: int) -> dict[str, Any]:
        transactions: list[Transaction] = []
        fence: tuple[str, ...] = ()
        for offset in range(0, byte_count, self.page_bytes):
            count = min(self.page_bytes, byte_count - offset)
            staged = self._emit(transactions, "HBM", "W", self._scratch, count, fence)
            copied = self._copy(transactions, "HBM", self._scratch, "HBF_LOGICAL", addr + offset, count, (staged,))
            fence = (copied,)
        return self._submit(name, transactions)

    def load_external_weights(self) -> dict[str, Any]:
        """Cold reload through bounded HBM staging, with explicit link traffic."""

        if self._image_record is not None or self.lifecycle or self._schedules:
            raise HBServeError("cannot cold-install weights over a retained HBF image")
        startup: list[Transaction] = []
        self._emit(startup, "BARRIER", None, 0, 0, duration_ns=self.controller_boot_ns)
        self._submit("controller_boot", startup)
        transactions: list[Transaction] = []
        chunk_bytes = min(self.placement.spec.model_load_chunk_bytes, self.placement.spec.hbm_runtime_reserve_bytes)
        slots = self.placement.spec.hbm_runtime_reserve_bytes // chunk_bytes
        fences: list[tuple[str, ...]] = [() for _ in range(slots)]
        chunk_index = 0
        total = 0
        external_capacity = self.placement.spec.external_capacity_bytes
        for memory_object in self.placement.receipt()["static_objects"]:
            if memory_object["target"] != "HBF_LOGICAL":
                raise HBServeError("cold reload encountered a non-HBF weight")
            if memory_object["addr"] + memory_object["bytes"] > external_capacity:
                raise HBServeError("external model checkpoint exceeds declared capacity")
            for offset in range(0, memory_object["bytes"], chunk_bytes):
                count = min(chunk_bytes, memory_object["bytes"] - offset)
                address = memory_object["addr"] + offset
                slot = chunk_index % slots
                scratch = self._scratch + slot * chunk_bytes
                fetched = self._emit(transactions, "EXTERNAL", "R", address, count, fences[slot])
                staged = self._emit(transactions, "HBM", "W", scratch, count, (fetched,))
                copied = self._copy(transactions, "HBM", scratch, "HBF_LOGICAL", address, count, (staged,))
                fences[slot] = (copied,)
                chunk_index += 1
                total += count
        record = self._submit("external_weight_reload", transactions)
        record["logical_model_bytes"] = total
        record["staging_buffer_bytes"] = self.placement.spec.hbm_runtime_reserve_bytes
        record["chunk_bytes"] = chunk_bytes
        record["staging_slots"] = slots
        checkpoint = self.checkpoint("weights-installed")
        self.lifecycle.append({"name": "weight_install_mapping_checkpoint", "completion": checkpoint})
        return record

    def publish_checkpoint(
        self, output: Path, *, include_kv: bool = True, publish: bool = True,
    ) -> dict[str, Any]:
        """Copy data, drain data/log, then publish and drain the root record."""

        if not include_kv and self._progress:
            raise HBServeError("weights-only checkpoint must precede request execution")
        if set(self._progress) != {item["request_id"] for item in self.placement.receipt()["final_state"]["live_kv"]}:
            raise HBServeError("checkpoint only supports currently live, never-released requests")
        output.mkdir(parents=True, exist_ok=True)
        image_path = output / f"epoch-{self._epoch}.hbfstate"
        if image_path.exists():
            raise HBServeError("recovery checkpoint epoch already exists")
        first_addr = self._cursor
        frontier = self.frontier_ns
        first_lifecycle = len(self.lifecycle)
        versions: list[dict[str, Any]] = []
        copies: list[Transaction] = []
        for object_id, segments in sorted(self._segments.items()):
            count = sum(segment["bytes"] for segment in segments)
            base = self._cursor
            self._cursor += _align_up(count, self.page_bytes)
            versions.append({
                "object_id": object_id, "version": self._epoch, "hbf_addr": base,
                "bytes": count, "segments": deepcopy(segments),
            })
            for segment in segments:
                self._copy(copies, "HBM", segment["hbm_addr"], "HBF_LOGICAL", base + segment["offset"], segment["bytes"])
        body = {
            "epoch": self._epoch, "identity": self.identity,
            "progress": self.progress, "schedules": self.schedules,
            "initial_emitted_tokens": dict(self._initial_emitted_tokens),
            "versions": versions, "allocation_begin": first_addr,
            "previous_publication_sha256": None if self._published is None else self._published["body_sha256"],
        }
        log_bytes = _align_up(len(_encoded(body)), self.page_bytes)
        log_addr = self._cursor
        self._cursor += log_bytes
        if self._cursor > self.placement.spec.hbf_capacity_bytes:
            raise HBServeError("append-only recovery arena exhausted; no snapshot reclamation is modeled")
        if copies:
            self._submit("kv_copy_on_write_snapshot", copies)
        self._metadata_write("application_version_log", log_addr, log_bytes)
        data_checkpoint = self.checkpoint(f"epoch-{self._epoch}-data-log-durable")
        self.lifecycle.append({"name": "data_and_log_mapping_drain", "completion": data_checkpoint})
        if publish:
            self._metadata_write("publication_root", self.root_addr, self.page_bytes)
        native = self.checkpoint_image(f"epoch-{self._epoch}-image", image_path)
        self.lifecycle.append({"name": "image_checkpoint", "completion": native})
        record = {
            "schema": CHECKPOINT_SCHEMA,
            "body": body, "body_sha256": canonical_sha256(body),
            "publication": "published" if publish else "staged_unpublished",
            "image": native["persistent_image"],
            "root_addr": self.root_addr, "root_bytes": self.page_bytes,
            "log_addr": log_addr, "log_bytes": log_bytes,
            "arena_next_addr": self._cursor,
            "elapsed_ns": self.frontier_ns - frontier,
            "durable_frontier_ns": self.frontier_ns,
            "kv_snapshot_bytes": sum(version["bytes"] for version in versions),
            "lifecycle": deepcopy(self.lifecycle[first_lifecycle:]),
        }
        record["checkpoint_sha256"] = canonical_sha256(record)
        write_json_atomic(output / f"epoch-{self._epoch}.json", record)
        if publish:
            pending = output / f"publish-{self._epoch}.json"
            write_json_atomic(pending, record)
            os.replace(pending, output / "published.json")
            self._published = deepcopy(record)
        self._epoch += 1
        return record

    def restore_checkpoint(self, published: Mapping[str, Any]) -> dict[str, Any]:
        """Replay the published log and refill fresh HBM from its KV versions."""

        verify_checkpoint(published)
        media = self._image_record
        if media is None or self._schedules or self.lifecycle:
            raise HBServeError("checkpoint restoration requires a fresh retained-image executor")
        if published["publication"] != "published" or published["body"]["identity"] != self.identity:
            raise HBServeError("recovery selected an unpublished or incompatible epoch")
        if media["body_sha256"] != published["body_sha256"]:
            if (
                media["publication"] != "staged_unpublished"
                or media["body"]["previous_publication_sha256"] != published["body_sha256"]
                or media["body"]["allocation_begin"] != published["arena_next_addr"]
            ):
                raise HBServeError("retained media is not an append-only child of the published epoch")
        body = published["body"]
        self._initial_emitted_tokens = dict(body["initial_emitted_tokens"])
        transactions: list[Transaction] = []
        boot = self._emit(transactions, "BARRIER", None, 0, 0, duration_ns=self.controller_boot_ns)
        fence = (boot,)
        for address, count in ((published["root_addr"], published["root_bytes"]), (published["log_addr"], published["log_bytes"])):
            for offset in range(0, count, self.page_bytes):
                copied = self._copy(transactions, "HBF_LOGICAL", address + offset, "HBM", self._scratch, self.page_bytes, fence)
                fence = (copied,)
        records = len(body["versions"]) + len(body["schedules"]) + 1
        self._emit(transactions, "BARRIER", None, 0, 0, fence, duration_ns=records * self.log_replay_ns_per_record)
        replay = self._submit("root_and_application_log_replay", transactions)
        replay["controller_boot_sensitivity_ns"] = self.controller_boot_ns
        replay["log_replay_records"] = records
        replay["cpu_log_replay_sensitivity_ns"] = records * self.log_replay_ns_per_record
        for event in body["schedules"]:
            raw = event["schedule"]
            slices = tuple(BatchSlice(**item) for item in raw["slices"])
            schedule = ScheduledBatch(
                batch_id=raw["batch_id"], model_id=raw["model_id"], slices=slices,
                not_before_ns=raw["not_before_ns"],
            )
            for item in slices:
                if not self.placement.is_admitted(item.request_id):
                    self.admit_request(self.compiler.requests[item.request_id])
            self.reserve(schedule.batch_id, slices)
            canonical = self.compiler.compile(schedule)
            if canonical.digest != event["canonical_sha256"]:
                raise HBServeError("restored schedule no longer compiles to the committed workload")
            mapped = self.placement.map_batch(canonical, session_frontier_ns=schedule.not_before_ns)
            self._record_completed(canonical, mapped, {})
        if self.progress != body["progress"] or set(self._segments) != {item["object_id"] for item in body["versions"]}:
            raise HBServeError("restored request/object frontier differs from the committed log")
        copies: list[Transaction] = []
        for version in body["versions"]:
            segments = self._segments[version["object_id"]]
            if segments != version["segments"] or version["version"] != body["epoch"]:
                raise HBServeError("restored KV allocation or version differs from the committed log")
            for segment in segments:
                self._copy(copies, "HBF_LOGICAL", version["hbf_addr"] + segment["offset"], "HBM", segment["hbm_addr"], segment["bytes"])
        if copies:
            self._submit("restore_committed_kv_to_volatile_hbm", copies)
        self._published = deepcopy(dict(published))
        return {
            "committed_epoch": body["epoch"],
            "media_epoch": media["body"]["epoch"],
            "unpublished_epoch_ignored": media["body_sha256"] != published["body_sha256"],
            "restored_kv_bytes": sum(item["bytes"] for item in body["versions"]),
            "progress": self.progress,
            "elapsed_ns": self.frontier_ns,
            "lifecycle": deepcopy(self.lifecycle),
        }

    def recompute_checkpoint(
        self, published: Mapping[str, Any]
    ) -> tuple[CanonicalServingBatch, BatchExecution]:
        """Re-prefill the committed context once; suppress the duplicate output.

        A client outside the failed node supplies its acknowledged token history.
        This is cheaper than replaying every original decode and is not an
        external durable KV baseline. The normal compiler generates every KV
        byte again, before the acknowledged frontier is accepted.
        """

        verify_checkpoint(published)
        body = published["body"]
        if published["publication"] != "published" or body["identity"] != self.identity:
            raise HBServeError("KV recomputation requires a compatible published checkpoint")
        if self._schedules or self._progress or len(body["progress"]) != 1:
            raise HBServeError("compact KV recomputation requires one fresh request")
        request_id, state = next(iter(body["progress"].items()))
        if state["emitted_tokens"] < 1:
            raise HBServeError("compact KV recomputation requires an acknowledged output boundary")
        request = self.compiler.requests[request_id]
        self._initial_emitted_tokens[request_id] = state["emitted_tokens"] - 1
        self.admit_request(request)
        slices = (BatchSlice(
            request_id=request_id, token_begin=0, token_count=state["processed_tokens"],
            context_tokens_before=0, emits_output=True, phase="prefill",
        ),)
        batch_id = max(event["schedule"]["batch_id"] for event in body["schedules"])
        self.reserve(batch_id, slices)
        batch = self.compiler.compile(ScheduledBatch(
            batch_id=batch_id, model_id=request.model_id, slices=slices,
            not_before_ns=self.frontier_ns,
        ))
        execution = self.execute(batch)
        if self.progress != body["progress"]:
            raise HBServeError("recomputed KV does not reach the committed token identity/frontier")
        return batch, execution

#!/usr/bin/env python3
"""Closed-loop adapter from HBServe batches to HBFSim simulation sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from hbfsim_client.simulation_session import (
    SimulationSession,
    SimulationSessionError,
    ResolvedSystemConfig,
)
from hbserve.contracts import (
    BatchSlice,
    CanonicalServingBatch,
    RequestSpec,
    HBServeError,
)
from hbserve.engine import BatchExecution
from hbserve.placement import HBServePlacement


class HbfSimExecutor:
    """One physical HBM/HBF/external memory system in a persistent session."""

    def __init__(
        self,
        *,
        simulator_path: Path,
        system_config_paths: Sequence[Path],
        placement: HBServePlacement,
    ) -> None:
        self.placement = placement
        try:
            self.system_config = ResolvedSystemConfig.load(system_config_paths)
        except SimulationSessionError as error:
            raise HBServeError(str(error)) from error
        geometry = self.system_config.hbf_geometry
        if placement.spec.hbf_page_size_bytes != geometry.page_size_bytes:
            raise HBServeError(
                "serving placement HBF page size differs from the physical config"
            )
        if (
            placement.hbf_geometry is not None
            and placement.hbf_geometry != geometry
        ):
            raise HBServeError(
                "serving placement HBF geometry differs from the physical config"
            )
        if placement.spec.hbf_capacity_bytes > geometry.capacity_bytes:
            raise HBServeError(
                "serving placement HBF capacity exceeds the physical raw geometry"
            )
        if (
            placement.enable_external
            and placement.spec.external_page_size_bytes
            != int(self.system_config.external_backing_identity["page_size_bytes"])
        ):
            raise HBServeError(
                "serving placement external page size differs from the config"
            )
        if (
            placement.enable_external
            and placement.spec.external_capacity_bytes
            > int(self.system_config.external_backing_identity["capacity_bytes"])
        ):
            raise HBServeError(
                "serving placement external capacity exceeds the physical config"
            )
        try:
            self._session = SimulationSession(
                simulator_path=simulator_path,
                system_config=self.system_config,
                enable_hbm=True,
                enable_hbf=placement.enable_hbf,
                enable_external=placement.enable_external,
                hbm_capacity_bytes=placement.spec.hbm_capacity_bytes,
                initial_hbf_logical_first_lpn=0,
                initial_hbf_logical_pages=(
                    placement.initial_hbf_logical_pages
                ),
            )
        except SimulationSessionError as error:
            raise HBServeError(str(error)) from error
        self._closed = False

    @property
    def frontier_ns(self) -> float:
        return self._session.completed_frontier_ns

    @property
    def execution_identity(self) -> Mapping[str, Any]:
        source_setup = self._session.source_receipt()
        source_setup.pop("final_measurement", None)
        return {
            "kind": "hbserve_hbfsim_simulation_session",
            "physical_memory_timing": True,
            "absolute_memory_timing_claim_eligible": False,
            "ttft_tpot_claim_eligible": False,
            "single_device": True,
            "network_timing": False,
            "claim_note": (
                "HBFSim supplies modeled physical memory timing; compute is "
                "the run's timing model (roofline, linear, or none), with no "
                "GPU-kernel/tile trace or end-to-end serving calibration"
            ),
            "placement": deepcopy(
                self.placement.frontier_ns_independent_state
            ),
            "source_setup": source_setup,
        }

    def admit_request(self, request: RequestSpec) -> None:
        self.placement.admit_request(request)

    def can_reserve(self, slices: Sequence[BatchSlice]) -> bool:
        return self.placement.can_reserve(slices)

    def reserve(
        self, batch_id: int, slices: Sequence[BatchSlice]
    ) -> Mapping[str, Any]:
        return self.placement.reserve(batch_id, slices)

    def preempt_request(self, request_id: str) -> Mapping[str, Any]:
        return self.placement.preempt_request(request_id)

    def release_request(self, request_id: str) -> None:
        self.placement.release_request(request_id)

    def execute(self, batch: CanonicalServingBatch) -> BatchExecution:
        if self._closed:
            raise HBServeError("serving executor is closed")
        origin = self._session.completed_frontier_ns
        try:
            mapped = self.placement.map_batch(
                batch,
                session_frontier_ns=origin,
            )
            # The serving batch ends with its completion barrier; that
            # barrier is the blocking frontier the next batch waits for.
            # Everything else still executes and stays dependable.  Only
            # the aggregate receipt is kept: per-transaction completion
            # rows are never read here and would dominate the result.
            final = mapped.transactions[-1]
            if final.target == "BARRIER":
                mapped = replace(mapped, frontier=(final.id,), completions=False)
            else:
                mapped = replace(mapped, completions=False)
            completion = self._session.submit(mapped)
        except (SimulationSessionError, HBServeError) as error:
            raise HBServeError(str(error)) from error
        if completion.get("logical_trace_sha256") != batch.digest:
            raise HBServeError(
                "HBFSim completion changed the canonical serving digest"
            )
        return BatchExecution(
            batch_id=batch.schedule.batch_id,
            batch_origin_ns=float(completion["batch_origin_ns"]),
            finish_ns=float(completion["blocking_finish_ns"]),
            canonical_sha256=batch.digest,
            mapped_sha256=mapped.transaction_trace_sha256,
            remap_receipt=mapped.receipt,
            physical_completion=completion,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._session.close()
        except SimulationSessionError as error:
            raise HBServeError(str(error)) from error
        self._closed = True

    def final_receipt(self) -> dict[str, Any]:
        if not self._closed:
            raise HBServeError(
                "serving executor must close before final receipt publication"
            )
        placement = self.placement.receipt()
        if not all(placement["final_invariants"].values()):
            raise HBServeError(
                "serving placement retained request KV state at publication"
            )
        return {
            "session": self._session.source_receipt(),
            "placement": placement,
        }

    def __enter__(self) -> "HbfSimExecutor":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()

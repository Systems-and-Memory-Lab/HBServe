#!/usr/bin/env python3
"""Build and run matched fixed-footprint memory windows across topologies.

The simulator sees only mapped address transactions. Every topology uses the
same logical trace; phase-level object identity is joined to HBFSim's
semantic-free device receipts after execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[2]

from hbfsim_client.fixed_footprint_metrics import (  # noqa: E402
    build_fixed_footprint_metrics,
)
from hbfsim_client.simulation_session import (  # noqa: E402
    ResolvedSystemConfig,
    SimulationSession,
)
from hbserve.windows.fixed_footprint_trace import (  # noqa: E402
    FixedFootprintTrace,
    build_fixed_footprint_trace,
)
from hbserve.windows.population import (  # noqa: E402
    build_explicit_fixed_population,
)
from hbserve.windows.remap import (  # noqa: E402
    DirectAttachedRemapper,
    HbmFrontedBackingRemapper,
    RemapError,
)
from hbserve.contracts import HBServeError
from hbserve.io import load_json_object
from hbserve.windows.placement import placement_order
from hbserve.windows.peer import PeerCapacityError, PeerKvMigrationRemapper
from hbserve.windows.memory_trace import canonical_sha256


DEFAULT_EXPERIMENT = ROOT / "configs/windows/miniquick-serving.json"
DEFAULT_SIMULATOR = None
CXL_SSD_TOPOLOGY_ID = "8h0f-cxl-ssd"
CXL_SSD_WRITE_COMPLETION_BOUNDARY = {
    "write_completion": "caller_completion_after_device_internal_dram_acceptance",
    "nand_destage": "excluded_after_internal_dram_acceptance",
    "persistence_basis": (
        "cxl_ssd_guarantees_eventual_flash_persistence_after_internal_dram_acceptance"
    ),
}
EXPERIMENT_SCHEMA = {
    "name": "hbserve.fixed_window_experiment",
    "version": 1,
}
PREFLIGHT_SCHEMA = {
    "name": "hbserve.fixed_window_preflight",
    "version": 1,
}
RESULT_SCHEMA = {
    "name": "hbserve.fixed_window_result",
    "version": 1,
}
TOPOLOGY_VIEW_SCHEMA = {
    "name": "hbserve.topology_view",
    "version": 1,
}
EXPECTED_CAPACITY_OOM_STAGE = "initial_population_placement"
# Only these constructor failures prove that the fixed population itself does
# not fit. Configuration, alignment, policy, and protocol errors must never be
# reclassified as an expected OOM merely because a row carries a declaration.
DIRECT_CAPACITY_OOM_FAILURES = {
    "all-HBM direct placement cannot contain the canonical address space and reservation": "hbm_population_exceeds_physical_capacity",
    "direct placement exceeds physical HBM capacity": (
        "hbm_placement_exceeds_physical_capacity"
    ),
    "direct HBF image and mapping pages exceed raw capacity": (
        "hbf_image_and_mapping_exceed_raw_capacity"
    ),
    "canonical address space exceeds the backing capacity": "backing_capacity_exceeded",
    "HBM-fronted HBF image and mapping pages exceed raw capacity": "hbf_backing_image_and_mapping_exceed_raw_capacity",
}


class FixedFootprintExperimentError(ValueError):
    """The shared fixed-footprint experiment contract is invalid."""


def _fail(message: str) -> NoReturn:
    raise FixedFootprintExperimentError(message)


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{description} must be an object")
    return dict(value)


def _array(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{description} must be an array")
    return value


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{description} must be an integer >= {minimum}")
    return value


def _text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{description} must be non-empty text")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, description: str) -> dict[str, Any]:
    candidate = path.resolve()
    if not candidate.is_file() or candidate.is_symlink():
        _fail(f"{description} must be a regular non-symlink file: {candidate}")
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": _sha256(candidate),
    }


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        return load_json_object(path, description)
    except HBServeError as error:
        _fail(f"cannot read {description} {path}: {error}")


def _root_path(value: Any, description: str, base: Path) -> Path:
    text = _text(value, description)
    path = Path(text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()



@dataclass(frozen=True)
class TopologyPlan:
    raw: Mapping[str, Any]
    id: str
    label: str
    composition: str
    integration_mode: str
    hbm_stacks: int
    hbf_stacks: int
    external_kind: str | None
    measurement_boundary: Mapping[str, Any] | None
    system_config: ResolvedSystemConfig
    expected_outcome: str
    expected_outcome_reason: str | None
    capacity_oom_validation: Mapping[str, Any] | None


@dataclass(frozen=True)
class ExperimentContext:
    experiment_path: Path
    experiment: Mapping[str, Any]
    layout: Any
    population: Mapping[str, Any]
    trace: FixedFootprintTrace
    topologies: tuple[TopologyPlan, ...]


def load_experiment_context(
    experiment_path: Path = DEFAULT_EXPERIMENT,
    *,
    simulator_path: Path | None = None,
) -> ExperimentContext:
    experiment_path = experiment_path.resolve()
    experiment = _read_json(experiment_path, "fixed-footprint experiment")
    if experiment.get("schema") != EXPERIMENT_SCHEMA:
        _fail("fixed-footprint experiment schema is unsupported")
    thermal_contract = _mapping(
        experiment.get("thermal"), "experiment thermal contract"
    )
    if (
        thermal_contract.get("reference_start_state")
        != "throttle_ceiling_steady_state_serving_excerpt"
    ):
        _fail(
            "experiment thermal contract must declare the steady-state "
            "serving-excerpt ceiling start"
        )
    _artifact(
        _root_path(
            thermal_contract.get("start_state_overlay"),
            "thermal start-state overlay",
            experiment_path.parent,
        ),
        "thermal start-state overlay",
    )
    population_config = _mapping(
        experiment.get("population"), "experiment population"
    )
    model_path = _root_path(
        population_config.get("model_descriptor"), "population model descriptor", experiment_path.parent
    )
    layout, population = build_explicit_fixed_population(
        model_descriptor_path=model_path,
        target_population_bytes=_integer(
            population_config.get("target_population_bytes"),
            "population target bytes",
            minimum=1,
        ),
        runtime_overhead_bytes=_integer(
            population_config.get("runtime_overhead_bytes"),
            "population runtime overhead bytes",
            minimum=1,
        ),
    )
    workload = _mapping(experiment.get("workload"), "experiment workload")
    if "inference_source" in workload:
        source = _mapping(workload["inference_source"], "inference source")
        source["artifact"] = str(_root_path(
            source.get("artifact"), "inference source artifact", experiment_path.parent
        ))
        workload["inference_source"] = source
    trace = build_fixed_footprint_trace(
        layout=layout,
        population=population,
        workload=workload,
    )
    mapping = dict(_mapping(experiment.get("mapping"), "experiment mapping"))
    direct = dict(_mapping(mapping.get("direct_placement", {}), "direct placement"))
    if "profile" in direct:
        direct["profile"] = str(_root_path(direct["profile"], "training profile", experiment_path.parent))
    mapping["direct_placement"] = direct
    experiment = dict(experiment) | {"mapping": mapping}

    raw_topologies = _array(
        experiment.get("topologies"), "experiment topologies"
    )
    topology_ids = tuple(
        _text(_mapping(item, "topology").get("id"), "topology id")
        for item in raw_topologies
    )
    if not topology_ids or len(set(topology_ids)) != len(topology_ids):
        _fail("experiment topology ids must be non-empty and unique")
    slots = _mapping(experiment.get("physical_slots"), "physical slots")
    slot_count = _integer(slots.get("slot_count"), "physical slot count", minimum=1)
    topologies: list[TopologyPlan] = []
    for index, raw_value in enumerate(raw_topologies):
        raw = dict(_mapping(raw_value, f"topology {index}"))
        if "mapping" in raw:
            overrides = dict(_mapping(raw["mapping"], "topology mapping"))
            placement = dict(_mapping(overrides.get("direct_placement", {}), "topology direct placement"))
            if "profile" in placement:
                placement["profile"] = str(_root_path(placement["profile"], "training profile", experiment_path.parent))
                overrides["direct_placement"] = placement
            raw["mapping"] = overrides
        config_paths = tuple(
            _root_path(value, f"topology {raw['id']} system config", experiment_path.parent)
            for value in _array(
                raw.get("system_configs"),
                f"topology {raw['id']} system configs",
            )
        )
        config = ResolvedSystemConfig.load(config_paths)
        if raw.get("integration_mode") == "peer_hbm_hbf":
            if simulator_path is None:
                _fail("peer KV preflight requires a simulator to resolve usable HBF capacity and reserves")
            config = config.resolve(simulator_path)
        external_value = raw.get("external_kind")
        external_kind = (
            None
            if external_value is None
            else _text(external_value, f"topology {raw['id']} external kind")
        )
        measurement_boundary_value = raw.get("measurement_boundary")
        measurement_boundary = (
            None
            if measurement_boundary_value is None
            else _mapping(
                measurement_boundary_value,
                f"topology {raw['id']} measurement boundary",
            )
        )
        expected_outcome_value = raw.get("expected_outcome")
        expected_outcome = "execute"
        expected_outcome_reason: str | None = None
        if expected_outcome_value is not None:
            expected_outcome = _text(
                expected_outcome_value,
                f"topology {raw['id']} expected outcome",
            )
            if expected_outcome != "capacity_oom":
                _fail(
                    f"topology {raw['id']} expected_outcome must be exactly "
                    "capacity_oom when declared"
                )
            expected_outcome_reason = _text(
                raw.get("expected_outcome_reason"),
                f"topology {raw['id']} expected outcome reason",
            )
        elif "expected_outcome_reason" in raw:
            _fail(
                f"topology {raw['id']} expected_outcome_reason requires "
                "expected_outcome=capacity_oom"
            )
        plan = TopologyPlan(
            raw=raw,
            id=str(raw["id"]),
            label=_text(raw.get("label"), f"topology {raw['id']} label"),
            composition=_text(
                raw.get("composition"), f"topology {raw['id']} composition"
            ),
            integration_mode=_text(
                raw.get("integration_mode"),
                f"topology {raw['id']} integration mode",
            ),
            hbm_stacks=_integer(
                raw.get("active_hbm_stacks"),
                f"topology {raw['id']} HBM stacks",
            ),
            hbf_stacks=_integer(
                raw.get("active_hbf_stacks"),
                f"topology {raw['id']} HBF stacks",
            ),
            external_kind=external_kind,
            measurement_boundary=measurement_boundary,
            system_config=config,
            expected_outcome=expected_outcome,
            expected_outcome_reason=expected_outcome_reason,
            capacity_oom_validation=None,
        )
        if plan.hbm_stacks + plan.hbf_stacks != slot_count:
            _fail(f"topology {plan.id} does not occupy the declared fixed slots")
        if plan.hbm_stacks and config.integer("hbm-stacks") != plan.hbm_stacks:
            _fail(f"topology {plan.id} HBM stack count differs from its device configuration")
        if plan.integration_mode == "hbm_fronted_external":
            identity = config.external_backing_identity
            if (
                plan.hbf_stacks
                or plan.external_kind != identity["kind"]
                or identity["capacity_bytes"] < layout.address_space_bytes
            ):
                _fail(f"topology {plan.id} external backing is inconsistent")
            if plan.external_kind == "cxl-ssd":
                if (
                    identity.get("device_cache", {}).get("enabled") is not True
                    or plan.measurement_boundary
                    != CXL_SSD_WRITE_COMPLETION_BOUNDARY
                ):
                    _fail(
                        "cached CXL-SSD topology must enable its device DRAM "
                        "cache and declare the internal-DRAM write boundary"
                    )
            elif plan.measurement_boundary is not None:
                _fail(
                    f"topology {plan.id} declares a medium-specific measurement "
                    "boundary but is not cached CXL-SSD"
                )
        elif plan.integration_mode == "all_hbm_upper_bound":
            if (
                plan.hbf_stacks
                or config.hbm_capacity_bytes < layout.address_space_bytes
            ):
                _fail("all-HBM upper bound cannot contain the fixed footprint")
        elif plan.integration_mode == "direct_hbm_hbf" and plan.hbf_stacks == 0:
            if not plan.hbm_stacks or plan.external_kind is not None:
                _fail(f"topology {plan.id} direct HBM-only selection is inconsistent")
        elif plan.integration_mode in {"direct_hbm_hbf", "hbm_fronted_hbf", "peer_hbm_hbf"}:
            if (
                not plan.hbf_stacks
                or config.hbf_geometry.stacks != plan.hbf_stacks
                or plan.external_kind is not None
                or (plan.integration_mode in {"hbm_fronted_hbf", "peer_hbm_hbf"} and not plan.hbm_stacks)
            ):
                _fail(
                    f"topology {plan.id} HBF geometry or backing selection "
                    "is inconsistent"
                )
            thermal = config.hbf_thermal_identity
            if not thermal["enabled"]:
                _fail(
                    f"topology {plan.id} does not enable the core HBF "
                    "thermal model"
                )
            if thermal["start_state"] != "throttle-ceiling":
                _fail(
                    f"topology {plan.id} must enter the window at the "
                    "thermal governor ceiling (steady-state serving "
                    "excerpt boundary)"
                )
        else:
            _fail(f"topology {plan.id} has unsupported integration mode")
        if (
            plan.expected_outcome == "capacity_oom"
            and plan.integration_mode not in {"direct_hbm_hbf", "hbm_fronted_hbf", "peer_hbm_hbf"}
        ):
            _fail(
                f"topology {plan.id} may declare expected_outcome=capacity_oom only "
                "for HBM/HBF initial placement"
            )
        # Constructor validation is the authoritative mapping/capacity check.
        try:
            mapper = new_remapper(
                plan, context_layout=layout, experiment=experiment, trace_sha256=trace.digest,
                logical_trace_sha256=canonical_sha256([phase.trace_group.digest for phase in trace.phases]),
            )
            if isinstance(mapper, PeerKvMigrationRemapper):
                mapper.preflight(phase.trace_group for phase in trace.phases)
        except RemapError as error:
            if plan.expected_outcome != "capacity_oom":
                raise
            failure_kind = DIRECT_CAPACITY_OOM_FAILURES.get(str(error))
            if isinstance(error, PeerCapacityError) and error.code in {
                "hbm_no_kv_slot", "hbf_static_image_capacity_exceeded",
                "static_kv_hbm_capacity_exceeded", "peer_total_kv_capacity_exceeded",
            }:
                failure_kind = error.code
            if failure_kind is None:
                # A declaration never blesses an arbitrary remapper failure.
                raise
            plan = replace(
                plan,
                capacity_oom_validation={
                    "declared": True,
                    "validated": True,
                    "stage": "empty_to_grown_KV_allocation" if isinstance(error, PeerCapacityError) else EXPECTED_CAPACITY_OOM_STAGE,
                    "failure_kind": failure_kind,
                    "failure_message": str(error),
                    "reason": plan.expected_outcome_reason,
                },
            )
        else:
            if plan.expected_outcome == "capacity_oom":
                _fail(
                    f"topology {plan.id} declares expected_outcome=capacity_oom but "
                    "its fixed population fits the execution path"
                )
        topologies.append(plan)
    return ExperimentContext(
        experiment_path=experiment_path,
        experiment=experiment,
        layout=layout,
        population=population,
        trace=trace,
        topologies=tuple(topologies),
    )


def new_remapper(
    topology: TopologyPlan,
    *,
    context_layout: Any,
    experiment: Mapping[str, Any],
    trace_sha256: str = "",
    logical_trace_sha256: str | None = None,
) -> DirectAttachedRemapper | HbmFrontedBackingRemapper | PeerKvMigrationRemapper:
    mapping = dict(_mapping(experiment.get("mapping"), "experiment mapping"))
    mapping.update(_mapping(topology.raw.get("mapping", {}), "topology mapping overrides"))
    if topology.integration_mode == "peer_hbm_hbf":
        if _mapping(experiment.get("workload"), "workload").get("window_shape") != "prefill_growth":
            _fail("peer KV requires an empty-to-grown prefill window, not preinstalled decode KV")
        peer = _mapping(mapping.get("peer_kv"), "peer KV policy")
        region = context_layout.region(context_layout.kv_region_id)
        capacity = topology.system_config.logical_hbf_capacity_bytes
        if capacity is None:
            _fail("peer KV requires native resolved HBF logical capacity")
        return PeerKvMigrationRemapper(
            address_space_bytes=context_layout.address_space_bytes,
            hbm_capacity_bytes=topology.system_config.hbm_capacity_bytes,
            migration_granularity_bytes=_integer(peer.get("migration_granularity_bytes"), "peer KV granularity", minimum=4096),
            transfer_chunk_bytes=_integer(peer.get("transfer_chunk_bytes"), "peer transfer chunk", minimum=4096),
            hbf_geometry=topology.system_config.hbf_geometry,
            hbf_logical_capacity_bytes=capacity,
            kv_range=(region.begin, region.end), hbm_stacks=topology.hbm_stacks,
            hbf_stacks=topology.hbf_stacks, policy=_text(peer.get("policy"), "peer KV policy"),
            hbf_mapping_mode=topology.system_config.values["hbf-mapping-mode"],
        )
    if topology.integration_mode in {"all_hbm_upper_bound", "direct_hbm_hbf"}:
        granularity = _integer(mapping.get("placement_granularity_bytes"), "direct placement granularity", minimum=4096)
        order, detail = placement_order(
            context_layout, mapping.get("direct_placement", {}), granularity, trace_sha256,
            logical_trace_sha256=logical_trace_sha256,
        )
        remapper = DirectAttachedRemapper(
            address_space_bytes=context_layout.address_space_bytes,
            hbm_capacity_bytes=(
                topology.system_config.hbm_capacity_bytes
                if topology.hbm_stacks
                else 0
            ),
            hbf_geometry=topology.system_config.hbf_geometry,
            hbm_stacks=topology.hbm_stacks,
            hbf_stacks=topology.hbf_stacks,
            placement_granularity_bytes=granularity,
            hbm_priority_units=order,
        )
        if order is not None:
            remapper.policy_name = "static_" + str(mapping["direct_placement"]["policy"])
            remapper.policy_detail = detail
        return remapper
    hbf_backing = topology.integration_mode == "hbm_fronted_hbf"
    cache_policy = _mapping(
        mapping.get("hbf_tiering" if hbf_backing else "external_offload"), "HBM-fronted cache policy"
    )
    policy = _text(cache_policy.get("policy", "address_only_lru"), "HBM-fronted policy")
    backing = {"backing_kind": "hbf", "hbf_geometry": topology.system_config.hbf_geometry}
    if not hbf_backing:
        identity = topology.system_config.external_backing_identity
        backing = {"backing_kind": "external", "external_capacity_bytes": int(identity["capacity_bytes"]),
                   "external_page_size_bytes": int(identity["page_size_bytes"])}
    return HbmFrontedBackingRemapper(
        address_space_bytes=context_layout.address_space_bytes,
        hbm_capacity_bytes=topology.system_config.hbm_capacity_bytes,
        migration_granularity_bytes=_integer(
            cache_policy.get("migration_granularity_bytes"),
            "external migration granularity",
            minimum=4096,
        ),
        transfer_chunk_bytes=_integer(
            cache_policy.get("transfer_chunk_bytes"),
            "external transfer chunk",
            minimum=4096,
        ),
        read_ahead_window_bytes=_integer(
            cache_policy.get("read_ahead_window_bytes"),
            "external read-ahead window",
            minimum=4096,
        ),
        reserved_hbm_bytes=_integer(cache_policy.get("reserved_hbm_bytes", 0), "reserved HBM bytes"),
        policy=policy,
        kv_priority_ranges=(tuple((region.begin, region.end) for region in context_layout.regions
                                  if region.placement_class == "kv") if policy == "class_aware" else None),
        **backing,
    )


def _topology_preflight(
    context: ExperimentContext, topology: TopologyPlan
) -> dict[str, Any]:
    remapper = (
        None
        if topology.capacity_oom_validation is not None
        else new_remapper(
            topology,
            context_layout=context.layout,
            experiment=context.experiment,
            trace_sha256=context.trace.digest,
            logical_trace_sha256=canonical_sha256([phase.trace_group.digest for phase in context.trace.phases]),
        )
    )
    slots = _mapping(
        context.experiment.get("physical_slots"), "physical slots"
    )
    physical_hbm = topology.hbm_stacks * _integer(
        slots.get("hbm_bytes_per_stack"), "HBM bytes per stack", minimum=1
    )
    physical_hbf = topology.hbf_stacks * _integer(
        slots.get("hbf_raw_bytes_per_stack"), "HBF bytes per stack", minimum=1
    )
    external_capacity = 0
    if topology.integration_mode == "hbm_fronted_external":
        external_capacity = int(
            topology.system_config.external_backing_identity["capacity_bytes"]
        )
    if remapper is None:
        placement = None
    elif isinstance(remapper, DirectAttachedRemapper):
        placement = {
            "kind": "direct_residency",
            "hbm_payload_bytes": remapper.hbm_resident_payload_bytes,
            "hbf_payload_bytes": remapper.hbf_resident_payload_bytes,
            "external_backing_bytes": 0,
        }
    elif isinstance(remapper, PeerKvMigrationRemapper):
        placement = remapper.preflight(phase.trace_group for phase in context.trace.phases)
        placement.update(kind="exclusive_peer_KV_born_on_write", hbm_payload_bytes=0,
                         hbf_payload_bytes=remapper.static_flash_bytes, external_backing_bytes=0)
    else:
        placement = {
            "kind": f"complete_{remapper.backing_kind}_backing_with_empty_hbm_cache",
            "hbm_cache_capacity_bytes": remapper.cache_slots * remapper.granularity,
            "hbm_stream_staging_bytes": remapper.stream_staging_bytes,
            "reserved_hbm_bytes": remapper.reserved_hbm_bytes,
            "hbf_payload_bytes": context.layout.address_space_bytes if remapper.backing_kind == "hbf" else 0,
            "external_backing_bytes": context.layout.address_space_bytes if remapper.backing_kind == "external" else 0,
            "policy": remapper.policy,
            "capacity_semantics": "inclusive_backing_plus_duplicate_HBM_cache_not_additive",
        }
    return {
        "id": topology.id,
        "label": topology.label,
        "composition": topology.composition,
        "integration_mode": topology.integration_mode,
        "role": (
            "timing_upper_bound"
            if topology.integration_mode == "all_hbm_upper_bound"
            else "no_hbf_baseline"
            if topology.hbf_stacks == 0
            else "hbf_topology_candidate"
        ),
        "active_hbm_stacks": topology.hbm_stacks,
        "active_hbf_stacks": topology.hbf_stacks,
        "physical_capacity_bytes": {
            "hbm": physical_hbm,
            "hbf_raw": physical_hbf,
            "external": external_capacity,
        },
        "effective_hbm_address_guard_bytes": (
            topology.system_config.hbm_capacity_bytes
            if topology.hbm_stacks
            else 0
        ),
        "population_bytes": context.layout.address_space_bytes,
        "population_fits_execution_path": remapper is not None,
        "expected_outcome": topology.expected_outcome,
        "capacity_oom_validation": (
            dict(topology.capacity_oom_validation)
            if topology.capacity_oom_validation is not None
            else None
        ),
        "initial_placement": placement,
        "initial_hbf_logical_pages": (
            remapper.initial_hbf_logical_pages
            if remapper is not None
            else None
        ),
        "external_backing": (
            dict(topology.system_config.external_backing_identity)
            if topology.integration_mode == "hbm_fronted_external"
            else None
        ),
        "measurement_boundary": (
            dict(topology.measurement_boundary)
            if topology.measurement_boundary is not None
            else None
        ),
        "hbf_mapping_mode": (
            topology.system_config.hbf_mapping_mode
            if topology.hbf_stacks
            else None
        ),
        "hbf_controller_dram_policy": (
            {
                "budget_bytes": topology.system_config.hbf_ctrl_dram_bytes,
                "budget_bytes_per_stack": (
                    topology.system_config.hbf_ctrl_dram_bytes
                    // topology.hbf_stacks
                ),
                "budget_basis": "resolved_device_configuration",
                "fraction_of_raw_capacity": topology.system_config.hbf_ctrl_dram_bytes / topology.system_config.hbf_geometry.capacity_bytes,
                "charges": [
                    "mapping_page_directory",
                    "data_write_buffer",
                    "remaining_mapping_cache",
                ],
            }
            if topology.hbf_stacks
            else None
        ),
        "hbf_thermal": (
            topology.system_config.hbf_thermal_identity
            if topology.hbf_stacks
            else None
        ),
        "system_configs": list(topology.system_config.artifacts),
        "claim_boundary": topology.raw.get("claim_boundary"),
    }


def _thermal_boundary_axis_preflight(
    context: ExperimentContext,
) -> dict[str, Any]:
    """Resolve every declared thermal boundary point and fail on divergence.

    The axis is a contract, not prose: each point's overlay is composed
    onto a reference HBF row and the resolved thermal identity must equal
    the declared boundary/neighbor values, so the registered sweep stays
    executable and its published derate ladder cannot drift from the
    configs.
    """

    thermal_contract = _mapping(
        context.experiment.get("thermal"), "experiment thermal contract"
    )
    axis = _mapping(
        thermal_contract.get("boundary_temperature_axis"),
        "thermal boundary-temperature axis",
    )
    base = next((topology for topology in context.topologies if topology.hbf_stacks), None)
    if base is None:
        return {"mechanism": axis.get("mechanism"), "validated_against_topology": None, "points": []}
    points: list[dict[str, Any]] = []
    for index, raw_point in enumerate(
        _array(axis.get("points"), "boundary axis points")
    ):
        point = _mapping(raw_point, f"boundary axis point {index}")
        declared_boundary = float(point.get("boundary_c"))
        declared_neighbor = float(point.get("neighbor_heat_c"))
        paths = list(base.system_config.paths)
        overlay = point.get("overlay")
        if overlay is not None:
            overlay_path = _root_path(
                overlay, f"boundary axis point {index} overlay", context.experiment_path.parent
            )
            _artifact(overlay_path, f"boundary axis point {index} overlay")
            paths.append(overlay_path)
        identity = ResolvedSystemConfig.load(tuple(paths)).hbf_thermal_identity
        if (
            not identity["enabled"]
            or abs(identity["neighbor_heat_c"] - declared_neighbor) > 1e-9
            or abs(
                identity["boundary_temperature_c"] - declared_boundary
            ) > 1e-9
        ):
            _fail(
                f"boundary axis point {index} resolves to "
                f"{identity.get('boundary_temperature_c')} C, not the "
                f"declared {declared_boundary} C"
            )
        points.append(
            {
                "boundary_c": declared_boundary,
                "neighbor_heat_c": declared_neighbor,
                "overlay": overlay,
                "basis": point.get("basis"),
                "resolved": {
                    "boundary_temperature_c": identity[
                        "boundary_temperature_c"
                    ],
                    "pacing_power_w": identity["pacing_power_w"],
                    "sustainable_read_GBps_per_stack": identity[
                        "sustainable_read_GBps_per_stack"
                    ],
                },
            }
        )
    return {
        "mechanism": _text(axis.get("mechanism"), "boundary axis mechanism"),
        "validated_against_topology": base.id,
        "points": points,
    }


def _preflight_from_context(context: ExperimentContext) -> dict[str, Any]:
    trace_receipt = context.trace.receipt()
    return {
        "schema": PREFLIGHT_SCHEMA,
        "execution_mode": "fixed_window",
        "result": "pass",
        "experiment_id": context.experiment.get("experiment_id"),
        "experiment": _artifact(
            context.experiment_path, "fixed-footprint experiment"
        ),
        "population": dict(context.population),
        "trace": trace_receipt,
        "topologies": [
            _topology_preflight(context, topology)
            for topology in context.topologies
        ],
        "thermal_boundary_axis": _thermal_boundary_axis_preflight(context),
        "invariants": {
            "comparison_unit": "complete_topology",
            "topology_rows": [
                topology.id for topology in context.topologies
            ],
            "same_population_for_every_topology": True,
            "same_trace_for_every_topology": True,
            "capacity_oom_rows": [
                topology.id
                for topology in context.topologies
                if topology.capacity_oom_validation is not None
            ],
            "declared_capacity_oom_is_exactly_validated": True,
            "undeclared_or_non_capacity_remapper_failure_is_fatal": True,
            "layer_ordered_trace": True,
            "kv_storage_order": "layer_major",
            "window_shape": context.trace.window_shape,
            "inference_KV_writes_present": True,
            "all_hbm_is_capacity_relaxed_upper_bound": any(topology.integration_mode == "all_hbm_upper_bound" for topology in context.topologies),
            "no_hbf_baselines": [
                topology.id
                for topology in context.topologies
                if topology.integration_mode == "hbm_fronted_external"
            ],
            "cached_cxl_ssd_write_completion_boundary": (
                "device_internal_dram_acceptance_with_nand_destage_excluded"
                if any(
                    topology.external_kind == "cxl-ssd"
                    for topology in context.topologies
                )
                else None
            ),
            "hbf_mapping_modes": {topology.id: topology.system_config.hbf_mapping_mode
                                  for topology in context.topologies if topology.hbf_stacks},
            "hbf_thermal_model": (
                "core_per_stack_lumped_RC_with_firmware_pacing"
            ),
            "hbf_rows_enter_at_thermal_governor_ceiling": True,
            "thermal_boundary_axis_overlays_resolve_to_declared_values": (
                True
            ),
            "single_completion_metric": "final_drain_time_ns",
        },

    }


def build_preflight(
    experiment_path: Path = DEFAULT_EXPERIMENT,
    *,
    simulator_path: Path | None = None,
) -> dict[str, Any]:
    return _preflight_from_context(
        load_experiment_context(experiment_path, simulator_path=simulator_path)
    )


def execute_topology(
    context: ExperimentContext,
    topology: TopologyPlan,
    *,
    simulator_path: Path,
) -> dict[str, Any]:
    if topology.capacity_oom_validation is not None:
        _fail(
            f"topology {topology.id} is a validated expected capacity OOM "
            "row and cannot execute"
        )
    remapper = new_remapper(
        topology,
        context_layout=context.layout,
        experiment=context.experiment,
        trace_sha256=context.trace.digest,
        logical_trace_sha256=canonical_sha256([phase.trace_group.digest for phase in context.trace.phases]),
    )
    session = SimulationSession(
        simulator_path=simulator_path,
        system_config=topology.system_config,
        enable_hbm=topology.hbm_stacks > 0,
        enable_hbf=topology.hbf_stacks > 0,
        enable_external=topology.integration_mode == "hbm_fronted_external",
        hbm_capacity_bytes=(
            topology.system_config.hbm_capacity_bytes
            if topology.hbm_stacks
            else 0
        ),
        initial_hbf_logical_first_lpn=remapper.initial_hbf_logical_first_lpn,
        initial_hbf_logical_pages=remapper.initial_hbf_logical_pages,
    )
    mapped_receipts: list[Mapping[str, Any]] = []
    phase_measurements: list[dict[str, Any]] = []
    host_start = time.perf_counter()
    try:
        for phase in context.trace.phases:
            mapped = remapper.remap(phase.trace_group)
            completion = session.submit(mapped)
            mapped_receipts.append(mapped.receipt)
            phase_measurements.append(
                {
                    "phase_id": phase.id,
                    "stage": phase.stage,
                    "layer": phase.layer,
                    "object_class": phase.object_class,
                    "objects": list(phase.objects),
                    "source_logical_read_bytes": phase.read_bytes,
                    "source_logical_write_bytes": phase.write_bytes,
                    "completion": completion,
                }
            )
        final_batch = remapper.finalize()
        if final_batch is not None:
            completion = session.submit(final_batch)
            mapped_receipts.append(final_batch.receipt)
            phase_measurements.append(
                {
                    "phase_id": "kv_final_backing_flush",
                    "stage": "final_backing_flush",
                    "layer": None,
                    "object_class": "kv_cache",
                    "objects": [context.layout.kv_region_id],
                    "source_logical_read_bytes": 0,
                    "source_logical_write_bytes": 0,
                    "completion": completion,
                }
            )
    finally:
        session.close()
    host_seconds = time.perf_counter() - host_start
    source = session.source_receipt()
    final_measurement = _mapping(
        source.get("final_measurement"), "simulation final measurement"
    )
    metrics = build_fixed_footprint_metrics(
        final_measurement=final_measurement,
        logical_read_bytes=context.trace.read_bytes,
        logical_write_bytes=context.trace.write_bytes,
        phase_measurements=phase_measurements,
        external_kind=topology.external_kind,
        active_hbf_stacks=topology.hbf_stacks,
        thermal_identity=(
            topology.system_config.hbf_thermal_identity
            if topology.hbf_stacks
            else None
        ),
    )
    return {
        "id": topology.id,
        "label": topology.label,
        "composition": topology.composition,
        "integration_mode": topology.integration_mode,
        "active_hbm_stacks": topology.hbm_stacks,
        "active_hbf_stacks": topology.hbf_stacks,
        "measurement_boundary": (
            dict(topology.measurement_boundary)
            if topology.measurement_boundary is not None
            else None
        ),
        "trace_sha256": context.trace.digest,
        "host_execution_seconds": host_seconds,
        "metrics": metrics,
        "remap_receipts": mapped_receipts,
        "simulation_session": source,
    }


def _medium_comparison_row(
    raw: Mapping[str, Any], *, active_stacks: int | None
) -> dict[str, Any]:
    transactions = _mapping(raw.get("transactions"), "medium transactions")
    read = _mapping(transactions.get("read"), "medium read transactions")
    write = _mapping(transactions.get("write"), "medium write transactions")
    read_bytes = int(raw["read_bytes"])
    write_bytes = int(raw["write_bytes"])
    total_bytes = read_bytes + write_bytes

    def per_stack(value: int | float) -> float | None:
        return value / active_stacks if active_stacks else None

    return {
        "label": raw.get("label"),
        "active_stacks": active_stacks,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total_bytes": total_bytes,
        "request_read_operations": int(read["request_transactions"]),
        "request_write_operations": int(write["request_transactions"]),
        "media_read_operations": raw.get("media_read_operations"),
        "media_write_operations": raw.get("media_write_operations"),
        "mean_read_latency_ns": read.get("mean_latency_ns"),
        "mean_write_latency_ns": write.get("mean_latency_ns"),
        "read_throughput_GBps": raw.get("read_throughput_GBps"),
        "write_throughput_GBps": raw.get("write_throughput_GBps"),
        "total_throughput_GBps": raw.get("total_throughput_GBps"),
        "average_read_bytes_per_active_stack": per_stack(read_bytes),
        "average_write_bytes_per_active_stack": per_stack(write_bytes),
        "average_total_bytes_per_active_stack": per_stack(total_bytes),
    }


def _object_comparison_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw_value in _array(
        metrics.get("object_distribution"), "object distribution"
    ):
        raw = _mapping(raw_value, "object distribution row")
        medium = _mapping(raw.get("medium"), "object distribution media")
        by_medium: dict[str, Any] = {}
        for name in ("hbm", "hbf", "external", "interconnect"):
            operations = _mapping(medium.get(name), f"object {name} metrics")
            read = _mapping(operations.get("read"), f"object {name} read")
            write = _mapping(operations.get("write"), f"object {name} write")
            by_medium[name] = {
                "mapped_read_bytes": int(read["mapped_physical_bytes"]),
                "mapped_write_bytes": int(write["mapped_physical_bytes"]),
                "read_operations": int(read["request_transactions"]),
                "write_operations": int(write["request_transactions"]),
                "mean_read_latency_ns": read.get("mean_latency_ns"),
                "mean_write_latency_ns": write.get("mean_latency_ns"),
            }
        result.append(
            {
                "object_class": raw["object_class"],
                "objects": list(raw.get("objects", [])),
                "source_logical_read_bytes": int(
                    raw["source_logical_read_bytes"]
                ),
                "source_logical_write_bytes": int(
                    raw["source_logical_write_bytes"]
                ),
                "by_medium": by_medium,
            }
        )
    return result


def build_topology_table(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    final_drain_time = {
        str(row["id"]): float(
            _mapping(row["metrics"], "row metrics")["final_drain_time_ns"]
        )
        for row in rows
    }
    # Comparison ratios exist only against reference rows that actually
    # executed: a subset run (--topologies) omits the ratios whose
    # denominator row is absent instead of inventing them.
    all_hbm = final_drain_time.get("all-hbm")
    dram = final_drain_time.get("8h0f-dram")
    ssd = final_drain_time.get("8h0f-ssd")
    cxl_ssd = final_drain_time.get(CXL_SSD_TOPOLOGY_ID)
    table: list[dict[str, Any]] = []
    for row in rows:
        metrics = _mapping(row["metrics"], "row metrics")
        media = _mapping(metrics.get("media"), "row media metrics")
        hbm_stacks = int(row["active_hbm_stacks"])
        hbf_stacks = int(row["active_hbf_stacks"])
        current = final_drain_time[str(row["id"])]
        comparisons: dict[str, Any] = {}
        if all_hbm is not None:
            comparisons["slowdown_vs_all_hbm"] = current / all_hbm
        if dram is not None:
            comparisons["speedup_vs_8h0f_dram"] = dram / current
        if ssd is not None:
            comparisons["speedup_vs_8h0f_ssd"] = ssd / current
        if cxl_ssd is not None:
            comparisons["speedup_vs_8h0f_cxl_ssd"] = cxl_ssd / current
        table.append(
            {
                "id": row["id"],
                "composition": row["composition"],
                "active_hbm_stacks": hbm_stacks,
                "active_hbf_stacks": hbf_stacks,
                "final_drain_time_ns": current,
                "measurement_boundary": row.get("measurement_boundary"),
                **comparisons,
                "effective_trace_throughput_GBps": _mapping(
                    metrics.get("source_logical_traffic"),
                    "source logical traffic",
                )["effective_trace_throughput_GBps"],
                "media": {
                    "hbm": _medium_comparison_row(
                        _mapping(media.get("hbm"), "HBM metrics"),
                        active_stacks=hbm_stacks,
                    ),
                    "hbf": _medium_comparison_row(
                        _mapping(media.get("hbf"), "HBF metrics"),
                        active_stacks=hbf_stacks,
                    ),
                    "external": _medium_comparison_row(
                        _mapping(media.get("external"), "external metrics"),
                        active_stacks=None,
                    ),
                    "interconnect": _medium_comparison_row(
                        _mapping(
                            media.get("interconnect"),
                            "interconnect metrics",
                        ),
                        active_stacks=None,
                    ),
                },
                "objects": _object_comparison_rows(metrics),
                "thermal": _mapping(
                    metrics.get("thermal"), "thermal governor telemetry"
                ),
                "auxiliary_media_traffic": _mapping(
                    metrics.get("media_traffic"), "media traffic accounting"
                ),
            }
        )
    return table


def run_reference_experiment(
    *,
    experiment_path: Path = DEFAULT_EXPERIMENT,
    simulator_path: Path = DEFAULT_SIMULATOR,
    topology_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    context = load_experiment_context(experiment_path, simulator_path=simulator_path)
    # The experiment config always declares the complete reference topology
    # set (that contract is unchanged); topology_ids only scopes which rows
    # this invocation executes, so bounds or single rows can run as their
    # own processes.
    executed = context.topologies
    if topology_ids is not None:
        if not topology_ids or len(set(topology_ids)) != len(topology_ids):
            _fail("selected topology ids must be non-empty and unique")
        known = {topology.id: topology for topology in context.topologies}
        missing = [name for name in topology_ids if name not in known]
        if missing:
            _fail(
                "unknown topology id(s) requested: " + ", ".join(missing)
            )
        executed = tuple(known[name] for name in topology_ids)
    blocked = [
        topology.id
        for topology in executed
        if topology.capacity_oom_validation is not None
    ]
    if blocked:
        _fail(
            "selected topology id(s) are validated expected capacity OOM "
            "rows and cannot execute: " + ", ".join(blocked)
        )
    if simulator_path is None:
        _fail("an external simulator_path is required for execution")
    simulator = simulator_path.resolve()
    simulator_artifact = _artifact(simulator, "HBFSim executable")
    preflight = _preflight_from_context(context)
    rows: list[dict[str, Any]] = []
    for topology in executed:
        print(
            json.dumps(
                {
                    "event": "reference_topology_start",
                    "topology": topology.id,
                    "trace_sha256": context.trace.digest,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        row = execute_topology(
            context, topology, simulator_path=simulator
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "reference_topology_complete",
                    "topology": topology.id,
                    "final_drain_time_ns": row["metrics"][
                        "final_drain_time_ns"
                    ],
                    "host_execution_seconds": row[
                        "host_execution_seconds"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if {row["trace_sha256"] for row in rows} != {context.trace.digest}:
        _fail("topology rows did not execute one common trace")
    return {
        "schema": RESULT_SCHEMA,
        "execution_mode": "fixed_window",
        "result": "pass",
        "experiment_id": context.experiment.get("experiment_id"),
        "runner": _artifact(Path(__file__), "fixed-footprint runner"),
        "simulator": simulator_artifact,
        "preflight": preflight,
        "trace": context.trace.receipt(),
        "reference_topology_results": rows,
        "topology_comparison": {
                "schema": TOPOLOGY_VIEW_SCHEMA,
                "result": "pass",
                "varied_variable": "declared_topology_and_mapping_configuration",
                "held_fixed": [
                    "population",
                    "logical_trace",
                    "completion_boundary",
                ],
                "comparison_table": build_topology_table(rows),
        },
        "interpretation": {
            "artifact_role": (
                "matched_fixed_window_topology_comparison;_"
                "memory_only_not_closed_loop_serving"
            ),
            "comparison_unit": "complete_topology",
            "workload_control": (
                "one_byte_identical_fixed_memory_trace_digest_for_"
                "every_row"
            ),
            "measurement_start": preflight["trace"]["measurement_window"]["start"],
            "read_locality": (
                "sequential_weight_streams_plus_sequential_and_paged_random_"
                "KV_reads_plus_indexed_metadata_and_embedding_reads"
            ),
            "write_scope": (
                "trace_supported_KV_appends_and_block_table_allocation_only;_"
                "uncalibrated_activation_scratch_traffic_excluded"
            ),
            "HBF_cell_model": (
                "SLC_one_physical_4KiB_page_per_program;_no_MLC_TLC_QLC"
            ),
            "reference_HBF_mapping_policy": (
                "mapping_mode_and_controller_budget_are_reported_per_topology"
            ),
            "hbf_thermal_boundary": (
                "core_lumped_RC_pacing_governor;_HBF_rows_enter_at_the_"
                "governor_ceiling_because_the_window_is_a_sustained_serving_"
                "excerpt;_thermal_cold_start_is_not_executed_here"
            ),
            "endurance_scope": (
                "finite_memory_window_only;_no_lifetime_projection"
            ),
            "all_hbm": "capacity_relaxed_timing_upper_bound",
            "8h0f_dram": "physical_no_hbf_overflow_baseline",
            "8h0f_ssd": "physical_no_hbf_overflow_baseline",
            "8h0f_cxl_ssd": (
                "cached_cxl_ssd_overflow_baseline_whose_writes_complete_after_"
                "device_internal_dram_acceptance;_later_nand_destage_is_excluded"
            ),
            "final_drain_time_ns": (
                "trace_start_through_caller_visible_completion_and_required_"
                "HBF_drain;_cached_CXL_SSD_writes_stop_at_device_internal_DRAM_"
                "acceptance_and_exclude_later_NAND_destage"
            ),
            "effective_trace_throughput_GBps": (
                "fixed_source_logical_bytes_divided_by_final_drain_time"
            ),
            "average_per_active_stack_traffic": (
                "medium_bytes_divided_by_active_stack_count_not_a_"
                "measurement_of_max_stack_imbalance"
            ),
            "auxiliary_media_traffic": (
                "actual_HBM_HBF_external_media_bytes_divided_by_source_"
                "logical_bytes"
            ),
        },
    }

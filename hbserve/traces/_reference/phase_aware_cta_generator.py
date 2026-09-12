#!/usr/bin/env python3
"""Generate compact CTA-x slices from a phase-aware request-range sidecar.

The compact request format intentionally remains unchanged::

    <object:u16, offset:u32, kernel:u16, bytes:u16, op:u8, flags:u8>

CTA/warp ownership is carried by a separate sidecar.  A generator plan may
either translate one captured CTA-x slice with held-out affine/periodic rules,
translate a validated tiled-swizzle grid including its phase-dependent CTA-y
tail mask, expand a validated causal tile loop, or copy an explicitly captured
target slice as an exact-anchor fallback.

This module is a bounded prototype for integration into
``stream_lazy_trace_plan.py``.  It does not claim production CTA scheduling or
per-request issue timing; output bundles use a deterministic legal order.
"""

from __future__ import annotations

from hbserve.traces.artifacts import resolve_input

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from hbserve.traces._reference.compact_request_template import (
    OPERATION_NAME,
    RECORD_BYTES,
    RECORD_STRUCT,
    known_request_flags,
)
from hbserve.traces._reference.shape_aware_cta_rules import (
    canonical_sha256,
    coordinate_from_validated_rule,
    coordinate_period_ctas,
    validate_rule as validate_shape_rule,
)
from hbserve.traces._reference.semantic_embedding_generator import (
    POLICY_KIND as SEMANTIC_EMBEDDING_KIND,
    iter_cta_records as iter_semantic_embedding_cta_records,
    range_census as semantic_embedding_range_census,
    validate_policy as validate_semantic_embedding_policy,
)


SIDECAR_SCHEMA = {
    "name": "hbfsim.phase_aware_compact_bundle_sidecar",
    "version": 1,
}
PLAN_SCHEMA = {"name": "hbfsim.phase_aware_cta_generator_plan", "version": 1}
OUTPUT_SCHEMA = {
    "name": "hbfsim.phase_aware_generated_compact_requests",
    "version": 1,
}
SEGMENT_DESCRIPTOR_SCHEMA = {
    "name": "hbfsim.phase_aware_cta_segment_generator",
    "version": 1,
}
SEGMENT_GENERATOR_KIND = "phase_aware_cta_v1"
VALIDATED_RULE_STATUSES = {"PASS_HELDOUT", "PASS_EXACT_DERIVATION"}
SHADOW_RULE_STATUS = "PASS_RETROSPECTIVE_SUPPLEMENTAL_SHADOW"
SHADOW_PLAN_CLASSIFICATION = (
    "retrospective supplemental executable shape/lifetime shadow"
)
SHADOW_CONTRACT_SCHEMA = {
    "name": "hbfsim.prefill_shadow_runtime_binding",
    "version": 1,
}
SHADOW_REPAIR_SCHEMA = {
    "name": "hbfsim.prefill_full_grid_shadow_repair",
    "version": 1,
}
VALIDATED_SHAPE_RULE_STATUS = "PASS_PROSPECTIVE_FRESH_HOLDOUT"
VALIDATED_SHAPE_PLAN_CLASSIFICATION = (
    "prospectively validated supplemental executable shape-rule plan"
)
VALIDATED_SHAPE_CONTRACT_SCHEMA = {
    "name": "hbfsim.prefill_validated_shape_runtime_binding",
    "version": 1,
}
VALIDATED_SHAPE_SOURCE_PLAN_SCHEMA = {
    "name": "hbfsim.prefill_shape_aware_shadow_plan",
    "version": 1,
}
VALIDATED_SHAPE_EVALUATION_SCHEMA = {
    "name": "hbfsim.prefill_shape_aware_fresh_holdout_evaluation",
    "version": 1,
}
QWEN_PROBE_SCHEMA = {"name": "hbfsim.qwen_nvbit_layer_probe", "version": 1}


class PhaseAwareGenerationError(ValueError):
    """Raised when a phase-aware rule cannot be applied without guessing."""


@dataclass(frozen=True)
class PreparedPhaseGenerator:
    """Validated, reusable view of one compact-template generation program."""

    source_extents: list[int]
    target_extents: list[int]
    object_metadata: list[dict[str, Any]]
    policies: dict[int, dict[str, Any]]
    by_kernel_x: dict[tuple[int, int], list[dict[str, int]]]
    shadow_runtime_binding: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedSegmentGenerator:
    """One digest-bound generator plus the target CTA-x extent per kernel."""

    generator: PreparedPhaseGenerator
    ranges: list[tuple[int, int, int]]
    provenance: dict[str, Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseAwareGenerationError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseAwareGenerationError(f"cannot read JSON object {path}") from error
    require(isinstance(value, dict), f"{path}: expected one JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label}: integer required")
    result = int(value)
    if minimum is not None:
        require(result >= minimum, f"{label}: must be at least {minimum}")
    return result


def validate_template(manifest: dict[str, Any], binary_path: Path) -> list[int]:
    require(manifest.get("status") == "PASS", "compact template has not passed")
    record_bytes = int((manifest.get("binary_format") or {}).get("record_bytes", RECORD_BYTES))
    require(record_bytes == RECORD_BYTES, "compact template record width is not 12 bytes")
    requests = integer(manifest.get("requests"), "template requests", minimum=0)
    require(binary_path.is_file(), f"compact template binary is absent: {binary_path}")
    require(
        binary_path.stat().st_size == requests * RECORD_BYTES,
        "compact template binary size disagrees with its manifest",
    )
    expected_digest = manifest.get("binary_sha256")
    if expected_digest is not None:
        require(
            str(expected_digest) == sha256_file(binary_path),
            "compact template binary digest disagrees with its manifest",
        )
    objects = manifest.get("objects")
    require(isinstance(objects, list), "compact template object table is absent")
    require(
        [integer(row.get("template_object_index"), "template object index", minimum=0)
         for row in objects]
        == list(range(len(objects))),
        "compact template object indices are not dense",
    )
    extents = [integer(row.get("bytes"), "template object extent", minimum=1) for row in objects]
    require(len(extents) <= 0x10000, "compact template object table exceeds u16")
    return extents


def validate_sidecar(
    sidecar: dict[str, Any], *, binary_path: Path, template_requests: int
) -> list[dict[str, int]]:
    require(sidecar.get("schema") == SIDECAR_SCHEMA, "unsupported phase-aware sidecar schema")
    require(sidecar.get("status") == "PASS", "phase-aware sidecar has not passed")
    source = sidecar.get("source")
    require(isinstance(source, dict), "phase-aware sidecar source binding is absent")
    require(
        source.get("compact_binary_sha256") == sha256_file(binary_path),
        "phase-aware sidecar refers to another compact binary",
    )
    require(
        integer(source.get("compact_requests"), "sidecar compact requests", minimum=0)
        == template_requests,
        "phase-aware sidecar request count disagrees with the compact template",
    )
    raw_bundles = sidecar.get("bundles")
    require(isinstance(raw_bundles, list) and raw_bundles, "phase-aware sidecar has no bundles")
    bundles: list[dict[str, int]] = []
    keys: set[tuple[int, int, int, int, int, int]] = set()
    intervals: list[tuple[int, int, int]] = []
    for expected_index, raw in enumerate(raw_bundles):
        require(isinstance(raw, dict), "phase-aware bundle must be an object")
        row = {
            "bundle_index": integer(raw.get("bundle_index"), "bundle index", minimum=0),
            "kernel_ordinal": integer(raw.get("kernel_ordinal"), "kernel ordinal", minimum=0),
            "cta_x": integer(raw.get("cta_x"), "CTA x", minimum=0),
            "cta_y": integer(raw.get("cta_y"), "CTA y", minimum=0),
            "cta_z": integer(raw.get("cta_z"), "CTA z", minimum=0),
            "warp_in_cta": integer(raw.get("warp_in_cta"), "warp in CTA", minimum=0),
            "warp_program_ordinal": integer(
                raw.get("warp_program_ordinal"), "warp program ordinal", minimum=0
            ),
            "request_ordinal_begin": integer(
                raw.get("request_ordinal_begin"), "request begin", minimum=0
            ),
            "request_ordinal_end_exclusive": integer(
                raw.get("request_ordinal_end_exclusive"), "request end", minimum=1
            ),
        }
        require(row["bundle_index"] == expected_index, "bundle indices are not dense")
        require(
            row["request_ordinal_begin"] < row["request_ordinal_end_exclusive"]
            <= template_requests,
            f"bundle {expected_index} request range escapes the compact template",
        )
        key = (
            row["kernel_ordinal"], row["cta_x"], row["cta_y"], row["cta_z"],
            row["warp_in_cta"], row["warp_program_ordinal"],
        )
        require(key not in keys, f"duplicate phase-aware program key {key}")
        keys.add(key)
        intervals.append(
            (row["request_ordinal_begin"], row["request_ordinal_end_exclusive"], expected_index)
        )
        bundles.append(row)
    intervals.sort()
    require(intervals[0][0] == 0, "bundle request ranges do not begin at request zero")
    for left, right in zip(intervals, intervals[1:]):
        require(left[1] <= right[0], f"bundle request ranges overlap: {left[2]} and {right[2]}")
        require(left[1] == right[0], f"bundle request ranges leave a gap after {left[2]}")
    require(
        intervals[-1][1] == template_requests,
        "bundle request ranges do not cover the compact template tail",
    )
    return bundles


def rule_bundle_selector(rule: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return an optional exact dynamic-bundle selector.

    Most CTA rules apply to every dynamic instruction that touches one
    object/CTA-y/z phase.  A small number of real kernels contain a fixed
    prologue/epilogue bundle inside an otherwise translated phase.  Such an
    exception is identified by the stable per-warp program key carried by the
    sidecar.  Requiring both fields avoids silently broadening an exception to
    an entire warp or program ordinal.
    """

    raw_warp = rule.get("warp_in_cta")
    raw_program = rule.get("warp_program_ordinal")
    require(
        (raw_warp is None) == (raw_program is None),
        "phase rule bundle selector requires both warp_in_cta and "
        "warp_program_ordinal",
    )
    if raw_warp is None:
        return None, None
    return (
        integer(raw_warp, "rule warp in CTA", minimum=0),
        integer(raw_program, "rule warp program ordinal", minimum=0),
    )


def validate_rule(
    rule: dict[str, Any],
) -> tuple[int, int, int, str, str | None, int | None, int | None]:
    object_index = integer(rule.get("object_index"), "rule object index", minimum=0)
    cta_y = integer(rule.get("cta_y"), "rule CTA y", minimum=0)
    cta_z = integer(rule.get("cta_z"), "rule CTA z", minimum=0)
    operation_value = rule.get("operation")
    operation: str | None
    if operation_value is None:
        operation = None
    else:
        require(
            isinstance(operation_value, str) and operation_value in {"R", "W"},
            "phase rule operation must be R or W",
        )
        operation = operation_value
    kind = str(rule.get("kind", ""))
    require(
        kind in {
            "affine", "period2_swizzled", "tiled_swizzle", "shape_aware_shadow"
        },
        f"unsupported phase rule {kind!r}",
    )
    validation = rule.get("validation")
    require(isinstance(validation, dict), "phase rule has no validation receipt")
    if kind == "shape_aware_shadow":
        require(
            validation.get("status") in {
                SHADOW_RULE_STATUS,
                VALIDATED_SHAPE_RULE_STATUS,
            },
            "shape-aware rule lacks a recognized supplemental validation status",
        )
    else:
        require(
            validation.get("status") in VALIDATED_RULE_STATUSES,
            "phase rule has not passed a recognized validation gate",
        )
    if kind == "affine":
        integer(rule.get("x_stride_bytes"), "affine x stride")
    elif kind == "period2_swizzled":
        integer(rule.get("cycle_stride_bytes"), "period-2 cycle stride")
        phases = rule.get("phase_offsets_bytes")
        require(isinstance(phases, list) and len(phases) == 2, "period-2 rule needs two phase offsets")
        integer(phases[0], "period-2 phase 0 offset")
        integer(phases[1], "period-2 phase 1 offset")
    elif kind == "tiled_swizzle":
        period = integer(rule.get("period"), "tiled-swizzle period", minimum=2)
        integer(rule.get("quotient_stride_bytes"), "tiled-swizzle quotient stride")
        phases = rule.get("phase_offsets_bytes")
        require(
            isinstance(phases, list) and len(phases) == period,
            "tiled-swizzle rule needs one offset per phase",
        )
        for phase, value in enumerate(phases):
            integer(value, f"tiled-swizzle phase {phase} offset")
    else:
        coordinate_rule = rule.get("coordinate_rule")
        require(
            isinstance(coordinate_rule, dict),
            "shape-aware shadow coordinate rule is absent",
        )
        validate_shape_rule(coordinate_rule)
    warp, program = rule_bundle_selector(rule)
    return object_index, cta_y, cta_z, kind, operation, warp, program


def rule_coordinate(rule: dict[str, Any], cta_x: int) -> int:
    kind = str(rule["kind"])
    if kind == "affine":
        return cta_x * int(rule["x_stride_bytes"])
    if kind == "period2_swizzled":
        quotient, phase = divmod(cta_x, 2)
        return (
            quotient * int(rule["cycle_stride_bytes"])
            + int(rule["phase_offsets_bytes"][phase])
        )
    if kind == "tiled_swizzle":
        quotient, phase = divmod(cta_x, int(rule["period"]))
        return (
            quotient * int(rule["quotient_stride_bytes"])
            + int(rule["phase_offsets_bytes"][phase])
        )
    if kind == "shape_aware_shadow":
        return coordinate_from_validated_rule(rule["coordinate_rule"], cta_x)
    raise PhaseAwareGenerationError(f"unsupported phase rule {kind!r}")


def validate_tiled_swizzle_policy(policy: dict[str, Any], ordinal: int) -> None:
    """Validate the grid topology that accompanies tiled address rules.

    Version 1 intentionally supports only a one-deep CTA-z grid.  This makes
    an omitted or newly appearing z phase a hard error instead of silently
    treating a 3-D launch as the already validated 2-D CUTLASS mapping.
    """

    period = integer(policy.get("period"), "tiled-swizzle period", minimum=2)
    grid_cta_y = integer(policy.get("grid_cta_y"), "tiled-swizzle CTA-y extent", minimum=1)
    grid_cta_z = integer(policy.get("grid_cta_z"), "tiled-swizzle CTA-z extent", minimum=1)
    require(grid_cta_z == 1, "tiled-swizzle v1 requires a one-deep CTA-z grid")
    raw_masks = policy.get("active_cta_y_by_phase")
    require(
        isinstance(raw_masks, list) and len(raw_masks) == period,
        "tiled-swizzle policy needs one CTA-y mask per phase",
    )
    for phase, raw_mask in enumerate(raw_masks):
        require(isinstance(raw_mask, list) and raw_mask,
                f"tiled-swizzle phase {phase} has no active CTA-y values")
        values = [integer(value, f"tiled-swizzle phase {phase} CTA y", minimum=0)
                  for value in raw_mask]
        require(values == sorted(set(values)),
                f"tiled-swizzle phase {phase} CTA-y mask is not sorted and unique")
        require(values[-1] < grid_cta_y,
                f"tiled-swizzle phase {phase} CTA-y mask escapes the launch grid")
    validation = policy.get("validation")
    require(isinstance(validation, dict),
            f"kernel {ordinal} tiled-swizzle policy has no validation receipt")
    require(validation.get("status") in VALIDATED_RULE_STATUSES,
            f"kernel {ordinal} tiled-swizzle topology has not passed a validation gate")


def policy_period(policy: dict[str, Any]) -> int:
    """Return the address/topology period relevant to endpoint validation."""

    if str(policy["kind"]) == "tiled_swizzle":
        return int(policy["period"])
    periods = []
    for rule in policy.get("rules") or []:
        kind = str(rule["kind"])
        if kind == "period2_swizzled":
            periods.append(2)
        elif kind == "shape_aware_shadow":
            periods.append(coordinate_period_ctas(rule["coordinate_rule"]))
        else:
            periods.append(1)
    return math.lcm(*periods) if periods else 1


def active_y_for_target(policy: dict[str, Any], target_cta_x: int) -> set[int] | None:
    if str(policy["kind"]) != "tiled_swizzle":
        return None
    phase = target_cta_x % int(policy["period"])
    return {int(value) for value in policy["active_cta_y_by_phase"][phase]}


def causal_is_implemented(policy: dict[str, Any]) -> bool:
    return policy.get("implementation_status") == "PASS_VALIDATED_CAUSAL_TILES"


def validate_causal_policy(policy: dict[str, Any], ordinal: int) -> None:
    """Validate the deliberately narrow causal-tile generator contract.

    Version 1 models a source CTA-x zero containing exactly one loop body.
    Fixed bundles are translated to the target query tile.  The contiguous
    loop body is emitted for tiles ``target_x .. 0``.  This matches causal
    FlashAttention without embedding a kernel name or model dimension in the
    generator, while all address translations remain explicit rules.
    """

    status = policy.get("implementation_status")
    if status == "UNIMPLEMENTED_FAIL_CLOSED":
        return
    require(
        status == "PASS_VALIDATED_CAUSAL_TILES",
        f"kernel {ordinal} causal policy has unsupported implementation status",
    )
    source_x = integer(policy.get("source_cta_x"), "causal source CTA x", minimum=0)
    require(source_x == 0, "causal-tile v1 requires source CTA x=0")
    require(
        policy.get("iteration_order") == "descending_inclusive",
        "causal-tile v1 requires descending_inclusive iteration order",
    )
    integer(
        policy.get("validated_target_cta_x_max"),
        "causal validated target CTA-x maximum",
        minimum=0,
    )
    validation = policy.get("validation")
    require(isinstance(validation, dict), "causal policy has no validation receipt")
    require(
        validation.get("status") in VALIDATED_RULE_STATUSES,
        "causal policy has not passed a recognized validation gate",
    )
    raw_loop_objects = policy.get("loop_object_indices")
    require(
        isinstance(raw_loop_objects, list) and raw_loop_objects,
        "causal policy has no loop objects",
    )
    loop_objects = [
        integer(value, "causal loop object index", minimum=0)
        for value in raw_loop_objects
    ]
    require(
        loop_objects == sorted(set(loop_objects)),
        "causal loop object indices must be sorted and unique",
    )
    rules = policy.get("rules")
    require(isinstance(rules, list) and rules, f"kernel {ordinal} has no causal rules")
    seen: set[tuple[int, int, int, int | None, int | None, str | None]] = set()
    operation_modes: dict[
        tuple[int, int, int, int | None, int | None], set[str | None]
    ] = {}
    for rule in rules:
        require(isinstance(rule, dict), "causal rule must be an object")
        object_index, cta_y, cta_z, kind, operation, warp, program = validate_rule(rule)
        require(kind == "affine", "causal-tile v1 supports affine address rules only")
        base_key = object_index, cta_y, cta_z, warp, program
        key = (*base_key, operation)
        require(key not in seen, f"kernel {ordinal} has duplicate causal rule {key}")
        seen.add(key)
        modes = operation_modes.setdefault(base_key, set())
        modes.add(operation)
        require(
            not (None in modes and len(modes) > 1),
            f"kernel {ordinal} mixes wildcard and operation-specific causal rules "
            f"for object/yz {base_key}",
        )


def policy_index(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported phase-aware generator plan schema")
    require(plan.get("status") == "PASS", "phase-aware generator plan has not passed")
    raw = plan.get("kernel_generators")
    require(isinstance(raw, list) and raw, "phase-aware generator plan has no kernels")
    result: dict[int, dict[str, Any]] = {}
    has_shadow_rules = False
    for policy in raw:
        require(isinstance(policy, dict), "kernel generator must be an object")
        ordinal = integer(policy.get("kernel_ordinal"), "generator kernel ordinal", minimum=0)
        require(ordinal not in result, f"duplicate kernel generator {ordinal}")
        kind = str(policy.get("kind", ""))
        require(
            kind in {
                "phase_rules", "tiled_swizzle", "exact_anchor", "causal",
                SEMANTIC_EMBEDDING_KIND,
            },
            f"unsupported kernel generator {kind!r}",
        )
        if kind in {"phase_rules", "tiled_swizzle"}:
            integer(policy.get("source_cta_x"), "source CTA x", minimum=0)
            rules = policy.get("rules")
            require(isinstance(rules, list) and rules, f"kernel {ordinal} has no phase rules")
            seen: set[
                tuple[int, int, int, int | None, int | None, str | None]
            ] = set()
            operation_modes: dict[
                tuple[int, int, int, int | None, int | None], set[str | None]
            ] = {}
            for rule in rules:
                require(isinstance(rule, dict), "phase rule must be an object")
                (
                    object_index, cta_y, cta_z, rule_kind, operation, warp, program
                ) = validate_rule(rule)
                has_shadow_rules = has_shadow_rules or rule_kind == "shape_aware_shadow"
                base_key = object_index, cta_y, cta_z, warp, program
                key = (*base_key, operation)
                require(key not in seen, f"kernel {ordinal} has duplicate rule {key}")
                seen.add(key)
                modes = operation_modes.setdefault(base_key, set())
                modes.add(operation)
                require(
                    not (None in modes and len(modes) > 1),
                    f"kernel {ordinal} mixes wildcard and operation-specific rules "
                    f"for object/yz {base_key}",
                )
            if kind == "tiled_swizzle":
                validate_tiled_swizzle_policy(policy, ordinal)
                period = int(policy["period"])
                require(
                    all(str(rule["kind"]) == "tiled_swizzle" for rule in rules),
                    f"kernel {ordinal} tiled-swizzle policy mixes address-rule kinds",
                )
                require(
                    all(int(rule["period"]) == period for rule in rules),
                    f"kernel {ordinal} tiled-swizzle rule periods disagree",
                )
        elif kind == "exact_anchor":
            captured = policy.get("captured_cta_x")
            require(
                isinstance(captured, list) and captured,
                f"kernel {ordinal} exact-anchor policy has no captured CTA-x list",
            )
            values = [integer(value, "captured CTA x", minimum=0) for value in captured]
            require(len(values) == len(set(values)), "exact-anchor CTA-x values repeat")
        elif kind == "causal":
            validate_causal_policy(policy, ordinal)
        result[ordinal] = policy
    retrospective_contract = plan.get("supplemental_shadow_runtime_binding")
    validated_contract = plan.get("supplemental_validated_shape_runtime_binding")
    if has_shadow_rules:
        require(
            not (retrospective_contract is not None
                 and validated_contract is not None),
            "shape-aware plan contains two competing runtime contracts",
        )
        if plan.get("classification") == SHADOW_PLAN_CLASSIFICATION:
            require(
                isinstance(retrospective_contract, dict)
                and retrospective_contract.get("schema")
                == SHADOW_CONTRACT_SCHEMA,
                "shape-aware shadow plan lacks its runtime binding contract",
            )
            require(
                validated_contract is None,
                "retrospective shape-aware plan contains a prospective contract",
            )
        elif plan.get("classification") == VALIDATED_SHAPE_PLAN_CLASSIFICATION:
            require(
                isinstance(validated_contract, dict)
                and validated_contract.get("schema")
                == VALIDATED_SHAPE_CONTRACT_SCHEMA,
                "prospectively validated shape-aware plan lacks its runtime binding contract",
            )
            require(
                retrospective_contract is None,
                "prospectively validated shape-aware plan contains a retrospective contract",
            )
        else:
            raise PhaseAwareGenerationError(
                "shape-aware rules are forbidden outside a recognized supplemental plan"
            )
    else:
        require(
            retrospective_contract is None and validated_contract is None,
            "shape runtime binding is present without shape-aware rules",
        )
    return result


def target_extents(plan: dict[str, Any], source_extents: list[int]) -> list[int]:
    """Return target-layout extents, defaulting to the captured layout.

    A longer context may legitimately address beyond an anchor object's
    captured extent.  The lazy full-model planner already knows the target
    object layout, so the generator plan may bind that explicit extent vector.
    It is never inferred from an address delta.
    """
    raw = plan.get("target_object_extents_bytes")
    if raw is None:
        return list(source_extents)
    require(
        isinstance(raw, list) and len(raw) == len(source_extents),
        "target object extent vector disagrees with the compact object table",
    )
    return [integer(value, "target object extent", minimum=1) for value in raw]


def bundle_order(row: dict[str, int]) -> tuple[int, int, int, int, int, int]:
    """Deterministic legal order; not a claim about GPU CTA scheduling."""
    return (
        row["kernel_ordinal"], row["cta_z"], row["cta_y"], row["warp_in_cta"],
        row["warp_program_ordinal"], row["bundle_index"],
    )


def read_bundle_records(source: BinaryIO, row: dict[str, int]) -> list[tuple[int, ...]]:
    begin = row["request_ordinal_begin"]
    count = row["request_ordinal_end_exclusive"] - begin
    source.seek(begin * RECORD_BYTES)
    payload = source.read(count * RECORD_BYTES)
    require(len(payload) == count * RECORD_BYTES, "compact template is truncated")
    return [tuple(int(value) for value in record) for record in RECORD_STRUCT.iter_unpack(payload)]


def validate_shadow_runtime_binding(
    *,
    plan: dict[str, Any],
    template_manifest: dict[str, Any],
    template_binary_path: Path,
    policies: dict[int, dict[str, Any]],
    artifact_root: Path | None = None,
) -> dict[str, Any] | None:
    """Verify every external and phase-lifetime dependency of a shadow plan.

    The generic descriptor and generator-plan digests protect the executable
    rule text.  A lifetime repair additionally depends on a particular
    framework allocation being live in the diagnosed logical phase, so this
    gate also binds the complete target probe manifest, the exact allocation
    row and tensor view, the source request operations, and the immutable
    shadow-repair receipt.  Any absent or changed component is a hard error.
    """

    validated_contract = plan.get("supplemental_validated_shape_runtime_binding")
    if validated_contract is not None:
        require(
            plan.get("supplemental_shadow_runtime_binding") is None,
            "shape-aware plan contains two competing runtime contracts",
        )
        return validate_prospective_shape_runtime_binding(
            plan=plan,
            template_manifest=template_manifest,
            policies=policies,
            artifact_root=artifact_root,
        )

    contract = plan.get("supplemental_shadow_runtime_binding")
    if contract is None:
        return None
    require(
        plan.get("classification") == SHADOW_PLAN_CLASSIFICATION,
        "shadow runtime binding is attached to another plan classification",
    )
    require(isinstance(contract, dict), "shadow runtime binding is malformed")
    required_fields = {
        "schema",
        "status",
        "classification",
        "original_kernel_ordinal",
        "template_kernel_ordinal",
        "logical_role",
        "target_probe_manifest",
        "shadow_repair",
        "phase_lifetime_bindings",
        "contract_fingerprint_sha256",
    }
    require(set(contract) == required_fields, "shadow runtime binding fields differ")
    require(contract.get("schema") == SHADOW_CONTRACT_SCHEMA,
            "unsupported shadow runtime binding schema")
    require(contract.get("status") == "PASS_RETROSPECTIVE_SUPPLEMENTAL_BINDING",
            "shadow runtime binding has not passed")
    require(contract.get("classification")
            == "digest-bound phase-lifetime and shape-rule adapter",
            "shadow runtime binding classification differs")
    unhashed = {key: value for key, value in contract.items()
                if key != "contract_fingerprint_sha256"}
    require(contract.get("contract_fingerprint_sha256") == canonical_sha256(unhashed),
            "shadow runtime binding fingerprint differs")

    def bound_json(value: Any, label: str) -> tuple[dict[str, Any], Path, str]:
        require(isinstance(value, dict) and set(value) == {"path", "sha256"},
                f"{label} artifact binding is malformed")
        raw = str(value["path"])
        path = resolve_input(raw)
        if not path.is_file() and artifact_root is not None:
            markers = (
                "experiments/hbfsim_trace_capture/results/",
                "results/hbfsim_trace_capture/",
            )
            marker = next((item for item in markers if item in raw), None)
            if marker is not None:
                path = (artifact_root / raw.split(marker, 1)[1]).resolve()
            elif not Path(raw).is_absolute():
                path = (artifact_root / raw).resolve()
        require(path.is_file(), f"missing {label} artifact {path}")
        digest = sha256_file(path)
        require(value["sha256"] == digest,
                f"{label} artifact digest differs")
        return load_json(path), path, digest

    target_manifest, target_path, target_digest = bound_json(
        contract["target_probe_manifest"], "shadow target probe manifest"
    )
    require(target_manifest.get("schema") == QWEN_PROBE_SCHEMA,
            "unsupported shadow target probe manifest schema")
    repair, repair_path, repair_digest = bound_json(
        contract["shadow_repair"], "full-grid shadow repair"
    )
    require(repair.get("schema") == SHADOW_REPAIR_SCHEMA
            and repair.get("status")
            == "PASS_NINE_SHADOW_REPAIRS_PENDING_FRESH_BOUNDARY_HOLDOUT",
            "full-grid shadow repair has not passed")
    require((repair.get("plan_fingerprint") or {}).get("sha256")
            == canonical_sha256(repair.get("plan")),
            "full-grid shadow-repair plan fingerprint differs")
    require(((repair.get("inputs") or {}).get("target_probe_manifest") or {})
            .get("sha256") == target_digest,
            "shadow repair and runtime contract bind different target manifests")

    original_ordinal = integer(
        contract.get("original_kernel_ordinal"),
        "shadow original kernel ordinal",
        minimum=0,
    )
    template_ordinal = integer(
        contract.get("template_kernel_ordinal"),
        "shadow template kernel ordinal",
        minimum=0,
    )
    require(set(policies) == {template_ordinal},
            "shadow contract and executable policy ordinal differ")
    repair_rows = [
        row for row in (repair.get("plan") or {}).get("kernel_plans") or []
        if int(row.get("ordinal", -1)) == original_ordinal
    ]
    require(len(repair_rows) == 1,
            "shadow repair lacks the contracted kernel")
    repair_kernel = repair_rows[0]
    require(contract.get("logical_role") == repair_kernel.get("logical_role"),
            "shadow logical role differs from the repair")
    repair_groups = {
        int(row["object_index"]): row for row in repair_kernel.get("groups") or []
    }
    objects = template_manifest.get("objects") or []
    require(set(repair_groups) == set(range(len(objects))),
            "shadow repair groups do not cover the compact object table")
    target_extents = plan.get("target_object_extents_bytes") or []
    require(len(target_extents) == len(objects),
            "shadow target extents do not cover the compact object table")
    stable_objects = {
        int(row["template_object_index"]): row
        for row in (plan.get("binding_receipt") or {}).get("stable_objects") or []
    }
    require(set(stable_objects) == set(repair_groups),
            "shadow stable-object bindings are incomplete")
    rules = policies[template_ordinal].get("rules") or []
    rules_by_object = {int(row["object_index"]): row for row in rules}
    require(len(rules_by_object) == len(rules)
            and set(rules_by_object) == set(repair_groups),
            "shadow executable rules do not cover every object exactly once")
    for object_index, group in repair_groups.items():
        rule = rules_by_object[object_index]
        require(rule.get("kind") == "shape_aware_shadow"
                and rule.get("coordinate_rule") == group.get("coordinate_rule"),
                f"shadow executable rule differs for object {object_index}")
        validation = rule.get("validation") or {}
        require(validation.get("status") == SHADOW_RULE_STATUS
                and validation.get("shadow_repair_sha256") == repair_digest
                and validation.get("full_grid_rule_sha256")
                == (group.get("full_grid_bounds") or {}).get("rule_sha256"),
                f"shadow rule validation receipt differs for object {object_index}")
        extent = int((group.get("object_binding") or {})["object_extent_bytes"])
        require(int(target_extents[object_index]) == extent
                and int(stable_objects[object_index]["target_bytes"]) == extent,
                f"shadow target extent differs for object {object_index}")

    expected_lifetime_indices = {
        index for index, group in repair_groups.items()
        if (group.get("object_binding") or {}).get("binding_mode")
        == "kernel_ordinal_and_framework_phase_allocation"
    }
    raw_lifetime = contract.get("phase_lifetime_bindings")
    require(isinstance(raw_lifetime, list),
            "shadow phase-lifetime binding list is absent")
    lifetime = {int(row["object_index"]): row for row in raw_lifetime}
    require(len(lifetime) == len(raw_lifetime)
            and set(lifetime) == expected_lifetime_indices,
            "shadow phase-lifetime bindings are missing or duplicated")
    allocations = (target_manifest.get("tensor_ownership") or {}).get(
        "allocations"
    ) or []
    all_records = [
        tuple(int(value) for value in record)
        for record in RECORD_STRUCT.iter_unpack(template_binary_path.read_bytes())
    ]
    overrides = []
    lifetime_fields = {
        "object_index",
        "allocation_id",
        "capture_alias_source_name",
        "capture_alias_kind",
        "semantic_source_name",
        "semantic_kind",
        "binding_mode",
        "logical_role",
        "request_operations",
        "address_begin",
        "object_extent_bytes",
        "allocation_row_sha256",
        "tensor_view_sha256",
        "shadow_binding_fingerprint_sha256",
    }
    for object_index, binding in sorted(lifetime.items()):
        require(isinstance(binding, dict) and set(binding) == lifetime_fields,
                f"shadow lifetime binding fields differ for object {object_index}")
        group = repair_groups[object_index]
        object_binding = group["object_binding"]
        require(binding["capture_alias_source_name"]
                == group["original_stable_name"]
                and binding["capture_alias_kind"] == group["original_kind"],
                f"shadow capture alias differs for object {object_index}")
        require(binding["semantic_kind"] == "activation"
                and binding["semantic_source_name"] == binding["allocation_id"]
                and binding["binding_mode"]
                == "digest_bound_framework_phase_allocation",
                f"shadow runtime semantics differ for object {object_index}")
        require(binding["logical_role"] == contract["logical_role"],
                f"shadow lifetime logical role differs for object {object_index}")
        require(binding["shadow_binding_fingerprint_sha256"]
                == object_binding.get("binding_fingerprint_sha256"),
                f"shadow lifetime fingerprint differs for object {object_index}")
        rows = [row for row in allocations
                if row.get("allocation_id") == binding["allocation_id"]]
        require(len(rows) == 1,
                f"target manifest lacks unique lifetime allocation for object {object_index}")
        allocation = rows[0]
        require(binding["allocation_row_sha256"] == canonical_sha256(allocation),
                f"lifetime allocation digest differs for object {object_index}")
        require(int(binding["address_begin"]) == int(allocation["address_begin"])
                and int(binding["object_extent_bytes"])
                == int(allocation["address_end_exclusive"])
                - int(allocation["address_begin"]),
                f"lifetime allocation extent differs for object {object_index}")
        expected_view = object_binding.get("tensor_view") or {}
        views = [
            row for row in allocation.get("tensor_views") or []
            if [int(value) for value in row.get("shape") or []]
            == expected_view.get("shape")
            and [int(value) for value in row.get("stride") or []]
            == expected_view.get("stride")
            and str(row.get("dtype", "")) == expected_view.get("dtype")
            and int(row.get("logical_bytes", 0))
            == int(expected_view.get("logical_bytes", -1))
            and int(row.get("first_seen_event", -1))
            == int(expected_view.get("first_seen_event", -2))
            and int(row.get("last_seen_event", -1))
            == int(expected_view.get("last_seen_event", -2))
        ]
        require(len(views) == 1
                and binding["tensor_view_sha256"] == canonical_sha256(views[0]),
                f"lifetime tensor-view digest differs for object {object_index}")
        operations = sorted({
            OPERATION_NAME[int(record[4])] for record in all_records
            if int(record[0]) == object_index
        })
        require(operations == binding["request_operations"]
                == sorted(group["request_operations"]),
                f"lifetime request operations differ for object {object_index}")
        stable = stable_objects[object_index]
        require(stable.get("kind") == "activation"
                and stable.get("stable_name") == binding["semantic_source_name"],
                f"lifetime stable-object semantics differ for object {object_index}")
        overrides.append({
            "source_template_object_index": object_index,
            "kind": "activation",
            "source_name": binding["semantic_source_name"],
            "binding_mode": binding["binding_mode"],
            "capture_alias_source_name": binding["capture_alias_source_name"],
            "capture_alias_kind": binding["capture_alias_kind"],
            "allocation_row_sha256": binding["allocation_row_sha256"],
            "tensor_view_sha256": binding["tensor_view_sha256"],
        })
    return {
        "status": "PASS_DIGEST_BOUND_SHADOW_RUNTIME_BINDING",
        "original_kernel_ordinal": original_ordinal,
        "template_kernel_ordinal": template_ordinal,
        "logical_role": contract["logical_role"],
        "target_probe_manifest": str(target_path),
        "target_probe_manifest_sha256": target_digest,
        "shadow_repair": str(repair_path),
        "shadow_repair_sha256": repair_digest,
        "contract_fingerprint_sha256": contract["contract_fingerprint_sha256"],
        "runtime_object_overrides": overrides,
    }


def validate_prospective_shape_runtime_binding(
    *,
    plan: dict[str, Any],
    template_manifest: dict[str, Any],
    policies: dict[int, dict[str, Any]],
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Bind executable shape rules to their fresh prospective holdout.

    The underlying formulas were originally diagnosed retrospectively.  They
    become executable supplemental evidence only because a later target CTA
    was selected and byte-bound before decode, then passed without any
    post-hoc shift search.  This validator keeps that chronology attached to
    every loadable descriptor and rejects K25, which never passed the fresh
    phase/lifetime gate.
    """

    require(
        plan.get("classification") == VALIDATED_SHAPE_PLAN_CLASSIFICATION,
        "prospectively validated shape runtime binding is attached to another plan",
    )
    contract = plan.get("supplemental_validated_shape_runtime_binding")
    require(isinstance(contract, dict), "validated shape runtime binding is malformed")
    required_fields = {
        "schema",
        "status",
        "classification",
        "original_kernel_ordinal",
        "template_kernel_ordinal",
        "validated_cta_x",
        "shape_plan",
        "fresh_evaluation",
        "rule_bindings",
        "contract_fingerprint_sha256",
    }
    require(set(contract) == required_fields,
            "validated shape runtime binding fields differ")
    require(contract.get("schema") == VALIDATED_SHAPE_CONTRACT_SCHEMA,
            "unsupported validated shape runtime binding schema")
    require(contract.get("status")
            == "PASS_PROSPECTIVE_FRESH_SHAPE_HOLDOUT_BINDING",
            "validated shape runtime binding has not passed")
    require(contract.get("classification")
            == "digest-bound prospective shape-rule adapter",
            "validated shape runtime binding classification differs")
    unhashed = {key: value for key, value in contract.items()
                if key != "contract_fingerprint_sha256"}
    require(contract.get("contract_fingerprint_sha256")
            == canonical_sha256(unhashed),
            "validated shape runtime binding fingerprint differs")

    def bound_json(value: Any, label: str) -> tuple[dict[str, Any], Path, str]:
        require(isinstance(value, dict) and set(value) == {"path", "sha256"},
                f"{label} artifact binding is malformed")
        raw = str(value["path"])
        path = resolve_input(raw)
        if not path.is_file() and artifact_root is not None:
            markers = (
                "experiments/hbfsim_trace_capture/results/",
                "results/hbfsim_trace_capture/",
            )
            marker = next((item for item in markers if item in raw), None)
            if marker is not None:
                path = (artifact_root / raw.split(marker, 1)[1]).resolve()
            elif not Path(raw).is_absolute():
                path = (artifact_root / raw).resolve()
        require(path.is_file(), f"missing {label} artifact {path}")
        digest = sha256_file(path)
        require(value["sha256"] == digest, f"{label} artifact digest differs")
        return load_json(path), path, digest

    shape_plan, shape_plan_path, shape_plan_digest = bound_json(
        contract["shape_plan"], "validated shape source plan"
    )
    require(shape_plan.get("schema") == VALIDATED_SHAPE_SOURCE_PLAN_SCHEMA
            and shape_plan.get("status")
            == "PASS_SHADOW_K7_K11_K35_K25_FAIL_CLOSED",
            "validated shape source plan has not passed")
    require((shape_plan.get("plan_fingerprint") or {}).get("sha256")
            == canonical_sha256(shape_plan.get("plan")),
            "validated shape source-plan fingerprint differs")
    evaluation, evaluation_path, evaluation_digest = bound_json(
        contract["fresh_evaluation"], "fresh shape evaluation"
    )
    require(evaluation.get("schema") == VALIDATED_SHAPE_EVALUATION_SCHEMA
            and evaluation.get("status")
            == "PASS_FRESH_SHAPE_RULE_HOLDOUT_AND_SUPPLEMENTAL_LAYER",
            "fresh shape evaluation has not passed")
    fresh = evaluation.get("fresh_shape_rule_holdout") or {}
    require(fresh.get("status") == "PASS"
            and fresh.get("posthoc_shift_search_performed") is False,
            "fresh shape gate was not applied prospectively")
    require(not fresh.get("failure_reasons")
            and not fresh.get("missing_left_group_keys")
            and not fresh.get("missing_right_group_keys")
            and not fresh.get("extra_left_group_keys")
            and not fresh.get("extra_right_group_keys"),
            "fresh shape evaluation has incomplete group coverage")
    freeze = evaluation.get("freeze_binding") or {}
    require(int(freeze.get(
        "fresh_target_address_records_decoded_before_capture_binding", -1
    )) == 0, "fresh target addresses were decoded before byte binding")
    census = evaluation.get("census") or {}
    require(census.get("enabled_kernels") == [7, 11, 35]
            and census.get("unresolved_kernels") == [25]
            and int(census.get("projected_fail_closed_unresolved_requests", -1))
            == 5_242_880,
            "fresh evaluation no longer keeps K25 fail-closed")
    require((evaluation.get("primary_gate_unchanged") or {}).get("status")
            == "PARTIAL_REMAINING_1D_FAR_HOLDOUTS",
            "fresh evaluation rewrites the immutable primary gate")

    original_ordinal = integer(
        contract.get("original_kernel_ordinal"),
        "validated shape original kernel ordinal",
        minimum=0,
    )
    template_ordinal = integer(
        contract.get("template_kernel_ordinal"),
        "validated shape template kernel ordinal",
        minimum=0,
    )
    validated_cta_x = integer(
        contract.get("validated_cta_x"), "validated shape CTA x", minimum=0
    )
    require(original_ordinal in {7, 11, 35},
            "validated shape contract includes an unpassed kernel")
    require(set(policies) == {template_ordinal},
            "validated shape contract and executable policy ordinal differ")
    source_rows = [
        row for row in (shape_plan.get("plan") or {}).get("kernel_plans") or []
        if int(row.get("ordinal", -1)) == original_ordinal
    ]
    require(len(source_rows) == 1
            and source_rows[0].get("shadow_generator_enabled") is True,
            "validated shape source plan lacks the contracted kernel")
    source_kernel = source_rows[0]

    evaluation_rows = [
        row for row in fresh.get("groups") or []
        if int(row.get("kernel_ordinal", -1)) == original_ordinal
    ]
    require(evaluation_rows, "fresh evaluation lacks the contracted kernel")
    evaluated = {
        (
            str(row.get("source_name", "")),
            str(row.get("source_kind", "")),
            int(row.get("cta_y", -1)),
            int(row.get("cta_z", -1)),
        ): row
        for row in evaluation_rows
    }
    require(len(evaluated) == len(evaluation_rows),
            "fresh evaluation repeats a shape group")

    objects = template_manifest.get("objects") or []
    stable = {
        int(row["template_object_index"]): row
        for row in (plan.get("binding_receipt") or {}).get("stable_objects") or []
    }
    identity_to_index = {
        (str(row.get("stable_name", "")), str(row.get("kind", ""))): index
        for index, row in stable.items()
    }
    require(len(identity_to_index) == len(stable),
            "validated shape stable object identities repeat")
    groups: dict[int, dict[str, Any]] = {}
    for row in source_kernel.get("groups") or []:
        identity = str(row.get("source_name", "")), str(row.get("source_kind", ""))
        require(identity in identity_to_index,
                f"validated shape source group is absent from compact objects: {identity!r}")
        object_index = identity_to_index[identity]
        require(object_index not in groups,
                "validated shape source plan repeats an object group")
        groups[object_index] = row
    policy = policies[template_ordinal]
    require(policy.get("kind") == "phase_rules"
            and int(policy.get("source_cta_x", -1)) == 0,
            "validated shape executable policy is not x0 phase-rules")
    rules = {int(row["object_index"]): row for row in policy.get("rules") or []}
    require(len(rules) == len(policy.get("rules") or [])
            and set(rules) == set(groups),
            "validated shape executable rules do not cover each object once")
    require(set(groups) == set(range(len(objects))) == set(stable),
            "validated shape object bindings are incomplete")
    target_extents = plan.get("target_object_extents_bytes") or []
    require(len(target_extents) == len(groups),
            "validated shape target extents are incomplete")

    expected_bindings = []
    for object_index, group in sorted(groups.items()):
        key = (
            str(group["source_name"]),
            str(group["source_kind"]),
            int(group["cta_y"]),
            int(group["cta_z"]),
        )
        require(key in evaluated,
                f"fresh evaluation lacks validated shape group {key!r}")
        observed = evaluated[key]
        coordinate_rule = group["coordinate_rule"]
        rule_digest = canonical_sha256(coordinate_rule)
        require(observed.get("status") == "PASS"
                and observed.get("coordinate_rule_sha256") == rule_digest
                and int(observed.get("left_requests", -1))
                == int(observed.get("right_requests", -2))
                == int(observed.get("matched_requests_at_frozen_shift", -3))
                == int(observed.get("expected_requests", -4))
                and float(observed.get("multiset_jaccard_at_frozen_shift", -1.0))
                == 1.0,
                f"fresh evaluation did not exactly pass shape group {key!r}")
        predicted_delta = (
            coordinate_from_validated_rule(coordinate_rule, validated_cta_x)
            - coordinate_from_validated_rule(coordinate_rule, 0)
        )
        require(int(observed.get("applied_frozen_shift_bytes", -1))
                == predicted_delta,
                f"fresh evaluation shift differs for shape group {key!r}")
        rule = rules[object_index]
        validation = rule.get("validation") or {}
        require(rule.get("kind") == "shape_aware_shadow"
                and rule.get("coordinate_rule") == coordinate_rule
                and validation.get("status") == VALIDATED_SHAPE_RULE_STATUS
                and validation.get("shape_plan_sha256") == shape_plan_digest
                and validation.get("fresh_evaluation_sha256") == evaluation_digest
                and validation.get("full_grid_rule_sha256")
                == (group.get("full_grid_bounds") or {}).get("rule_sha256")
                and int(validation.get("validated_cta_x", -1)) == validated_cta_x,
                f"validated shape executable rule differs for object {object_index}")
        require(str(stable[object_index].get("stable_name")) == key[0]
                and str(stable[object_index].get("kind")) == key[1],
                f"validated shape stable object differs for object {object_index}")
        extent = int((group.get("object_binding") or {})[
            "object_extent_bytes"
        ])
        require(int(target_extents[object_index]) == extent
                and int(stable[object_index]["target_bytes"]) == extent,
                f"validated shape target extent differs for object {object_index}")
        operations = sorted(str(row["operation"])
                            for row in observed.get("by_operation") or [])
        require(operations and all(value in {"R", "W"} for value in operations),
                f"fresh evaluation has invalid operations for shape group {key!r}")
        expected_bindings.append({
            "object_index": object_index,
            "source_name": key[0],
            "source_kind": key[1],
            "cta_y": key[2],
            "cta_z": key[3],
            "coordinate_rule_sha256": rule_digest,
            "request_operations": operations,
            "expected_requests": int(observed["expected_requests"]),
        })
    require(set(evaluated) == {
        (
            str(row["source_name"]), str(row["source_kind"]),
            int(row["cta_y"]), int(row["cta_z"]),
        )
        for row in groups.values()
    }, "fresh evaluation contains an unmatched contracted-kernel group")
    require(contract.get("rule_bindings") == expected_bindings,
            "validated shape rule-binding receipt differs")
    return {
        "status": "PASS_DIGEST_BOUND_PROSPECTIVE_SHAPE_RUNTIME_BINDING",
        "original_kernel_ordinal": original_ordinal,
        "template_kernel_ordinal": template_ordinal,
        "validated_cta_x": validated_cta_x,
        "shape_plan": str(shape_plan_path),
        "shape_plan_sha256": shape_plan_digest,
        "fresh_evaluation": str(evaluation_path),
        "fresh_evaluation_sha256": evaluation_digest,
        "contract_fingerprint_sha256": contract[
            "contract_fingerprint_sha256"
        ],
        "runtime_object_overrides": [],
    }


def prepare_generator(
    *,
    template_manifest: dict[str, Any],
    template_binary_path: Path,
    sidecar: dict[str, Any],
    plan: dict[str, Any],
    artifact_root: Path | None = None,
) -> PreparedPhaseGenerator:
    """Validate and index a generator once for repeated CTA-x emission."""

    source_extents = validate_template(template_manifest, template_binary_path)
    bundles = validate_sidecar(
        sidecar,
        binary_path=template_binary_path,
        template_requests=int(template_manifest["requests"]),
    )
    policies = policy_index(plan)
    extents = target_extents(plan, source_extents)
    for ordinal, policy in list(policies.items()):
        if str(policy["kind"]) != SEMANTIC_EMBEDDING_KIND:
            continue
        normalized = validate_semantic_embedding_policy(
            policy, object_extents=extents
        )
        require(
            int(normalized["kernel_ordinal"]) == ordinal,
            "semantic embedding kernel ordinal changed during validation",
        )
        policies[ordinal] = {**policy, "_semantic_program": normalized}
    by_kernel_x: dict[tuple[int, int], list[dict[str, int]]] = {}
    for bundle in bundles:
        by_kernel_x.setdefault(
            (bundle["kernel_ordinal"], bundle["cta_x"]), []
        ).append(bundle)
    # A bundle-specific exception is deliberately stronger than a normal
    # object/phase rule.  Prove that its exact per-warp program key exists in
    # the captured source and that it actually touches the named
    # object/operation.  Otherwise a typo could leave the broad fallback rule
    # active while the plan still appeared to validate.
    with template_binary_path.open("rb") as source:
        for ordinal, policy in policies.items():
            if str(policy["kind"]) not in {"phase_rules", "tiled_swizzle", "causal"}:
                continue
            raw_rules = policy.get("rules")
            if not isinstance(raw_rules, list):
                continue
            source_x = int(policy["source_cta_x"])
            source_bundles = by_kernel_x.get((ordinal, source_x), [])
            bundle_facts: set[tuple[int, int, int, str, int, int]] = set()
            for bundle in source_bundles:
                for record in read_bundle_records(source, bundle):
                    bundle_facts.add((
                        int(record[0]), int(bundle["cta_y"]), int(bundle["cta_z"]),
                        OPERATION_NAME[int(record[4])], int(bundle["warp_in_cta"]),
                        int(bundle["warp_program_ordinal"]),
                    ))
            for rule in raw_rules:
                warp, program = rule_bundle_selector(rule)
                if warp is None:
                    continue
                operation = rule.get("operation")
                prefix = (
                    int(rule["object_index"]), int(rule["cta_y"]), int(rule["cta_z"])
                )
                matches = any(
                    fact[:3] == prefix
                    and (operation is None or fact[3] == operation)
                    and fact[4:] == (warp, program)
                    for fact in bundle_facts
                )
                require(
                    matches,
                    f"kernel {ordinal}: bundle-specific rule does not match a source "
                    f"request: {(*prefix, operation, warp, program)}",
                )
    for ordinal, policy in policies.items():
        if str(policy["kind"]) != "tiled_swizzle":
            continue
        source_x = int(policy["source_cta_x"])
        source_bundles = by_kernel_x.get((ordinal, source_x), [])
        captured = {(int(row["cta_y"]), int(row["cta_z"])) for row in source_bundles}
        expected = {(value, 0) for value in active_y_for_target(policy, source_x) or set()}
        require(
            captured == expected,
            f"kernel {ordinal}: source CTA topology differs from its tiled-swizzle mask",
        )
        for phase, raw_mask in enumerate(policy["active_cta_y_by_phase"]):
            requested = {(int(value), 0) for value in raw_mask}
            require(
                requested <= captured,
                f"kernel {ordinal}: phase {phase} requests an uncaptured CTA-y/z template",
            )
    raw_objects = template_manifest.get("objects") or []
    stable_rows = {
        int(row["template_object_index"]): row
        for row in (plan.get("binding_receipt") or {}).get("stable_objects") or []
        if isinstance(row, dict) and "template_object_index" in row
    }
    object_metadata = []
    for index, row in enumerate(raw_objects):
        stable = stable_rows.get(index, {})
        object_metadata.append({
            "source_name": str(row.get("source_name", "")),
            "stable_name": str(stable.get("stable_name", row.get("source_name", ""))),
            "kind": str(stable.get("kind", row.get("kind", ""))),
        })
    require(
        len(object_metadata) == len(source_extents),
        "compact object metadata disagrees with its extent vector",
    )
    shadow_runtime_binding = validate_shadow_runtime_binding(
        plan=plan,
        template_manifest=template_manifest,
        template_binary_path=template_binary_path,
        policies=policies,
        artifact_root=artifact_root,
    )
    return PreparedPhaseGenerator(
        source_extents=source_extents,
        target_extents=extents,
        object_metadata=object_metadata,
        policies=policies,
        by_kernel_x=by_kernel_x,
        shadow_runtime_binding=shadow_runtime_binding,
    )


def prepare_segment_generator(
    *,
    descriptor: dict[str, Any],
    template_manifest: dict[str, Any],
    template_binary_path: Path,
    artifact_root: Path | None = None,
) -> PreparedSegmentGenerator:
    """Load and validate a digest-bound lazy-plan generator descriptor."""

    require(
        descriptor.get("schema") == SEGMENT_DESCRIPTOR_SCHEMA,
        "unsupported phase-aware segment-generator schema",
    )
    require(descriptor.get("status") == "PASS", "segment generator has not passed")
    require(
        descriptor.get("kind") == SEGMENT_GENERATOR_KIND,
        "unsupported segment generator kind",
    )

    def bound_json(value: Any, *, label: str, schema: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        require(isinstance(value, dict), f"{label} binding is malformed")
        path = resolve_input(str(value.get("path", "")))
        require(path.is_file(), f"missing {label} {path}")
        require(
            value.get("sha256") == sha256_file(path),
            f"{label} digest differs from its plan binding",
        )
        loaded = load_json(path)
        require(loaded.get("schema") == schema, f"unsupported {label} schema")
        return loaded, path

    sidecar, sidecar_path = bound_json(
        descriptor.get("sidecar"), label="phase-aware sidecar", schema=SIDECAR_SCHEMA
    )
    plan, plan_path = bound_json(
        descriptor.get("generator_plan"),
        label="phase-aware generator plan",
        schema=PLAN_SCHEMA,
    )
    prepared = prepare_generator(
        template_manifest=template_manifest,
        template_binary_path=template_binary_path,
        sidecar=sidecar,
        plan=plan,
        artifact_root=artifact_root,
    )
    raw_ranges = descriptor.get("target_cta_x_ranges")
    require(
        isinstance(raw_ranges, list) and raw_ranges,
        "phase-aware segment generator has no target CTA-x ranges",
    )
    ranges: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for raw in raw_ranges:
        require(isinstance(raw, dict), "target CTA-x range is malformed")
        kernel = integer(raw.get("kernel_ordinal"), "range kernel ordinal", minimum=0)
        begin = integer(raw.get("begin"), "range begin", minimum=0)
        end = integer(raw.get("end_exclusive"), "range end", minimum=1)
        require(end > begin, "target CTA-x range is empty")
        require(kernel not in seen, f"duplicate target CTA-x range for kernel {kernel}")
        seen.add(kernel)
        ranges.append((kernel, begin, end))
    require(
        seen == set(prepared.policies),
        "target CTA-x ranges do not cover exactly the generator policies",
    )
    source_kernel_ordinals = {
        integer(row.get("ordinal"), "template kernel ordinal", minimum=0)
        for row in template_manifest.get("kernels") or []
    }
    require(
        seen == source_kernel_ordinals,
        "generator policies do not cover exactly the compact template kernels",
    )
    return PreparedSegmentGenerator(
        generator=prepared,
        ranges=sorted(ranges),
        provenance={
            "schema": SEGMENT_DESCRIPTOR_SCHEMA,
            "status": "PASS",
            "kind": SEGMENT_GENERATOR_KIND,
            "sidecar": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "generator_plan": str(plan_path),
            "generator_plan_sha256": sha256_file(plan_path),
            "target_cta_x_ranges": [
                {
                    "kernel_ordinal": kernel,
                    "begin": begin,
                    "end_exclusive": end,
                }
                for kernel, begin, end in sorted(ranges)
            ],
            "shadow_runtime_binding": prepared.shadow_runtime_binding,
        },
    )


def policy_rule_lookup(
    policy: dict[str, Any], *, label: str
) -> dict[
    tuple[int, int, int, str | None, int | None, int | None], dict[str, Any]
]:
    rules: dict[
        tuple[int, int, int, str | None, int | None, int | None], dict[str, Any]
    ] = {}
    for rule in policy.get("rules") or []:
        warp, program = rule_bundle_selector(rule)
        key = (
            int(rule["object_index"]),
            int(rule["cta_y"]),
            int(rule["cta_z"]),
            str(rule["operation"]) if "operation" in rule else None,
            warp,
            program,
        )
        require(key not in rules, f"duplicate {label} rule {key}")
        rules[key] = rule
    return rules


def checked_generated_offset(
    *, prepared: PreparedPhaseGenerator, kernel_ordinal: int,
    target_cta_x: int, object_index: int, source_offset: int,
    predicted_delta: int, byte_count: int,
) -> int:
    """Return one translated offset or fail with an actionable bound receipt."""

    final_offset = source_offset + predicted_delta
    metadata = prepared.object_metadata[object_index]
    detail = (
        f"kernel {kernel_ordinal}: target CTA x={target_cta_x}; "
        f"object_index={object_index}; "
        f"source_name={metadata['source_name']!r}; "
        f"stable_name={metadata['stable_name']!r}; kind={metadata['kind']!r}; "
        f"source_offset={source_offset}; predicted_delta={predicted_delta}; "
        f"final_offset={final_offset}; request_bytes={byte_count}; "
        f"target_extent={prepared.target_extents[object_index]}"
    )
    require(
        0 <= final_offset <= 0xFFFFFFFF,
        f"generated object offset exceeds compact u32 ({detail})",
    )
    require(
        final_offset + byte_count <= prepared.target_extents[object_index],
        f"generated request escapes its target object extent ({detail})",
    )
    return final_offset


def translated_record(
    *,
    prepared: PreparedPhaseGenerator,
    policy: dict[str, Any],
    rules: dict[
        tuple[int, int, int, str | None, int | None, int | None], dict[str, Any]
    ],
    bundle: dict[str, int],
    record: tuple[int, ...],
    coordinate: int,
) -> tuple[int, ...]:
    object_index, offset, kernel, byte_count, operation, flags = record
    require(object_index < len(prepared.target_extents),
            "compact request references an unknown object")
    require(operation in {0, 1}, "compact request has an unsupported operation")
    require(known_request_flags(flags), "compact request has unsupported flags")
    base_key = object_index, bundle["cta_y"], bundle["cta_z"]
    operation_name = OPERATION_NAME[operation]
    selector = bundle["warp_in_cta"], bundle["warp_program_ordinal"]
    rule = rules.get((*base_key, operation_name, *selector))
    if rule is None:
        rule = rules.get((*base_key, None, *selector))
    if rule is None:
        rule = rules.get((*base_key, operation_name, None, None))
    if rule is None:
        rule = rules.get((*base_key, None, None, None))
    require(
        rule is not None,
        f"kernel {kernel}: no validated causal rule for "
        f"object/yz/operation {(*base_key, operation_name)}",
    )
    source_offset = offset
    predicted_delta = rule_coordinate(rule, coordinate) - rule_coordinate(
        rule, int(policy["source_cta_x"])
    )
    offset = checked_generated_offset(
        prepared=prepared,
        kernel_ordinal=kernel,
        target_cta_x=coordinate,
        object_index=object_index,
        source_offset=source_offset,
        predicted_delta=predicted_delta,
        byte_count=byte_count,
    )
    return object_index, offset, kernel, byte_count, operation, flags


def causal_source_programs(
    *,
    prepared: PreparedPhaseGenerator,
    source: BinaryIO,
    kernel_ordinal: int,
) -> tuple[
    dict[str, Any],
    dict[
        tuple[int, int, int, str | None, int | None, int | None],
        dict[str, Any],
    ],
    list[list[tuple[dict[str, int], list[tuple[int, ...]], bool]]],
]:
    """Return validated per-warp source programs and loop membership."""

    policy = prepared.policies[kernel_ordinal]
    require(str(policy["kind"]) == "causal", "causal source requested for another policy")
    require(
        causal_is_implemented(policy),
        f"kernel {kernel_ordinal}: causal CTA generator is reserved but unimplemented; "
        "use an exact anchor until request-count and tail-mask rules pass held-out validation",
    )
    source_x = int(policy["source_cta_x"])
    selected = sorted(
        prepared.by_kernel_x.get((kernel_ordinal, source_x), []), key=bundle_order
    )
    require(bool(selected),
            f"kernel {kernel_ordinal}: source CTA x={source_x} has no captured bundles")
    rules = policy_rule_lookup(policy, label="causal")
    loop_objects = {int(value) for value in policy["loop_object_indices"]}
    groups: list[list[tuple[dict[str, int], list[tuple[int, ...]], bool]]] = []
    current_key: tuple[int, int, int] | None = None
    current: list[tuple[dict[str, int], list[tuple[int, ...]], bool]] = []
    for bundle in selected:
        key = int(bundle["cta_z"]), int(bundle["cta_y"]), int(bundle["warp_in_cta"])
        if current_key is not None and key != current_key:
            groups.append(current)
            current = []
        current_key = key
        records = read_bundle_records(source, bundle)
        membership = {int(record[0]) in loop_objects for record in records}
        require(
            len(membership) == 1,
            f"kernel {kernel_ordinal}: one source bundle mixes fixed and causal-loop objects",
        )
        current.append((bundle, records, membership == {True}))
    if current:
        groups.append(current)
    require(groups, f"kernel {kernel_ordinal}: causal source has no per-warp programs")
    for group in groups:
        loop_positions = [index for index, (_bundle, _records, loop) in enumerate(group) if loop]
        require(loop_positions, f"kernel {kernel_ordinal}: causal source warp has no loop body")
        require(
            loop_positions == list(range(loop_positions[0], loop_positions[-1] + 1)),
            f"kernel {kernel_ordinal}: causal source loop body is not contiguous",
        )
        for bundle, records, _loop in group:
            for record in records:
                # Validate rule coverage and the captured coordinate without
                # changing the source record.
                translated_record(
                    prepared=prepared,
                    policy=policy,
                    rules=rules,
                    bundle=bundle,
                    record=record,
                    coordinate=source_x,
                )
    return policy, rules, groups


def causal_generated_bundle_records(
    *,
    prepared: PreparedPhaseGenerator,
    source: BinaryIO,
    kernel_ordinal: int,
    target_cta_x: int,
) -> Iterator[tuple[dict[str, int], list[tuple[int, ...]]]]:
    policy, rules, groups = causal_source_programs(
        prepared=prepared, source=source, kernel_ordinal=kernel_ordinal
    )
    maximum = int(policy["validated_target_cta_x_max"])
    require(
        target_cta_x <= maximum,
        f"kernel {kernel_ordinal}: causal target CTA x={target_cta_x} exceeds "
        f"validated maximum {maximum}",
    )
    for group in groups:
        loop_positions = [index for index, (_bundle, _records, loop) in enumerate(group) if loop]
        loop_begin, loop_end = loop_positions[0], loop_positions[-1] + 1
        emitted_program = 0

        def emit(
            rows: list[tuple[dict[str, int], list[tuple[int, ...]], bool]],
            coordinate: int,
        ) -> Iterator[tuple[dict[str, int], list[tuple[int, ...]]]]:
            nonlocal emitted_program
            for bundle, records, _loop in rows:
                translated = [
                    translated_record(
                        prepared=prepared,
                        policy=policy,
                        rules=rules,
                        bundle=bundle,
                        record=record,
                        coordinate=coordinate,
                    )
                    for record in records
                ]
                generated_bundle = dict(bundle)
                generated_bundle["cta_x"] = target_cta_x
                generated_bundle["warp_program_ordinal"] = emitted_program
                emitted_program += 1
                yield generated_bundle, translated

        yield from emit(group[:loop_begin], target_cta_x)
        loop_body = group[loop_begin:loop_end]
        for tile in range(target_cta_x, -1, -1):
            yield from emit(loop_body, tile)
        yield from emit(group[loop_end:], target_cta_x)


def causal_generation_census(
    *,
    prepared: PreparedPhaseGenerator,
    source: BinaryIO,
    kernel_ordinal: int,
    begin: int,
    end: int,
) -> tuple[Counter[str], dict[int, Counter[str]]]:
    """Count a causal grid analytically instead of materializing O(n^2) requests."""

    require(end > begin, "causal census range is empty")
    policy, rules, groups = causal_source_programs(
        prepared=prepared, source=source, kernel_ordinal=kernel_ordinal
    )
    maximum = int(policy["validated_target_cta_x_max"])
    require(
        end - 1 <= maximum,
        f"kernel {kernel_ordinal}: causal range ends beyond validated maximum {maximum}",
    )
    fixed_multiplier = end - begin
    loop_multiplier = sum(target + 1 for target in range(begin, end))
    totals: Counter[str] = Counter()
    objects: dict[int, Counter[str]] = {}
    active_yz = {
        (int(bundle["cta_y"]), int(bundle["cta_z"]))
        for group in groups
        for bundle, _records, _loop in group
    }
    totals["target_ctas"] = fixed_multiplier
    totals["active_grid_ctas"] = len(active_yz) * fixed_multiplier
    for group in groups:
        for bundle, records, loop in group:
            multiplier = loop_multiplier if loop else fixed_multiplier
            coordinates = {0, end - 1} if loop else {begin, end - 1}
            for record in records:
                for coordinate in coordinates:
                    translated_record(
                        prepared=prepared,
                        policy=policy,
                        rules=rules,
                        bundle=bundle,
                        record=record,
                        coordinate=coordinate,
                    )
                object_index, _offset, _kernel, byte_count, operation, _flags = record
                counter = objects.setdefault(object_index, Counter())
                counter["requests"] += multiplier
                counter["bytes"] += byte_count * multiplier
                counter["r_requests" if operation == 0 else "w_requests"] += multiplier
                counter["r_bytes" if operation == 0 else "w_bytes"] += byte_count * multiplier
                totals["requests"] += multiplier
                totals["bytes"] += byte_count * multiplier
    return totals, objects


def segment_generation_census(
    *, prepared: PreparedSegmentGenerator, source_path: Path
) -> dict[str, Any]:
    """Compute output request/byte totals without materializing repeated CTAs."""

    totals: Counter[str] = Counter()
    by_object: dict[int, Counter[str]] = {}
    with source_path.open("rb") as source:
        for kernel_ordinal, begin, end in prepared.ranges:
            policy = prepared.generator.policies[kernel_ordinal]
            kind = str(policy["kind"])
            if kind == SEMANTIC_EMBEDDING_KIND:
                semantic_totals, semantic_objects = semantic_embedding_range_census(
                    policy["_semantic_program"], begin=begin, end=end
                )
                totals.update(semantic_totals)
                for object_index, counter in semantic_objects.items():
                    by_object.setdefault(object_index, Counter()).update(counter)
                continue
            if kind == "causal":
                causal_totals, causal_objects = causal_generation_census(
                    prepared=prepared.generator,
                    source=source,
                    kernel_ordinal=kernel_ordinal,
                    begin=begin,
                    end=end,
                )
                totals.update(causal_totals)
                for object_index, counter in causal_objects.items():
                    by_object.setdefault(object_index, Counter()).update(counter)
                continue
            if kind != "exact_anchor":
                # The census multiplies one constant-size CTA template instead
                # of materializing every target CTA.  Validate the address
                # extrema first: affine rules need both endpoints; period-2
                # rules need the first/last member of each parity.  This closes
                # the prior hole where CTA ``begin`` fit an object but a far
                # target CTA could escape its target extent only at streaming
                # time after a partial output had already been written.
                validation_points = {begin, end - 1}
                period = policy_period(policy)
                if period > 1:
                    for phase in range(period):
                        first = begin + ((phase - begin) % period)
                        last = end - 1 - (((end - 1) - phase) % period)
                        if first < end:
                            validation_points.add(first)
                        if last >= begin:
                            validation_points.add(last)
                for target_cta_x in sorted(validation_points):
                    for _bundle, records in generated_bundle_records(
                        prepared=prepared.generator,
                        source=source,
                        kernel_ordinal=kernel_ordinal,
                        target_cta_x=target_cta_x,
                    ):
                        # Exhausting the iterator performs rule, flag, and
                        # target-object-extent checks for every request.
                        for _record in records:
                            pass
            if kind == "exact_anchor":
                target_groups = [(target_cta_x, 1) for target_cta_x in range(begin, end)]
            elif kind == "tiled_swizzle":
                period = int(policy["period"])
                target_groups = []
                for phase in range(period):
                    first = begin + ((phase - begin) % period)
                    if first < end:
                        multiplier = 1 + (end - 1 - first) // period
                        target_groups.append((first, multiplier))
            else:
                target_groups = [(begin, end - begin)]
            for target_cta_x, multiplier in target_groups:
                local_requests = 0
                local_bytes = 0
                local_ctas: set[tuple[int, int]] = set()
                for _bundle, records in generated_bundle_records(
                    prepared=prepared.generator,
                    source=source,
                    kernel_ordinal=kernel_ordinal,
                    target_cta_x=target_cta_x,
                ):
                    local_ctas.add((int(_bundle["cta_y"]), int(_bundle["cta_z"])))
                    for record in records:
                        object_index, _offset, _kernel, byte_count, operation, _flags = record
                        counter = by_object.setdefault(object_index, Counter())
                        counter["requests"] += multiplier
                        counter["bytes"] += byte_count * multiplier
                        counter["r_requests" if operation == 0 else "w_requests"] += multiplier
                        counter["r_bytes" if operation == 0 else "w_bytes"] += byte_count * multiplier
                        local_requests += multiplier
                        local_bytes += byte_count * multiplier
                totals["requests"] += local_requests
                totals["bytes"] += local_bytes
                totals["active_grid_ctas"] += len(local_ctas) * multiplier
            totals["target_ctas"] += end - begin
    return {
        "totals": dict(sorted(totals.items())),
        "by_object": {
            str(index): dict(sorted(counter.items()))
            for index, counter in sorted(by_object.items())
        },
    }


def generated_bundle_records(
    *,
    prepared: PreparedPhaseGenerator,
    source: BinaryIO,
    kernel_ordinal: int,
    target_cta_x: int,
) -> Iterator[tuple[dict[str, int], list[tuple[int, ...]]]]:
    """Yield translated request records for one kernel and one target CTA-x.

    The yielded bundle order is deterministic and preserves each warp's
    program order.  It deliberately does not claim the production GPU's
    cross-CTA scheduling order.
    """

    target_cta_x = integer(target_cta_x, "target CTA x", minimum=0)
    policy = prepared.policies.get(kernel_ordinal)
    require(policy is not None, f"kernel {kernel_ordinal}: generator policy is absent")
    kind = str(policy["kind"])
    if kind == SEMANTIC_EMBEDDING_KIND:
        records = list(iter_semantic_embedding_cta_records(
            policy["_semantic_program"], cta_x=target_cta_x
        ))
        require(records, f"kernel {kernel_ordinal}: semantic embedding CTA is empty")
        yield ({
            "bundle_index": target_cta_x,
            "kernel_ordinal": kernel_ordinal,
            "cta_x": target_cta_x,
            "cta_y": 0,
            "cta_z": 0,
            "warp_in_cta": 0,
            "warp_program_ordinal": 0,
            "request_ordinal_begin": 0,
            "request_ordinal_end_exclusive": len(records),
        }, records)
        return
    if kind == "causal":
        yield from causal_generated_bundle_records(
            prepared=prepared,
            source=source,
            kernel_ordinal=kernel_ordinal,
            target_cta_x=target_cta_x,
        )
        return
    if kind == "exact_anchor":
        captured = {int(value) for value in policy["captured_cta_x"]}
        require(
            target_cta_x in captured,
            f"kernel {kernel_ordinal}: exact anchor for CTA x={target_cta_x} is absent",
        )
        source_x = target_cta_x
        rules: dict[
            tuple[int, int, int, str | None, int | None, int | None],
            dict[str, Any],
        ] = {}
    else:
        source_x = int(policy["source_cta_x"])
        rules = policy_rule_lookup(policy, label="phase")
    rule_deltas = {
        id(rule): rule_coordinate(rule, target_cta_x)
        - rule_coordinate(rule, source_x)
        for rule in rules.values()
    }

    selected = sorted(
        prepared.by_kernel_x.get((kernel_ordinal, source_x), []), key=bundle_order
    )
    require(
        bool(selected),
        f"kernel {kernel_ordinal}: source CTA x={source_x} has no captured bundles",
    )
    active_y = active_y_for_target(policy, target_cta_x)
    for bundle in selected:
        if active_y is not None and (
            int(bundle["cta_z"]) != 0 or int(bundle["cta_y"]) not in active_y
        ):
            continue
        output: list[tuple[int, ...]] = []
        for record in read_bundle_records(source, bundle):
            object_index, offset, kernel, byte_count, operation, flags = record
            require(kernel == kernel_ordinal, "bundle request uses another kernel ordinal")
            require(
                object_index < len(prepared.target_extents),
                "compact request references an unknown object",
            )
            require(operation in {0, 1}, "compact request has an unsupported operation")
            require(known_request_flags(flags), "compact request has unsupported flags")
            source_offset = offset
            predicted_delta = 0
            if kind in {"phase_rules", "tiled_swizzle"}:
                base_key = object_index, bundle["cta_y"], bundle["cta_z"]
                operation_name = OPERATION_NAME[operation]
                selector = bundle["warp_in_cta"], bundle["warp_program_ordinal"]
                rule = rules.get((*base_key, operation_name, *selector))
                if rule is None:
                    rule = rules.get((*base_key, None, *selector))
                if rule is None:
                    rule = rules.get((*base_key, operation_name, None, None))
                if rule is None:
                    rule = rules.get((*base_key, None, None, None))
                require(
                    rule is not None,
                    f"kernel {kernel_ordinal}: no validated phase rule for "
                    f"object/yz/operation {(*base_key, operation_name)}",
                )
                predicted_delta = rule_deltas[id(rule)]
            offset = checked_generated_offset(
                prepared=prepared,
                kernel_ordinal=kernel_ordinal,
                target_cta_x=target_cta_x,
                object_index=object_index,
                source_offset=source_offset,
                predicted_delta=predicted_delta,
                byte_count=byte_count,
            )
            output.append(
                (object_index, offset, kernel, byte_count, operation, flags)
            )
        yield bundle, output


def generate(
    *,
    template_manifest_path: Path,
    template_binary_path: Path,
    sidecar_path: Path,
    plan_path: Path,
    target_cta_x: int,
    output_path: Path,
    output_manifest_path: Path,
    output_sidecar_path: Path,
) -> dict[str, Any]:
    target_cta_x = integer(target_cta_x, "target CTA x", minimum=0)
    paths = [output_path, output_manifest_path, output_sidecar_path]
    for path in paths:
        require(not path.exists(), f"refusing to overwrite {path}")
    manifest = load_json(template_manifest_path)
    sidecar = load_json(sidecar_path)
    plan = load_json(plan_path)
    prepared = prepare_generator(
        template_manifest=manifest,
        template_binary_path=template_binary_path,
        sidecar=sidecar,
        plan=plan,
    )

    emitted_bundles: list[dict[str, int]] = []
    totals: Counter[str] = Counter()
    digest = hashlib.sha256()
    for kernel_ordinal, policy in sorted(prepared.policies.items()):
        kind = str(policy["kind"])
        if kind == "causal" and not causal_is_implemented(policy):
            raise PhaseAwareGenerationError(
                f"kernel {kernel_ordinal}: causal CTA generator is reserved but unimplemented; "
                "use an exact anchor until request-count and tail-mask rules pass held-out validation"
            )
        if kind == "exact_anchor":
            captured = {int(value) for value in policy["captured_cta_x"]}
            require(
                target_cta_x in captured,
                f"kernel {kernel_ordinal}: exact anchor for CTA x={target_cta_x} is absent",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with template_binary_path.open("rb") as source, output_path.open("xb") as target:
            for kernel_ordinal, _policy in sorted(prepared.policies.items()):
                for bundle, records in generated_bundle_records(
                    prepared=prepared,
                    source=source,
                    kernel_ordinal=kernel_ordinal,
                    target_cta_x=target_cta_x,
                ):
                    request_begin = int(totals["requests"])
                    for record in records:
                        object_index, offset, kernel, byte_count, operation, flags = record
                        payload = RECORD_STRUCT.pack(
                            object_index, offset, kernel, byte_count, operation, flags
                        )
                        target.write(payload)
                        digest.update(payload)
                        totals["requests"] += 1
                        totals["bytes"] += byte_count
                        totals["read_requests" if operation == 0 else "write_requests"] += 1
                        totals["read_bytes" if operation == 0 else "write_bytes"] += byte_count
                    emitted_bundles.append(
                        {
                            "bundle_index": len(emitted_bundles),
                            "kernel_ordinal": kernel_ordinal,
                            "cta_x": target_cta_x,
                            "cta_y": bundle["cta_y"],
                            "cta_z": bundle["cta_z"],
                            "warp_in_cta": bundle["warp_in_cta"],
                            "warp_program_ordinal": bundle["warp_program_ordinal"],
                            "request_ordinal_begin": request_begin,
                            "request_ordinal_end_exclusive": int(totals["requests"]),
                        }
                    )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    output_sidecar = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "classification": "generated CTA/warp/program compact request-range sidecar",
        "source": {
            "compact_binary": str(output_path.resolve()),
            "compact_binary_sha256": digest.hexdigest(),
            "compact_requests": int(totals["requests"]),
            "generator_plan": str(plan_path.resolve()),
            "generator_plan_sha256": sha256_file(plan_path),
            "input_sidecar": str(sidecar_path.resolve()),
            "input_sidecar_sha256": sha256_file(sidecar_path),
        },
        "target_cta_x": target_cta_x,
        "bundles": emitted_bundles,
        "ordering": "kernel,cta_z,cta_y,warp,warp_program_ordinal",
        "not_claimed": ["production cross-CTA scheduling order", "per-request issue timestamps"],
    }
    output_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    output_sidecar_path.write_text(
        json.dumps(output_sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "PASS",
        "classification": "phase-aware generated standard 12-byte compact requests",
        "target_cta_x": target_cta_x,
        "record_bytes": RECORD_BYTES,
        "output": str(output_path.resolve()),
        "output_sha256": digest.hexdigest(),
        "output_binary_bytes": output_path.stat().st_size,
        "output_sidecar": str(output_sidecar_path.resolve()),
        "output_sidecar_sha256": sha256_file(output_sidecar_path),
        "totals": dict(sorted(totals.items())),
        "kernel_generators": [
            {
                "kernel_ordinal": ordinal,
                "kind": str(policy["kind"]),
                "source_cta_x": policy.get("source_cta_x"),
            }
            for ordinal, policy in sorted(prepared.policies.items())
        ],
        "target_object_extents_bytes": prepared.target_extents,
        "inputs": {
            "template_manifest": str(template_manifest_path.resolve()),
            "template_manifest_sha256": sha256_file(template_manifest_path),
            "template_binary": str(template_binary_path.resolve()),
            "template_binary_sha256": sha256_file(template_binary_path),
            "sidecar": str(sidecar_path.resolve()),
            "sidecar_sha256": sha256_file(sidecar_path),
            "generator_plan": str(plan_path.resolve()),
            "generator_plan_sha256": sha256_file(plan_path),
        },
        "not_claimed": [
            "production cross-CTA scheduling order",
            "per-request issue timestamps",
            "post-cache or physical HBF traffic",
        ],
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--template-binary", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--generator-plan", type=Path, required=True)
    parser.add_argument("--target-cta-x", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    args = parser.parse_args()
    result = generate(
        template_manifest_path=args.template_manifest,
        template_binary_path=args.template_binary,
        sidecar_path=args.sidecar,
        plan_path=args.generator_plan,
        target_cta_x=args.target_cta_x,
        output_path=args.output,
        output_manifest_path=args.output_manifest,
        output_sidecar_path=args.output_sidecar,
    )
    print(json.dumps({
        "status": result["status"],
        "target_cta_x": result["target_cta_x"],
        "requests": result["totals"]["requests"],
        "bytes": result["totals"]["bytes"],
        "output_sha256": result["output_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

"""Explicit static placement and independent training-window access profiles."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from hbserve.contracts import HBServeError
from hbserve.io import load_json_object, write_json_atomic
from hbserve.windows.memory_trace import canonical_sha256


POLICIES = ("capacity_balanced", "weights_first", "kv_first", "profiled_hotset")
PROFILE_SCHEMA = {"name": "hbserve.window_access_profile", "version": 1}


def union_bytes(intervals: Sequence[tuple[int, int]]) -> int:
    total = 0
    previous_end = 0
    for begin, end in sorted(intervals):
        total += max(0, end - max(begin, previous_end))
        previous_end = max(previous_end, end)
    return total


def access_profile(context, granularity: int, *, role: str) -> dict:
    if role not in {"training", "measurement"}:
        raise HBServeError("access profile role must be training or measurement")
    if granularity < 4096 or granularity % 4096:
        raise HBServeError("profile granularity must be a positive multiple of 4096")
    units = defaultdict(lambda: {"read_bytes": 0, "write_bytes": 0})
    intervals = []
    writes = []
    phase_digests = []
    for phase in context.trace.phases:
        phase_digests.append(phase.trace_group.digest)
        for transaction in phase.trace_group.memory_transactions:
            end = transaction.addr + transaction.bytes
            intervals.append((transaction.addr, end))
            if transaction.op == "W":
                writes.append((transaction.addr, end))
            field = "read_bytes" if transaction.op == "R" else "write_bytes"
            first = transaction.addr // granularity
            last = (end - 1) // granularity
            for unit in range(first, last + 1):
                units[unit][field] += min(end, (unit + 1) * granularity) - max(transaction.addr, unit * granularity)
    return {
        "schema": PROFILE_SCHEMA, "role": role,
        "layout_sha256": context.layout.digest,
        "trace_sha256": context.trace.digest,
        "logical_trace_sha256": canonical_sha256(phase_digests),
        "locality_seed": context.trace.workload["locality_seed"],
        "granularity_bytes": granularity,
        "read_bytes": context.trace.read_bytes, "write_bytes": context.trace.write_bytes,
        "unique_accessed_bytes": union_bytes(intervals),
        "unique_written_bytes": union_bytes(writes),
        "accessed_placement_bytes": sum(
            min(granularity, context.layout.address_space_bytes - unit * granularity)
            for unit in units),
        "units": [{"unit": unit, **units[unit]} for unit in sorted(units)],
    }


def placement_order(layout, config: Mapping, granularity: int, trace_sha256: str,
                    *, logical_trace_sha256: str | None = None):
    policy = config.get("policy", "capacity_balanced")
    if policy not in POLICIES:
        raise HBServeError(f"unknown direct placement policy: {policy}")
    if policy == "capacity_balanced":
        if "profile" in config:
            raise HBServeError("only profiled_hotset consumes a training profile")
        return None, {}
    total_units = (layout.address_space_bytes + granularity - 1) // granularity
    detail = {"static_policy": policy, "online_migration": False,
              "measurement_trace_used_for_placement": False}
    if policy == "profiled_hotset":
        if not isinstance(config.get("profile"), str) or not config["profile"]:
            raise HBServeError("profiled_hotset requires an independent training profile path")
        profile_path = Path(config["profile"])
        profile = load_json_object(profile_path, "training access profile")
        if (profile.get("schema") != PROFILE_SCHEMA or profile.get("role") != "training"
                or profile.get("layout_sha256") != layout.digest
                or profile.get("granularity_bytes") != granularity):
            raise HBServeError("training profile schema, role, layout or granularity differs")
        if not logical_trace_sha256:
            raise HBServeError("profiled_hotset requires the measurement logical trace digest")
        if (profile.get("trace_sha256") == trace_sha256
                or profile.get("logical_trace_sha256") == logical_trace_sha256):
            raise HBServeError("the measurement trace cannot be its own training profile")
        counts = {}
        for item in profile["units"]:
            unit = item["unit"]
            if (isinstance(unit, bool) or not isinstance(unit, int) or
                    not 0 <= unit < total_units or unit in counts):
                raise HBServeError("training profile has invalid or repeated placement units")
            values = (item["read_bytes"], item["write_bytes"])
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise HBServeError("training access counts must be nonnegative integers")
            counts[unit] = values

        def score(unit):
            reads, writes = counts.get(unit, (0, 0))
            return (-int(writes > 0), -(reads + writes), unit)

        order = sorted(range(total_units), key=score)
        detail.update({"training_profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                       "training_trace_sha256": profile["trace_sha256"],
                       "training_logical_trace_sha256": profile["logical_trace_sha256"],
                       "training_locality_seed": profile["locality_seed"],
                       "priority": "written_units_then_training_byte_frequency_then_address",
                       "training_cost_included_in_measurement": False})
        detail["planning_inputs"] = ["independent_training_access_counts", "canonical_address"]
    else:
        if "profile" in config:
            raise HBServeError("only profiled_hotset consumes a training profile")
        classes = ("metadata", "immutable_weight", "kv") if policy == "weights_first" else ("metadata", "kv", "immutable_weight")
        units = []
        for placement_class in classes:
            for region in layout.regions:
                if region.placement_class == placement_class:
                    units.extend(range(region.begin // granularity, (region.end - 1) // granularity + 1))
        order = list(dict.fromkeys((*units, *range(total_units))))
        detail["priority"] = list(classes)
        detail["planning_inputs"] = ["layout_placement_class", "canonical_address"]
    detail["priority_order_sha256"] = canonical_sha256(order)
    return order, detail


def main(argv=None) -> int:
    from hbserve.windows.experiment import load_experiment_context

    parser = argparse.ArgumentParser(description="Profile a separate training window without executing the simulator")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    context = load_experiment_context(arguments.experiment)
    if context.experiment["mapping"].get("direct_placement", {}).get("policy", "capacity_balanced") != "capacity_balanced":
        parser.error("training must be generated independently of the measured placement policy")
    granularity = context.experiment["mapping"]["placement_granularity_bytes"]
    profile = access_profile(context, granularity, role="training")
    write_json_atomic(arguments.output, profile)
    print(f"HBServe training profile: {arguments.output.resolve()}")
    return 0

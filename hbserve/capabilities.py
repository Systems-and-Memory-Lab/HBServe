"""Machine-readable HBServe capability boundary for workload selection."""

from __future__ import annotations

from typing import Any


CAPABILITY_SCHEMA = {"name": "hbserve.capabilities", "version": 2}


def current_capabilities() -> dict[str, Any]:
    """Return the implemented behavior of the one serving path, not a roadmap."""

    return {
        "schema": CAPABILITY_SCHEMA,
        "request_fields": [
            "request_id",
            "arrival_ns",
            "model_id",
            "prompt_tokens",
            "output_tokens",
            "token_ids_optional",
        ],
        "conversation_ancestry": False,
        "prefix_block_hash_identity": False,
        "prefix_cache_lifecycle": False,
        "scheduler": "token_budgeted_mixed_iteration_continuous_batcher_v1",
        "kv_allocation_policy": "paged_blocks_incremental_v1",
        "kv_placement_targets": {"hot": ["hbm"], "cold": ["hbf", "external"]},
        "kv_migration_policy": "whole_request_coldest_first_v1",
        "preemption_policy": "youngest_request_first_swap_or_recompute_v1",
        "timing_models": ["roofline", "memory_only", "linear"],
        "compute_prefetch_overlap": "next_layer_memory_during_compute",
        "physical_feedback": True,
        "dense_models": True,
        "moe_models": True,
        "model_catalog_converter": "hbserve.catalog",
    }

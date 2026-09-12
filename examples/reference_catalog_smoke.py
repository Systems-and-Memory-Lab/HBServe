#!/usr/bin/env python3
"""Regenerate a retained 64-request CTA anchor, not a whole 32B inference.

This checks packaging and source consumption. It does not newly validate the
GPU-cache model, extrapolation, full-model coverage or native layout mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hbserve.traces.catalog import export_entry, read_catalog
from hbserve.traces.common import load_json, new_output, save
from hbserve.traces._reference.full_model_trace_plan import build_plan
from hbserve.traces.reference import generate

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ID = "template-d47846e781b6b8e8"


def run(catalog: Path, output: Path) -> dict:
    root = new_output(output)
    document = read_catalog(catalog)
    entry, = [row for row in document["templates"] if row["id"] == TEMPLATE_ID]
    export_entry(catalog=catalog, entry_id=TEMPLATE_ID, output=root / "source")
    mapping = load_json(root / "source/artifact-map.json")["artifacts"]
    plan = build_plan(
        template_manifest_path=root / "source" / mapping[entry["source"]]["path"],
        template_binary_path=root / "source" / mapping[entry["binary_source"]]["path"],
        model_id="Qwen3-32B-K12-CTA0-source-window-only", layers=1, template_layer=0,
        phase="prefill", context_tokens=8192, batch=1, activation_policy="shared_template_arena",
    )
    plan["coverage"] = {
        "status": "CAPTURED_CTA_ANCHOR_ONLY_NOT_A_FULL_LAYER_OR_MODEL",
        "missing": ["other CTAs", "other kernels and layers", "full-model inference"],
    }
    plan["publication_smoke"] = {
        "source_template_id": TEMPLATE_ID, "source_scope": entry["label"],
        "new_validation": False, "layer_count_means": "one source-window instance, not the real model depth",
    }
    save(root / "source-window.plan.json", plan)
    result = generate(plan=root / "source-window.plan.json",
                      cache_config=load_json(ROOT / "examples/reference-cache.json"),
                      output=root / "generated")
    receipt = {"status": "PASS_RETAINED_CTA_SOFTWARE_SMOKE", "template_id": TEMPLATE_ID,
               "counts": result["cache"]["counts"],
               "post_cache_sha256": result["cache"]["output_sha256"],
               "generation_seconds": result["generation_seconds"],
               "full_model": False, "new_hardware_fidelity_validation": False}
    save(root / "smoke.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "reference_templates/catalog.json")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.catalog, args.output_root), indent=2, sort_keys=True))

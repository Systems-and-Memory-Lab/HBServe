#!/usr/bin/env python3
"""Create a tiny SYNTHETIC trace fixture; this is not a hardware capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from hbserve.traces.common import new_output, save, sha256_file
from hbserve.traces._reference.compact_request_template import RECORD_STRUCT, SCHEMA
from hbserve.traces._reference.full_model_trace_plan import build_plan


def create_fixture(output: Path, *, phase: str = "decode") -> Path:
    root = new_output(output)
    records = [
        (0, 0, 0, 32, 0, 0),
        (0, 0, 0, 32, 0, 0),
        (1, 0, 0, 32, 0, 0),
        (1, 32, 1, 32, 1, 0),
        (2, 64, 1, 32, 1, 0),
        (2, 64, 1, 32, 0, 0),
    ]
    binary = root / "template.bin"
    binary.write_bytes(b"".join(RECORD_STRUCT.pack(*row) for row in records))
    manifest = root / "template.json"
    save(manifest, {
        "schema": SCHEMA, "status": "PASS", "requests": len(records),
        "request_bytes": len(records) * 32, "binary_bytes": binary.stat().st_size,
        "binary_sha256": sha256_file(binary),
        "evidence": "synthetic contract fixture, not measured hardware",
        "kernels": [{"ordinal": 0}, {"ordinal": 1}],
        "objects": [
            {"template_object_index": index, "kind": kind, "source_name": name,
             "bytes": 4096, "compiled": {"requests": 2, "bytes": 64}}
            for index, (kind, name) in enumerate([
                ("weight", "model.layers.0.weight"),
                ("kv_cache", "kv"), ("activation", "temporary"),
            ])
        ],
    })
    plan = build_plan(
        template_manifest_path=manifest, template_binary_path=binary,
        model_id="synthetic-contract-fixture", layers=2, phase=phase,
        context_tokens=4, batch=1, activation_policy="shared_template_arena",
        template_layer=0,
    )
    plan["fixture_evidence"] = "synthetic only; no hardware holdout validation"
    path = root / "plan.json"
    save(path, plan)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    args = parser.parse_args()
    print(create_fixture(args.output_root, phase=args.phase))

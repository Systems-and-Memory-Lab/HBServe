#!/usr/bin/env python3
"""Run the compact reference-cache transform with the validated C++ engine.

This is an execution-speed replacement for ``cache_compact_trace.py``.  The
plan remains interpreted in Python, while the engine receives only a dense
``index/base/extent`` object table plus the fixed-width compact stream.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, BinaryIO

from hbserve.traces._reference.cache_compact_trace import SCHEMA
from hbserve.traces._reference.compact_request_template import load_json, require, sha256_file
from hbserve.traces._reference.full_model_trace_plan import PLAN_SCHEMA


ENGINE_SCHEMA = {"name": "hbfsim.fast_compact_cache_engine", "version": 1}


def _context(path: Path | None, stream: BinaryIO | None, mode: str):
    require((path is None) != (stream is None), "provide exactly one path or stream")
    return path.open(mode) if path is not None else nullcontext(stream)


def _artifact_path(manifest: Path, suffix: str) -> Path:
    name = manifest.name.removesuffix(".json")
    return manifest.parent / f"{name}.{suffix}"


def _nonzero(raw: dict[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(raw.items())
        if int(value) != 0
    }


def transform_fast(
    *,
    plan_path: Path,
    input_path: Path | None,
    input_stream: BinaryIO | None,
    output_path: Path | None,
    output_stream: BinaryIO | None,
    manifest_path: Path,
    engine_path: Path,
    capacity_bytes: int,
    line_bytes: int,
    sector_bytes: int,
    associativity: int,
    write_policy: str,
    write_allocate: bool,
    write_miss_fetch: bool,
    final_drain: bool,
    affine_prefix_input_records: int = 0,
    affine_period_input_records: int = 0,
    affine_period_object_deltas_path: Path | None = None,
    affine_read_template_output: Path | None = None,
    affine_writeback_fallback_output: Path | None = None,
    stage_boundaries_path: Path | None = None,
    timed_input_output: bool = False,
    read_fill_bytes: int = 0,
    writeback_bytes: int = 0,
    read_fill_object_boundary: str = "reject",
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    manifest_path = manifest_path.resolve()
    engine_path = engine_path.resolve()
    plan = load_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported lazy-plan schema")
    objects = plan.get("objects")
    require(isinstance(objects, list) and objects, "plan has no object table")
    require(
        [int(row["target_object_index"]) for row in objects]
        == list(range(len(objects))),
        "plan target object indices are not dense",
    )
    require(engine_path.is_file(), f"fast cache engine is missing: {engine_path}")
    require(write_policy in {"write-back", "write-through"}, "unsupported write policy")
    require(type(read_fill_bytes) is int and read_fill_bytes in {0, 64}, "unsupported read fill size")
    require(type(writeback_bytes) is int and writeback_bytes in {0, 64}, "unsupported writeback size")
    require(read_fill_object_boundary == "reject", "fills must remain in owned objects")
    require(isinstance(timed_input_output, bool),
            "timed_input_output must be boolean")
    require(affine_prefix_input_records >= 0, "affine prefix must be nonnegative")
    require(affine_period_input_records >= 0, "affine period must be nonnegative")
    require(
        (affine_period_input_records > 0)
        == (affine_period_object_deltas_path is not None),
        "affine period and object-delta table must be supplied together",
    )
    require(
        (affine_read_template_output is None)
        == (affine_writeback_fallback_output is None),
        "affine template and fallback outputs must be supplied together",
    )
    require(
        affine_prefix_input_records == 0 or affine_period_input_records > 0,
        "affine prefix requires an affine period",
    )
    if stage_boundaries_path is not None:
        stage_boundaries_path = stage_boundaries_path.resolve()
        require(stage_boundaries_path.is_file(), "stage boundary table is missing")
        require(not final_drain,
                "stage-boundary audit requires no unassigned final drain")
        require(affine_period_input_records == 0,
                "stage-boundary and affine-period audits cannot be combined")
    require(not (timed_input_output and affine_period_input_records),
            "timed input/output cannot emit affine-period artifacts")
    require(not (timed_input_output and final_drain),
            "timed input/output requires final_drain=false")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    object_table = _artifact_path(manifest_path, "objects.tsv")
    engine_stats = _artifact_path(manifest_path, "engine.json")
    object_table.write_text(
        "".join(
            f"{index}\t{int(row['logical_address'])}\t{int(row['bytes'])}\n"
            for index, row in enumerate(objects)
        ),
        encoding="ascii",
    )
    command = [
        str(engine_path),
        "--objects", str(object_table),
        "--stats", str(engine_stats),
        "--capacity-bytes", str(capacity_bytes),
        "--line-bytes", str(line_bytes),
        "--sector-bytes", str(sector_bytes),
        "--associativity", str(associativity),
        "--write-policy", write_policy,
        "--write-allocate", "true" if write_allocate else "false",
        "--write-miss-fetch", "true" if write_miss_fetch else "false",
        "--final-drain", "true" if final_drain else "false",
        "--timed-input-output", "true" if timed_input_output else "false",
    ]
    if read_fill_bytes or writeback_bytes:
        command += ["--read-fill-bytes", str(read_fill_bytes),
                    "--writeback-bytes", str(writeback_bytes),
                    "--read-fill-object-edge", read_fill_object_boundary]
    if affine_period_input_records:
        assert affine_period_object_deltas_path is not None
        command += [
            "--affine-prefix-input-records", str(affine_prefix_input_records),
            "--affine-period-input-records", str(affine_period_input_records),
            "--affine-period-object-deltas",
            str(affine_period_object_deltas_path.resolve()),
        ]
    if affine_read_template_output is not None:
        assert affine_writeback_fallback_output is not None
        command += [
            "--affine-read-template-output",
            str(affine_read_template_output.resolve()),
            "--affine-writeback-fallback-output",
            str(affine_writeback_fallback_output.resolve()),
        ]
    if stage_boundaries_path is not None:
        command += ["--stage-boundaries", str(stage_boundaries_path)]
    started = time.perf_counter()
    with _context(input_path, input_stream, "rb") as source, _context(
        output_path, output_stream, "wb"
    ) as target:
        assert source is not None and target is not None
        completed = subprocess.run(
            command,
            stdin=source,
            stdout=target,
            stderr=subprocess.PIPE,
            check=False,
        )
    elapsed = time.perf_counter() - started
    require(
        completed.returncode == 0,
        "fast cache engine failed: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )
    engine = load_json(engine_stats)
    require(engine.get("schema") == ENGINE_SCHEMA, "fast cache engine schema changed")
    require(engine.get("status") == "PASS", "fast cache engine did not pass")
    counts = _nonzero(engine.get("counts") or {})
    cache_stats = _nonzero(engine.get("cache_stats") or {})
    # ReferenceLRUCache.drain() touches this Counter key even when the cache
    # contains no dirty sectors.  Preserve that observable manifest detail.
    if final_drain:
        cache_stats.setdefault("final_dirty_sector_writebacks", 0)
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": (
            "streaming named-reference-cache compact request transform; "
            "C++ execution engine"
        ),
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "input": str(input_path.resolve()) if input_path is not None else "stream",
        "input_sha256": str(engine["input_sha256"]),
        "output": str(output_path.resolve()) if output_path is not None else "stream",
        "output_sha256": str(engine["output_sha256"]),
        "record_bytes": 20 if timed_input_output else 12,
        "untimed_record_bytes": 12,
        "timed_input_output": timed_input_output,
        "input_untimed_sha256": str(engine["input_untimed_sha256"]),
        "output_untimed_sha256": str(engine["output_untimed_sha256"]),
        "counts": counts,
        "cache": {
            "mode": "reference-lru",
            "capacity_bytes": capacity_bytes,
            "line_bytes": line_bytes,
            "sector_bytes": sector_bytes,
            "associativity": associativity,
            "write_policy": write_policy,
            "read_admission": "sass-ef-lru",
            "write_allocate": write_allocate,
            "write_miss_fetch": write_miss_fetch,
            "final_drain": final_drain,
            "stats": cache_stats,
            "final_state": engine["final_state"],
        },
        "output_run_census": engine.get("output_run_census"),
        "affine_period_audit": engine.get("affine_period_audit"),
        "stage_audit": engine.get("stage_audit"),
        "engine": {
            "binary": str(engine_path),
            "binary_sha256": sha256_file(engine_path),
            "object_table": str(object_table),
            "object_table_sha256": sha256_file(object_table),
            "raw_stats": str(engine_stats),
            "raw_stats_sha256": sha256_file(engine_stats),
            "command": command,
        },
        "elapsed_seconds": elapsed,
        "input_records_per_second": (
            counts.get("input_requests", 0) / elapsed if elapsed else None
        ),
        "not_claimed": [
            "cycle-accurate NVIDIA cache behavior",
            "production request issue timestamps",
            "physical HBF timing",
        ],
    }
    if read_fill_bytes or writeback_bytes:
        result["cache"].update(read_fill_bytes=read_fill_bytes,
                               writeback_bytes=writeback_bytes,
                               read_fill_object_boundary=read_fill_object_boundary)
        result["read_fill"] = engine.get("read_fill")
        result["writeback"] = engine.get("writeback")
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", required=True, help="compact input path or '-'")
    parser.add_argument("--output", required=True, help="compact output path or '-'")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--capacity-bytes", type=int, default=40 * 1024 * 1024)
    parser.add_argument("--line-bytes", type=int, default=128)
    parser.add_argument("--sector-bytes", type=int, default=32)
    parser.add_argument("--associativity", type=int, default=16)
    parser.add_argument(
        "--write-policy", choices=("write-through", "write-back"), default="write-back"
    )
    parser.add_argument("--no-write-allocate", action="store_true")
    parser.add_argument("--no-write-miss-fetch", action="store_true")
    parser.add_argument("--final-drain", action="store_true")
    parser.add_argument("--affine-prefix-input-records", type=int, default=0)
    parser.add_argument("--affine-period-input-records", type=int, default=0)
    parser.add_argument("--affine-period-object-deltas", type=Path)
    parser.add_argument("--affine-read-template-output", type=Path)
    parser.add_argument("--affine-writeback-fallback-output", type=Path)
    parser.add_argument("--stage-boundaries", type=Path)
    parser.add_argument("--timed-input-output", action="store_true")
    args = parser.parse_args()
    input_path = None if args.input == "-" else Path(args.input)
    output_path = None if args.output == "-" else Path(args.output)
    result = transform_fast(
        plan_path=args.plan,
        input_path=input_path,
        input_stream=sys.stdin.buffer if input_path is None else None,
        output_path=output_path,
        output_stream=sys.stdout.buffer if output_path is None else None,
        manifest_path=args.manifest,
        engine_path=args.engine,
        capacity_bytes=args.capacity_bytes,
        line_bytes=args.line_bytes,
        sector_bytes=args.sector_bytes,
        associativity=args.associativity,
        write_policy=args.write_policy,
        write_allocate=not args.no_write_allocate,
        write_miss_fetch=not args.no_write_miss_fetch,
        final_drain=args.final_drain,
        affine_prefix_input_records=args.affine_prefix_input_records,
        affine_period_input_records=args.affine_period_input_records,
        affine_period_object_deltas_path=args.affine_period_object_deltas,
        affine_read_template_output=args.affine_read_template_output,
        affine_writeback_fallback_output=args.affine_writeback_fallback_output,
        stage_boundaries_path=args.stage_boundaries,
        timed_input_output=args.timed_input_output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                **result["counts"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        file=sys.stderr if output_path is None else sys.stdout,
    )


if __name__ == "__main__":
    main()

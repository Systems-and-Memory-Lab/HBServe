"""Capture-driven address generation through one continuous reference cache.

The fine producer and cache consumer communicate through a bounded OS pipe;
no pre-cache flat trace is materialized. Capture/templates remain explicit
external inputs. No synthetic replacement is chosen when a binding is missing.
"""

from __future__ import annotations

import argparse
from contextvars import copy_context
import json
import os
from pathlib import Path
import sys
from threading import Thread
import time
from typing import Any, Sequence

from .common import load_json, new_output, require, save, sha256_file
from .artifacts import artifact_context, artifact_receipt

CACHE_FIELDS = {
    "capacity_bytes", "line_bytes", "sector_bytes", "associativity",
    "write_policy", "write_allocate", "write_miss_fetch", "final_drain",
}
CACHE_OPTIONAL_FIELDS = {"read_fill_bytes", "writeback_bytes", "read_fill_object_boundary"}


def generate(*, plan: Path, cache_config: dict[str, Any], output: Path,
             chunk_records: int = 65536, backend: str = "python",
             to_stdout: bool = False, cache_engine: Path | None = None,
             artifact_map: Path | None = None) -> dict[str, Any]:
    with artifact_context(artifact_map):
        return _generate(plan=plan, cache_config=cache_config, output=output,
                         chunk_records=chunk_records, backend=backend,
                         to_stdout=to_stdout, cache_engine=cache_engine)


def _generate(*, plan: Path, cache_config: dict[str, Any], output: Path,
              chunk_records: int, backend: str, to_stdout: bool,
              cache_engine: Path | None) -> dict[str, Any]:
    from ._reference.cache_compact_trace import transform_compact
    from ._reference.stream_lazy_trace_plan import stream_plan

    require(CACHE_FIELDS <= set(cache_config) <= CACHE_FIELDS | CACHE_OPTIONAL_FIELDS,
            f"cache config requires {sorted(CACHE_FIELDS)}; optional {sorted(CACHE_OPTIONAL_FIELDS)}")
    for key in ("capacity_bytes", "line_bytes", "sector_bytes", "associativity"):
        require(type(cache_config[key]) is int and cache_config[key] > 0,
                f"{key} must be a positive integer")
    for key in ("write_allocate", "write_miss_fetch", "final_drain"):
        require(type(cache_config[key]) is bool, f"{key} must be a boolean")
    require(cache_config["write_policy"] in {"write-back", "write-through"},
            "unsupported write policy")
    require(cache_config["line_bytes"] % cache_config["sector_bytes"] == 0,
            "line bytes must be a multiple of sector bytes")
    require(cache_config["capacity_bytes"] % (
        cache_config["line_bytes"] * cache_config["associativity"]) == 0,
        "cache capacity must contain a whole number of sets")
    for key in ("read_fill_bytes", "writeback_bytes"):
        require(type(cache_config.get(key, 0)) is int and cache_config.get(key, 0) in {0, 64},
                f"{key} must be 0 (legacy sectors) or 64")
    require(cache_config.get("read_fill_object_boundary", "reject") == "reject",
            "public reference route requires rejecting fills outside owned objects")
    if cache_config.get("read_fill_bytes", 0) or cache_config.get("writeback_bytes", 0):
        require(cache_engine is not None, "64B fill/writeback requires --cache-engine")
        require(cache_config.get("read_fill_bytes") == 64 and cache_config["sector_bytes"] == 32,
                "64B mode requires 32B sectors and 64B read fills")
        require(cache_config["line_bytes"] % 64 == 0, "64B fills require compatible cache lines")
    if cache_config.get("writeback_bytes", 0):
        require(cache_config["write_policy"] == "write-back" and cache_config["write_allocate"],
                "64B writeback requires write-back with write allocation")
    require(type(chunk_records) is int and chunk_records > 0, "chunk_records must be positive")
    require(backend in {"python", "numpy", "auto"}, "unsupported producer backend")
    if cache_engine is not None:
        require(cache_engine.is_file() and os.access(cache_engine, os.X_OK),
                f"cache engine is missing or not executable: {cache_engine}")
    plan = plan.resolve()
    require(plan.is_file(), f"missing source plan: {plan}")
    output = new_output(output)
    save(output / "cache-config.json", cache_config)
    started = time.perf_counter()
    read_fd, write_fd = os.pipe()
    producer_result: dict[str, Any] = {}
    producer_errors: list[BaseException] = []

    def produce() -> None:
        try:
            with os.fdopen(write_fd, "wb") as sink:
                producer_result.update(stream_plan(
                    plan_path=plan, output_path=None, output_stream=sink,
                    output_manifest_path=output / "issued.manifest.json",
                    chunk_records=chunk_records, backend=backend,
                ))
        except BaseException as error:
            producer_errors.append(error)

    context = copy_context()
    worker = Thread(target=context.run, args=(produce,), name="hbserve-reference-producer", daemon=True)
    worker.start()
    try:
        with os.fdopen(read_fd, "rb") as source:
            arguments = dict(
                plan_path=plan, input_path=None, input_stream=source,
                output_path=None if to_stdout else output / "post-cache.partial.bin",
                output_stream=sys.stdout.buffer if to_stdout else None,
                manifest_path=output / "cache-stage.json",
                **{key: value for key, value in cache_config.items() if key in CACHE_FIELDS},
            )
            if cache_engine is None:
                cache_result = transform_compact(cache_mode="reference-lru", **arguments)
            else:
                from ._reference.cache_compact_trace_fast import transform_fast
                cache_result = transform_fast(
                    engine_path=cache_engine, **arguments,
                    **{key: value for key, value in cache_config.items() if key in CACHE_OPTIONAL_FIELDS},
                )
    finally:
        # Closing the reader also releases a producer blocked after a consumer
        # error. A producer failure must never become a successful partial trace.
        worker.join()
    if producer_errors:
        raise RuntimeError(f"fine generator failed; output is incomplete: {producer_errors[0]}") from producer_errors[0]
    require(cache_result["input_sha256"] == producer_result["output_sha256"],
            "producer/cache stream digests differ")
    if not to_stdout:
        (output / "post-cache.partial.bin").rename(output / "post-cache.bin")
        cache_result["output"] = str(output / "post-cache.bin")
    # Publish a consumable manifest only after the entire producer succeeds.
    # Stdout consumers must likewise wait for this manifest and successful exit.
    save(output / "post-cache.manifest.json", cache_result)
    result = {
        "schema": {"name": "hbserve.trace_generation", "version": 1},
        "generator": "reference",
        "status": "GENERATED_NAMED_REFERENCE_POST_CACHE_NOT_NEW_FIDELITY_VALIDATION",
        "plan_sha256": sha256_file(plan),
        "producer": producer_result,
        "cache": cache_result,
        "precache_flat_file_written": False,
        "continuous_cache_across_plan_segments": True,
        "generation_seconds": time.perf_counter() - started,
        "source_resolution": artifact_receipt(),
        "not_claimed": ["measured hardware post-L2 addresses", "exact GPU issue timing",
                        "accuracy beyond source-plan coverage", "arbitrary-model support"],
    }
    save(output / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hbserve trace reference", description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="existing capture-bound lazy generation plan")
    parser.add_argument("--artifact-map", type=Path, help="optional exact-byte source relocation map from catalog export")
    parser.add_argument("--cache-config", type=Path, required=True, help="explicit reference-cache configuration JSON")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-records", type=int, default=65536)
    parser.add_argument("--backend", choices=("python", "numpy", "auto"), default="python")
    parser.add_argument("--cache-engine", type=Path,
                        help="optional existing C++ GPU reference-cache engine; never an HBFSim executor")
    parser.add_argument("--stdout", action="store_true", help="pipe post-cache binary to a consumer; sidecars remain in output-root")
    args = parser.parse_args(argv)
    result = generate(plan=args.plan, cache_config=load_json(args.cache_config),
                      output=args.output_root, chunk_records=args.chunk_records,
                      backend=args.backend, to_stdout=args.stdout, cache_engine=args.cache_engine,
                      artifact_map=args.artifact_map)
    print(json.dumps({"generator": "reference", "status": result["status"],
                      "counts": result["cache"]["counts"],
                      "generation_seconds": result["generation_seconds"]}, sort_keys=True),
          file=sys.stderr if args.stdout else sys.stdout)
    return 0

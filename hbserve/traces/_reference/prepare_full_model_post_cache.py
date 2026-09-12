#!/usr/bin/env python3
"""Materialize one digest-bound post-cache stream for matched replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from hbserve.traces._reference.cache_compact_trace import SCHEMA as CACHE_SCHEMA
from hbserve.traces._reference.compact_request_template import RECORD_BYTES, load_json, require, sha256_file
from hbserve.traces._reference.stream_lazy_trace_plan import SCHEMA as PRODUCER_SCHEMA


SCHEMA = {"name": "hbfsim.prepared_full_model_post_cache", "version": 1}


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    require(resolved.is_file(), f"artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def prepare(
    *,
    plan_path: Path,
    output_dir: Path,
    runtime_parameters: tuple[str, ...] = (),
    cache_mode: str = "reference-lru",
    cache_capacity_bytes: int = 40 * 1024 * 1024,
    cache_line_bytes: int = 128,
    cache_sector_bytes: int = 32,
    cache_associativity: int = 16,
    cache_write_policy: str = "write-back",
    cache_write_allocate: bool = True,
    cache_write_miss_fetch: bool = True,
    cache_final_drain: bool = False,
    stream_backend: str = "auto",
    stream_chunk_records: int = 1_000_000,
) -> dict[str, Any]:
    require(cache_mode in {"bypass", "reference-lru"}, "unsupported cache mode")
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_root = Path(__file__).resolve().parent
    paths = {
        "producer_manifest": output_dir / "producer.manifest.json",
        "producer_stderr": output_dir / "producer.stderr",
        "cache_manifest": output_dir / "cache.manifest.json",
        "cache_stdout": output_dir / "cache.stdout",
        "cache_stderr": output_dir / "cache.stderr",
        "compact": output_dir / "post_cache.compact.bin",
        "result": output_dir / "prepared_post_cache.result.json",
    }
    require(
        not any(path.exists() for path in paths.values()),
        "refusing to overwrite a prepared post-cache artifact",
    )

    producer_command = [
        sys.executable,
        str(script_root / "stream_lazy_trace_plan.py"),
        "--plan",
        str(plan_path),
        "--output",
        "-",
        "--output-manifest",
        str(paths["producer_manifest"]),
        "--backend",
        stream_backend,
        "--chunk-records",
        str(stream_chunk_records),
    ]
    for parameter in runtime_parameters:
        producer_command.extend(("--runtime-parameter", parameter))

    cache_command = [
        sys.executable,
        str(script_root / "cache_compact_trace.py"),
        "--plan",
        str(plan_path),
        "--input",
        "-",
        "--output",
        str(paths["compact"]),
        "--manifest",
        str(paths["cache_manifest"]),
        "--cache-mode",
        cache_mode,
        "--capacity-bytes",
        str(cache_capacity_bytes),
        "--line-bytes",
        str(cache_line_bytes),
        "--sector-bytes",
        str(cache_sector_bytes),
        "--associativity",
        str(cache_associativity),
        "--write-policy",
        cache_write_policy,
    ]
    if not cache_write_allocate:
        cache_command.append("--no-write-allocate")
    if not cache_write_miss_fetch:
        cache_command.append("--no-write-miss-fetch")
    if cache_final_drain:
        cache_command.append("--final-drain")

    with (
        paths["producer_stderr"].open("wb") as producer_stderr,
        paths["cache_stdout"].open("wb") as cache_stdout,
        paths["cache_stderr"].open("wb") as cache_stderr,
    ):
        producer = subprocess.Popen(
            producer_command,
            stdout=subprocess.PIPE,
            stderr=producer_stderr,
            bufsize=0,
        )
        assert producer.stdout is not None
        cache = subprocess.Popen(
            cache_command,
            stdin=producer.stdout,
            stdout=cache_stdout,
            stderr=cache_stderr,
            bufsize=0,
        )
        producer.stdout.close()
        try:
            cache_code = cache.wait()
            producer_code = producer.wait()
        except BaseException:
            if cache.poll() is None:
                cache.kill()
            if producer.poll() is None:
                producer.kill()
            cache.wait()
            producer.wait()
            raise
    require(producer_code == 0, f"lazy producer failed with exit {producer_code}")
    require(cache_code == 0, f"cache transform failed with exit {cache_code}")

    producer_result = load_json(paths["producer_manifest"])
    cache_result = load_json(paths["cache_manifest"])
    require(producer_result.get("schema") == PRODUCER_SCHEMA, "producer schema changed")
    require(cache_result.get("schema") == CACHE_SCHEMA, "cache schema changed")
    require(producer_result.get("status") == "PASS", "lazy producer did not pass")
    require(cache_result.get("status") == "PASS", "cache transform did not pass")
    require(
        producer_result.get("plan") == cache_result.get("plan") == str(plan_path),
        "producer/cache plan paths disagree",
    )
    plan_sha256 = sha256_file(plan_path)
    require(
        producer_result.get("plan_sha256") == plan_sha256,
        "producer plan digest disagrees",
    )
    require(
        producer_result.get("output_sha256") == cache_result.get("input_sha256"),
        "producer/cache digest disagrees",
    )
    compact_sha256 = sha256_file(paths["compact"])
    require(
        cache_result.get("output_sha256") == compact_sha256,
        "cache output digest disagrees with the materialized stream",
    )
    producer_counts = producer_result.get("totals")
    cache_counts = cache_result.get("counts")
    require(isinstance(producer_counts, dict), "producer totals are malformed")
    require(isinstance(cache_counts, dict), "cache counts are malformed")
    require(
        int(producer_counts["requests"]) == int(cache_counts["input_requests"])
        and int(producer_counts["bytes"]) == int(cache_counts["input_bytes"]),
        "producer/cache count or byte conservation failed",
    )
    require(
        paths["compact"].stat().st_size
        == int(cache_counts["output_requests"]) * RECORD_BYTES,
        "materialized compact byte length disagrees with its request count",
    )

    cache_contract = {
        "mode": cache_mode,
        "capacity_bytes": cache_capacity_bytes,
        "line_bytes": cache_line_bytes,
        "sector_bytes": cache_sector_bytes,
        "associativity": cache_associativity,
        "write_policy": cache_write_policy,
        "write_allocate": cache_write_allocate,
        "write_miss_fetch": cache_write_miss_fetch,
        "final_drain": cache_final_drain,
    }
    result = {
        "schema": SCHEMA,
        "status": "PASS_PREPARED_FULL_MODEL_POST_CACHE",
        "classification": (
            "one immutable materialized post-cache compact stream shared by "
            "matched closed-loop service-point replays"
        ),
        "inputs": {
            "plan": str(plan_path),
            "plan_sha256": plan_sha256,
            "runtime_parameters": list(runtime_parameters),
        },
        "cache_contract": cache_contract,
        "conservation": {
            "producer_to_cache_sha256": producer_result["output_sha256"],
            "cache_to_replay_sha256": compact_sha256,
            "producer_requests": int(producer_counts["requests"]),
            "producer_bytes": int(producer_counts["bytes"]),
            "post_cache_requests": int(cache_counts["output_requests"]),
            "post_cache_bytes": int(cache_counts["output_bytes"]),
            "post_cache_compact_file_bytes": paths["compact"].stat().st_size,
        },
        "commands": {"producer": producer_command, "cache": cache_command},
        "artifacts": {
            name: artifact(paths[name])
            for name in (
                "producer_manifest",
                "producer_stderr",
                "cache_manifest",
                "cache_stdout",
                "cache_stderr",
                "compact",
            )
        },
        "not_claimed": [
            "a different trace or cache state for each service point",
            "cycle-accurate NVIDIA cache behavior",
            "physical HBF timing",
        ],
    }
    paths["result"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-parameter", action="append", default=[])
    parser.add_argument(
        "--cache-mode", choices=("bypass", "reference-lru"), default="reference-lru"
    )
    parser.add_argument("--cache-capacity-bytes", type=int, default=40 * 1024 * 1024)
    parser.add_argument("--cache-line-bytes", type=int, default=128)
    parser.add_argument("--cache-sector-bytes", type=int, default=32)
    parser.add_argument("--cache-associativity", type=int, default=16)
    parser.add_argument(
        "--cache-write-policy",
        choices=("write-through", "write-back"),
        default="write-back",
    )
    parser.add_argument("--no-cache-write-allocate", action="store_true")
    parser.add_argument("--no-cache-write-miss-fetch", action="store_true")
    parser.add_argument("--cache-final-drain", action="store_true")
    parser.add_argument(
        "--stream-backend", choices=("auto", "python", "numpy"), default="auto"
    )
    parser.add_argument("--stream-chunk-records", type=int, default=1_000_000)
    args = parser.parse_args()
    result = prepare(
        plan_path=args.plan,
        output_dir=args.output_dir,
        runtime_parameters=tuple(args.runtime_parameter),
        cache_mode=args.cache_mode,
        cache_capacity_bytes=args.cache_capacity_bytes,
        cache_line_bytes=args.cache_line_bytes,
        cache_sector_bytes=args.cache_sector_bytes,
        cache_associativity=args.cache_associativity,
        cache_write_policy=args.cache_write_policy,
        cache_write_allocate=not args.no_cache_write_allocate,
        cache_write_miss_fetch=not args.no_cache_write_miss_fetch,
        cache_final_drain=args.cache_final_drain,
        stream_backend=args.stream_backend,
        stream_chunk_records=args.stream_chunk_records,
    )
    print(
        json.dumps(
            {"status": result["status"], **result["conservation"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

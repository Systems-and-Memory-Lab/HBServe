#!/usr/bin/env python3
"""Generate a sector-level embedding gather without a captured request anchor.

The supported kernel is deliberately narrow: one output element is copied
from ``embedding[token_ids[row], column]`` by a grid-stride loop.  The launch
must have a one-dimensional grid, a whole number of warps per CTA, and a row
width that is a whole number of warps.  Under those conditions every warp
iteration has the following exact, deterministic request program after the
32-byte reference coalescer used by this trace pipeline::

    token-id sector read
    two embedding-row sector reads
    two output sector writes

The module emits the existing 12-byte compact request records.  It models no
GPU scheduling, issue timestamps, cache behavior, or physical HBF traffic.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import struct
from typing import Any, Iterator, Sequence


POLICY_KIND = "semantic_embedding_grid_stride"
VALIDATION_STATUS = "PASS_EXACT_FULL_GRID_ORACLE"
READ = 0
WRITE = 1


class SemanticEmbeddingError(ValueError):
    """Raised when an embedding policy cannot be applied without guessing."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise SemanticEmbeddingError(message)


def integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label}: integer required",
    )
    result = int(value)
    if minimum is not None:
        require(result >= minimum, f"{label}: must be at least {minimum}")
    return result


def token_ids_sha256_le_u32(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        require(0 <= int(value) <= 0xFFFFFFFF, "token ID does not fit u32")
        digest.update(struct.pack("<I", int(value)))
    return digest.hexdigest()


def validate_policy(
    policy: dict[str, Any], *, object_extents: Sequence[int],
    required_validation_status: str = VALIDATION_STATUS,
) -> dict[str, Any]:
    """Validate and normalize one semantic embedding policy."""

    require(policy.get("kind") == POLICY_KIND, "unsupported embedding policy kind")
    validation = policy.get("validation")
    require(isinstance(validation, dict), "embedding policy has no validation receipt")
    require(
        validation.get("status") == required_validation_status,
        "embedding policy has an unexpected validation status",
    )

    context_tokens = integer(policy.get("context_tokens"), "context tokens", minimum=1)
    hidden_elements = integer(policy.get("hidden_elements"), "hidden elements", minimum=1)
    element_bytes = integer(policy.get("element_bytes"), "element bytes", minimum=1)
    token_id_bytes = integer(policy.get("token_id_bytes"), "token-ID bytes", minimum=1)
    sector_bytes = integer(policy.get("sector_bytes"), "sector bytes", minimum=1)
    grid_x = integer(policy.get("grid_x"), "grid x", minimum=1)
    block_threads = integer(policy.get("block_threads"), "block threads", minimum=1)
    warp_threads = integer(policy.get("warp_threads"), "warp threads", minimum=1)
    kernel_ordinal = integer(policy.get("kernel_ordinal"), "kernel ordinal", minimum=0)

    require(sector_bytes == 32, "compact embedding generator currently requires 32-byte sectors")
    require(
        warp_threads * element_bytes == 2 * sector_bytes,
        "one embedding warp must cover exactly two sectors",
    )
    require(block_threads % warp_threads == 0, "CTA size is not a whole number of warps")
    require(hidden_elements % warp_threads == 0, "embedding row is not warp aligned")

    raw_tokens = policy.get("token_ids")
    require(isinstance(raw_tokens, list), "embedding policy has no token IDs")
    token_ids = [integer(value, "token ID", minimum=0) for value in raw_tokens]
    require(len(token_ids) == context_tokens, "token-ID count differs from context length")
    observed_digest = token_ids_sha256_le_u32(token_ids)
    require(
        policy.get("token_ids_sha256_le_u32") == observed_digest,
        "embedding token-ID digest differs",
    )

    indices = {
        role: integer(policy.get(f"{role}_object_index"), f"{role} object index", minimum=0)
        for role in ("input", "weight", "output")
    }
    require(len(set(indices.values())) == 3, "embedding object indices must be distinct")
    require(
        all(index < len(object_extents) for index in indices.values()),
        "embedding policy references an unknown object",
    )

    input_required = context_tokens * token_id_bytes
    output_required = context_tokens * hidden_elements * element_bytes
    row_bytes = hidden_elements * element_bytes
    vocab_rows = int(object_extents[indices["weight"]]) // row_bytes
    require(vocab_rows > 0, "embedding weight object has no complete row")
    require(
        int(object_extents[indices["weight"]]) % row_bytes == 0,
        "embedding weight extent is not a whole number of rows",
    )
    require(max(token_ids) < vocab_rows, "token ID escapes the embedding weight")
    require(
        input_required <= int(object_extents[indices["input"]]),
        "token-ID input escapes its object",
    )
    require(
        output_required <= int(object_extents[indices["output"]]),
        "embedding output escapes its object",
    )
    total_elements = context_tokens * hidden_elements
    require(total_elements % warp_threads == 0, "embedding tail is not a complete warp")

    return {
        "kernel_ordinal": kernel_ordinal,
        "context_tokens": context_tokens,
        "hidden_elements": hidden_elements,
        "element_bytes": element_bytes,
        "token_id_bytes": token_id_bytes,
        "sector_bytes": sector_bytes,
        "grid_x": grid_x,
        "block_threads": block_threads,
        "warp_threads": warp_threads,
        "token_ids": token_ids,
        "token_ids_sha256_le_u32": observed_digest,
        "input_object_index": indices["input"],
        "weight_object_index": indices["weight"],
        "output_object_index": indices["output"],
        "row_bytes": row_bytes,
        "total_elements": total_elements,
    }


def warp_iteration_count(program: dict[str, Any], cta_x: int, warp: int) -> int:
    first = (
        cta_x * int(program["block_threads"])
        + warp * int(program["warp_threads"])
    )
    total = int(program["total_elements"])
    if first >= total:
        return 0
    stride = int(program["grid_x"]) * int(program["block_threads"])
    return 1 + (total - 1 - first) // stride


def iter_cta_records(
    program: dict[str, Any], *, cta_x: int
) -> Iterator[tuple[int, int, int, int, int, int]]:
    """Yield one CTA in the captured warp-major, loop-iteration order."""

    cta_x = integer(cta_x, "CTA x", minimum=0)
    require(cta_x < int(program["grid_x"]), "embedding CTA x escapes the launch grid")
    block_threads = int(program["block_threads"])
    warp_threads = int(program["warp_threads"])
    grid_stride = int(program["grid_x"]) * block_threads
    hidden = int(program["hidden_elements"])
    element_bytes = int(program["element_bytes"])
    token_id_bytes = int(program["token_id_bytes"])
    sector = int(program["sector_bytes"])
    kernel = int(program["kernel_ordinal"])
    tokens = program["token_ids"]
    total = int(program["total_elements"])

    for warp in range(block_threads // warp_threads):
        element = cta_x * block_threads + warp * warp_threads
        while element < total:
            token_index, hidden_index = divmod(element, hidden)
            require(
                hidden_index % warp_threads == 0,
                "embedding warp begins at an unaligned row position",
            )
            token_sector = (token_index * token_id_bytes // sector) * sector
            weight_offset = int(tokens[token_index]) * int(program["row_bytes"])
            weight_offset += hidden_index * element_bytes
            output_offset = element * element_bytes
            yield (
                int(program["input_object_index"]), token_sector,
                kernel, sector, READ, 0,
            )
            yield (
                int(program["weight_object_index"]), weight_offset,
                kernel, sector, READ, 0,
            )
            yield (
                int(program["weight_object_index"]), weight_offset + sector,
                kernel, sector, READ, 0,
            )
            yield (
                int(program["output_object_index"]), output_offset,
                kernel, sector, WRITE, 0,
            )
            yield (
                int(program["output_object_index"]), output_offset + sector,
                kernel, sector, WRITE, 0,
            )
            element += grid_stride


def range_census(
    program: dict[str, Any], *, begin: int, end: int
) -> tuple[Counter[str], dict[int, Counter[str]]]:
    """Return an analytic census for a half-open CTA-x range."""

    begin = integer(begin, "CTA begin", minimum=0)
    end = integer(end, "CTA end", minimum=1)
    require(begin < end <= int(program["grid_x"]), "embedding CTA range escapes the grid")
    warps = int(program["block_threads"]) // int(program["warp_threads"])
    iterations = sum(
        warp_iteration_count(program, cta_x, warp)
        for cta_x in range(begin, end)
        for warp in range(warps)
    )
    sector = int(program["sector_bytes"])
    totals: Counter[str] = Counter({
        "target_ctas": end - begin,
        "active_grid_ctas": end - begin,
        "requests": iterations * 5,
        "bytes": iterations * 5 * sector,
    })
    by_object: dict[int, Counter[str]] = {}
    roles = (
        (int(program["input_object_index"]), iterations, READ),
        (int(program["weight_object_index"]), iterations * 2, READ),
        (int(program["output_object_index"]), iterations * 2, WRITE),
    )
    for object_index, requests, operation in roles:
        counter = Counter({"requests": requests, "bytes": requests * sector})
        prefix = "r" if operation == READ else "w"
        counter[f"{prefix}_requests"] = requests
        counter[f"{prefix}_bytes"] = requests * sector
        by_object[object_index] = counter
    return totals, by_object

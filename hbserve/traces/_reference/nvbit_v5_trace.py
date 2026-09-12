#!/usr/bin/env python3
"""Strict streaming decoder for Accel-Sim/NVBit tracer formats 5 and 7.

The decoder preserves the execution-location prefix and the active-lane to
address relation.  It deliberately does not infer timestamps, cache traffic,
physical addresses, or device requests.

Version 7 is the opt-in memory-reference metadata extension used by this
project.  Version 5 remains accepted without manufacturing v7 operand labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import lzma
from pathlib import Path
import re
from typing import Iterator, TextIO


class NVBitV5TraceError(ValueError):
    """Raised when a raw trace cannot be decoded without guessing."""


MEMORY_SPACE_NAMES = (
    "NONE",
    "LOCAL",
    "GENERIC",
    "GLOBAL",
    "SHARED",
    "CONSTANT",
    "GLOBAL_TO_SHARED",
    "SURFACE",
    "TEXTURE",
    "DISTRIBUTED_SHARED",
    "TENSOR_MEM",
    "TENSOR_CORE_MEM",
)


@dataclass(frozen=True)
class NVBitMemoryReferenceMetadata:
    format_version: int
    instruction_memory_space: int
    instruction_memory_space_name: str
    mref_memory_space: int
    mref_memory_space_name: str
    num_mref: int
    instrumentation_mref_index: int
    address_mref_index: int
    is_load: bool
    is_store: bool


@dataclass(frozen=True)
class NVBitV5Instruction:
    cta: tuple[int, int, int]
    warp_in_cta: int
    cluster: tuple[int, int, int]
    cluster_cta: tuple[int, int, int]
    cluster_rank: int
    pc: int
    active_mask: int
    destination_registers: tuple[str, ...]
    opcode: str
    source_registers: tuple[str, ...]
    gmma_flag: str | None
    memory_width: int
    address_format: int | None
    active_lanes: tuple[int, ...]
    addresses: tuple[int, ...]
    immediates: tuple[str, ...]
    memory_reference_metadata: NVBitMemoryReferenceMetadata | None
    sm_id: int | None = None
    warp_in_sm: int | None = None


@dataclass(frozen=True)
class NVBitV5Record:
    line_number: int
    raw_record_index: int
    raw_line_sha256: str
    instruction: NVBitV5Instruction


def open_trace(path: Path) -> TextIO:
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="strict")
    return path.open("rt", encoding="utf-8", errors="strict")


def _integer(token: str, *, base: int = 0, label: str) -> int:
    try:
        return int(token, base)
    except ValueError as error:
        raise NVBitV5TraceError(f"invalid {label}: {token!r}") from error


def _take(tokens: list[str], cursor: int, count: int, label: str) -> tuple[list[str], int]:
    if count < 0 or cursor + count > len(tokens):
        raise NVBitV5TraceError(f"truncated {label}")
    return tokens[cursor : cursor + count], cursor + count


def _expand_addresses(
    tokens: list[str], cursor: int, active_lanes: tuple[int, ...]
) -> tuple[int, tuple[int, ...], int]:
    if cursor >= len(tokens):
        raise NVBitV5TraceError("missing address compression mode")
    address_format = _integer(tokens[cursor], label="address compression mode")
    cursor += 1
    active_count = len(active_lanes)

    if address_format == 0:  # list_all
        payload, cursor = _take(tokens, cursor, active_count, "list_all addresses")
        addresses = tuple(_integer(value, label="lane address") for value in payload)
    elif address_format == 1:  # base_stride
        payload, cursor = _take(tokens, cursor, 2, "base_stride payload")
        base = _integer(payload[0], label="base address")
        stride = _integer(payload[1], label="address stride")
        addresses = tuple(base + ordinal * stride for ordinal in range(active_count))
    elif address_format == 2:  # base_delta
        payload_count = 1 if active_count == 0 else active_count
        payload, cursor = _take(tokens, cursor, payload_count, "base_delta payload")
        if active_count == 0:
            addresses = ()
        else:
            values = [_integer(payload[0], label="base address")]
            for token in payload[1:]:
                values.append(values[-1] + _integer(token, label="address delta"))
            addresses = tuple(values)
    else:
        raise NVBitV5TraceError(
            f"unsupported address compression mode {address_format}"
        )

    if len(addresses) != active_count:
        raise NVBitV5TraceError("active-mask/address cardinality mismatch")
    # Shared-memory records legitimately use a zero-based address.  The raw
    # decoder is address-space neutral; the importer separately requires a
    # positive process VA for global-memory events.
    if any(address < 0 for address in addresses):
        raise NVBitV5TraceError("active lane has a negative address")
    return address_format, addresses, cursor


def _memory_space(value: str, name: str, *, label: str) -> tuple[int, str]:
    enum_value = _integer(value, label=f"{label} enum")
    if enum_value < 0 or enum_value >= len(MEMORY_SPACE_NAMES):
        raise NVBitV5TraceError(f"{label} enum is outside the NVBit v1.8 range")
    expected_name = MEMORY_SPACE_NAMES[enum_value]
    if name != expected_name:
        raise NVBitV5TraceError(
            f"{label} enum/name mismatch: {enum_value} is {expected_name}, found {name}"
        )
    return enum_value, name


def _parse_memory_reference_metadata(
    tokens: list[str], cursor: int
) -> tuple[NVBitMemoryReferenceMetadata, int]:
    payload, cursor = _take(tokens, cursor, 10, "MEM_META_V1 payload")
    if payload[0] != "MEM_META_V1":
        raise NVBitV5TraceError("missing MEM_META_V1 marker")
    instruction_space, instruction_name = _memory_space(
        payload[1], payload[2], label="instruction memory space"
    )
    mref_space, mref_name = _memory_space(
        payload[3], payload[4], label="memory-reference space"
    )
    num_mref = _integer(payload[5], label="memory-reference count")
    loop_index = _integer(payload[6], label="instrumentation-loop mref index")
    address_index = _integer(payload[7], label="address mref index")
    is_load = _integer(payload[8], label="isLoad flag")
    is_store = _integer(payload[9], label="isStore flag")
    if is_load not in (0, 1) or is_store not in (0, 1):
        raise NVBitV5TraceError("isLoad/isStore flags must be 0 or 1")
    if num_mref < 0 or num_mref > 2:
        raise NVBitV5TraceError("memory-reference count must be in [0, 2]")
    if num_mref == 0:
        if loop_index != -1 or address_index != -1:
            raise NVBitV5TraceError("non-memory metadata must use mref indexes -1/-1")
    elif not (
        0 <= loop_index < num_mref and 0 <= address_index < num_mref
    ):
        raise NVBitV5TraceError("memory-reference index is outside num_mref")

    if instruction_name == "GLOBAL_TO_SHARED":
        if num_mref != 2 or address_index != 1 - loop_index:
            raise NVBitV5TraceError(
                "GLOBAL_TO_SHARED requires two reversed instrumentation/address indexes"
            )
        expected_mref_name = "SHARED" if address_index == 0 else "GLOBAL"
        if mref_name != expected_mref_name:
            raise NVBitV5TraceError(
                "GLOBAL_TO_SHARED mref index/space mismatch: "
                f"index {address_index} must be {expected_mref_name}"
            )
    elif num_mref > 0:
        if address_index != loop_index:
            raise NVBitV5TraceError(
                "single-space instruction changed the instrumentation mref index"
            )
        if mref_space != instruction_space:
            raise NVBitV5TraceError(
                "single-space instruction/mref memory spaces disagree"
            )

    return (
        NVBitMemoryReferenceMetadata(
            format_version=1,
            instruction_memory_space=instruction_space,
            instruction_memory_space_name=instruction_name,
            mref_memory_space=mref_space,
            mref_memory_space_name=mref_name,
            num_mref=num_mref,
            instrumentation_mref_index=loop_index,
            address_mref_index=address_index,
            is_load=bool(is_load),
            is_store=bool(is_store),
        ),
        cursor,
    )


def parse_instruction_record(
    line: str, *, tracer_version: int = 5, trace_core: bool = False
) -> NVBitV5Instruction | None:
    """Decode one raw record, returning ``None`` for headers/comments/blanks."""
    if tracer_version not in (5, 7):
        raise NVBitV5TraceError(
            f"unsupported Accel-Sim/NVBit tracer version {tracer_version}"
        )
    stripped = line.strip()
    if not stripped or stripped.startswith(("-", "#")):
        return None
    tokens = stripped.split()
    if len(tokens) < 18:
        raise NVBitV5TraceError("truncated instruction record")

    execution_location_fields = 13 if trace_core else 11
    prefix = [
        _integer(token, label="execution-location field")
        for token in tokens[:execution_location_fields]
    ]
    cursor = execution_location_fields
    sm_id = prefix[11] if trace_core else None
    warp_in_sm = prefix[12] if trace_core else None
    if trace_core and (sm_id is None or sm_id < 0 or warp_in_sm is None or warp_in_sm < 0):
        raise NVBitV5TraceError("negative SM or SM-local warp identifier")
    pc = _integer(tokens[cursor], base=16, label="virtual PC")
    cursor += 1
    active_mask = _integer(tokens[cursor], base=16, label="active mask")
    cursor += 1
    if active_mask < 0 or active_mask > 0xFFFFFFFF:
        raise NVBitV5TraceError("active mask does not fit 32 lanes")

    destination_count = _integer(tokens[cursor], label="destination count")
    cursor += 1
    destination_tokens, cursor = _take(
        tokens, cursor, destination_count, "destination registers"
    )
    if cursor >= len(tokens):
        raise NVBitV5TraceError("missing opcode")
    opcode = tokens[cursor]
    cursor += 1
    if cursor >= len(tokens):
        raise NVBitV5TraceError("missing source count")
    source_count = _integer(tokens[cursor], label="source count")
    cursor += 1
    source_tokens, cursor = _take(tokens, cursor, source_count, "source registers")

    gmma_flag: str | None = None
    if "GMMA" in opcode.upper():
        if cursor >= len(tokens):
            raise NVBitV5TraceError("missing GMMA flag")
        gmma_flag = tokens[cursor]
        cursor += 1
    if cursor >= len(tokens):
        raise NVBitV5TraceError("missing memory width")
    memory_width = _integer(tokens[cursor], label="memory width")
    cursor += 1
    if memory_width < 0:
        raise NVBitV5TraceError("negative memory width")

    active_lanes = tuple(
        lane for lane in range(32) if active_mask & (1 << lane)
    )
    address_format: int | None = None
    addresses: tuple[int, ...] = ()
    if memory_width > 0:
        address_format, addresses, cursor = _expand_addresses(
            tokens, cursor, active_lanes
        )
    # Tracer v5 emits two immediate fields.  Non-memory instructions may also
    # carry inert address placeholders before them, so only memory records are
    # checked exactly here.  V7 appends one fixed-size MEM_META_V1 block.
    memory_reference_metadata: NVBitMemoryReferenceMetadata | None = None
    if tracer_version == 5:
        immediates = tuple(tokens[cursor:])
        if memory_width > 0 and len(immediates) != 2:
            raise NVBitV5TraceError(
                f"expected two trailing immediates, found {len(immediates)}"
            )
    else:
        try:
            metadata_cursor = tokens.index("MEM_META_V1", cursor)
        except ValueError as error:
            raise NVBitV5TraceError("v7 record has no MEM_META_V1 block") from error
        pre_metadata = tokens[cursor:metadata_cursor]
        if memory_width > 0 and len(pre_metadata) != 2:
            raise NVBitV5TraceError(
                f"expected two v7 immediates, found {len(pre_metadata)}"
            )
        if len(pre_metadata) < 2:
            raise NVBitV5TraceError("v7 record has truncated immediates")
        # For non-memory instructions, ignore the legacy inert address
        # placeholder while preserving the actual two immediate fields.
        immediates = tuple(pre_metadata[-2:])
        memory_reference_metadata, metadata_end = (
            _parse_memory_reference_metadata(tokens, metadata_cursor)
        )
        if metadata_end != len(tokens):
            raise NVBitV5TraceError("unexpected tokens after MEM_META_V1")
        if memory_width > 0 and memory_reference_metadata.num_mref == 0:
            raise NVBitV5TraceError("memory record declares zero memory references")
        if memory_width == 0:
            if memory_reference_metadata.num_mref != 0:
                raise NVBitV5TraceError(
                    "zero-width record declares memory references"
                )
            if (
                memory_reference_metadata.mref_memory_space
                != memory_reference_metadata.instruction_memory_space
            ):
                raise NVBitV5TraceError(
                    "zero-width record has inconsistent instruction/mref spaces"
                )

    return NVBitV5Instruction(
        cta=tuple(prefix[0:3]),
        warp_in_cta=prefix[3],
        cluster=tuple(prefix[4:7]),
        cluster_cta=tuple(prefix[7:10]),
        cluster_rank=prefix[10],
        pc=pc,
        active_mask=active_mask,
        destination_registers=tuple(destination_tokens),
        opcode=opcode,
        source_registers=tuple(source_tokens),
        gmma_flag=gmma_flag,
        memory_width=memory_width,
        address_format=address_format,
        active_lanes=active_lanes,
        addresses=addresses,
        immediates=immediates,
        memory_reference_metadata=memory_reference_metadata,
        sm_id=sm_id,
        warp_in_sm=warp_in_sm,
    )


def iter_instruction_records(path: Path) -> Iterator[NVBitV5Record]:
    """Yield decoded records with a per-file raw instruction ordinal."""
    header = read_trace_header(path)
    tracer_version = int(header["accelsim_tracer_version"])
    trace_core = bool(header["trace_core"])
    raw_record_index = 0
    with open_trace(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                instruction = parse_instruction_record(
                    line, tracer_version=tracer_version, trace_core=trace_core
                )
            except NVBitV5TraceError as error:
                raise NVBitV5TraceError(
                    f"{path}:{line_number}: {error}: {line.strip()}"
                ) from error
            if instruction is None:
                continue
            yield NVBitV5Record(
                line_number=line_number,
                raw_record_index=raw_record_index,
                raw_line_sha256=hashlib.sha256(
                    line.rstrip("\r\n").encode("utf-8")
                ).hexdigest(),
                instruction=instruction,
            )
            raw_record_index += 1


def iter_instructions(path: Path) -> Iterator[NVBitV5Instruction]:
    """Yield decoded instructions without computing unused raw-line digests.

    Full-grid structural audits can contain tens of millions of text records.
    They still need the strict parser and contextual error reporting, but do
    not need a SHA-256 digest for every source line or a wrapper dataclass.
    """
    header = read_trace_header(path)
    tracer_version = int(header["accelsim_tracer_version"])
    trace_core = bool(header["trace_core"])
    with open_trace(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                instruction = parse_instruction_record(
                    line, tracer_version=tracer_version, trace_core=trace_core
                )
            except NVBitV5TraceError as error:
                raise NVBitV5TraceError(
                    f"{path}:{line_number}: {error}: {line.strip()}"
                ) from error
            if instruction is not None:
                yield instruction


def read_trace_header(path: Path) -> dict[str, object]:
    """Read a supported v5/v7 file header without scanning its body."""
    values: dict[str, str] = {}
    with open_trace(path) as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("-") and "=" in stripped:
                key, value = stripped[1:].split("=", 1)
                values[key.strip()] = value.strip()
                continue
            if stripped.startswith("#"):
                continue
            break
    required = ("kernel name", "kernel id", "cuda stream id", "accelsim tracer version")
    missing = [key for key in required if key not in values]
    if missing:
        raise NVBitV5TraceError(f"{path}: missing header fields {missing}")
    version = _integer(values["accelsim tracer version"], label="tracer version")
    if version not in (5, 7):
        raise NVBitV5TraceError(
            f"{path}: expected tracer version 5 or 7, found {version}"
        )
    metadata_version = (
        _integer(values["mref metadata version"], label="mref metadata version")
        if values.get("mref metadata version") is not None
        else None
    )
    if version == 7 and metadata_version != 1:
        raise NVBitV5TraceError(
            f"{path}: tracer version 7 requires mref metadata version 1"
        )
    if version == 5 and metadata_version is not None:
        raise NVBitV5TraceError(
            f"{path}: tracer version 5 cannot claim mref metadata"
        )
    trace_core_value = (
        _integer(values["enable core id"], label="core-id flag")
        if values.get("enable core id") is not None
        else 0
    )
    if trace_core_value not in (0, 1):
        raise NVBitV5TraceError("core-id flag must be 0 or 1")
    cta_begin_raw = values.get("trace cta x begin")
    cta_end_raw = values.get("trace cta x end exclusive")
    if (cta_begin_raw is None) != (cta_end_raw is None):
        raise NVBitV5TraceError(
            f"{path}: CTA-x filter header must declare both range endpoints"
        )
    cta_range: dict[str, object] = {}
    if cta_begin_raw is not None:
        assert cta_end_raw is not None
        cta_begin = _integer(cta_begin_raw, label="CTA-x trace begin")
        cta_end = _integer(cta_end_raw, label="CTA-x trace end exclusive")
        if cta_begin < 0 or (cta_end != -1 and cta_end <= cta_begin):
            raise NVBitV5TraceError(
                f"{path}: invalid CTA-x trace range [{cta_begin},{cta_end})"
            )
        cta_range = {
            "trace_cta_x_begin": cta_begin,
            "trace_cta_x_end_exclusive": cta_end,
            "trace_cta_x_filter_declared": True,
        }
    return {
        "kernel_name": values["kernel name"],
        "kernel_id": _integer(values["kernel id"], label="kernel id"),
        "cuda_stream_id": _integer(values["cuda stream id"], label="CUDA stream id"),
        "accelsim_tracer_version": version,
        "trace_core": bool(trace_core_value),
        **cta_range,
        **(
            {"mref_metadata_version": metadata_version}
            if metadata_version is not None
            else {}
        ),
        "nvbit_version": values.get("nvbit version"),
        "grid_dim": values.get("grid dim"),
        "block_dim": values.get("block dim"),
        "shared_memory_bytes": (
            _integer(values["shmem"], label="shared-memory bytes")
            if values.get("shmem") is not None
            else None
        ),
        "shared_memory_base_address": (
            _integer(values["shmem base_addr"], label="shared-memory base address")
            if values.get("shmem base_addr") is not None
            else None
        ),
    }


_KERNEL_FILE_RE = re.compile(r"^kernel-(?P<kernel_id>\d+)-.+\.trace(?:\.xz)?$")
_MEMCPY_HTOD_RE = re.compile(
    r"^MemcpyHtoD,0x(?P<destination>[0-9a-fA-F]{16}),(?P<bytes>\d+)$"
)


def kernel_id_from_filename(path: Path) -> int:
    match = _KERNEL_FILE_RE.match(path.name)
    if match is None:
        raise NVBitV5TraceError(f"malformed kernel trace filename: {path.name}")
    return int(match.group("kernel_id"))


def discover_trace_files(
    trace_root: Path, kernel_list: Path | None = None
) -> tuple[list[Path], dict[str, object]]:
    """Return launch order, preferring the tracer's formal kernel list."""
    trace_root = trace_root.resolve()
    available = sorted(trace_root.glob("kernel-*.trace*"))
    if not available:
        raise NVBitV5TraceError(f"no kernel trace files under {trace_root}")
    if kernel_list is not None:
        list_paths = [kernel_list.resolve()]
    else:
        list_paths = sorted(trace_root.glob("kernelslist_ctx_*"))

    if not list_paths:
        ordered = sorted(available, key=lambda path: (kernel_id_from_filename(path), path.name))
        return ordered, {
            "source": "numeric kernel id fallback; no kernelslist file found",
            "kernel_lists": [],
        }

    names: list[str] = []
    non_kernel_entries: list[dict[str, object]] = []
    list_descriptors = []
    for list_path in list_paths:
        if not list_path.is_file():
            raise NVBitV5TraceError(f"kernel list does not exist: {list_path}")
        digest = hashlib.sha256(list_path.read_bytes()).hexdigest()
        for line_number, raw in enumerate(
            list_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            name = raw.strip()
            if not name:
                continue
            memcpy = _MEMCPY_HTOD_RE.fullmatch(name)
            if memcpy is not None:
                non_kernel_entries.append(
                    {
                        "kernel_list_file": list_path.name,
                        "line_number": line_number,
                        "kind": "MemcpyHtoD",
                        "destination_gpu_va": int(memcpy.group("destination"), 16),
                        "bytes": int(memcpy.group("bytes")),
                        "raw_entry": name,
                    }
                )
                continue
            if Path(name).name != name:
                raise NVBitV5TraceError(f"unsafe kernel-list entry: {name!r}")
            if _KERNEL_FILE_RE.fullmatch(name) is None:
                raise NVBitV5TraceError(
                    f"unsupported kernelslist entry at {list_path}:{line_number}: "
                    f"{name!r}"
                )
            names.append(name)
        list_descriptors.append({"file": list_path.name, "sha256": digest})
    if len(names) != len(set(names)):
        raise NVBitV5TraceError("kernel list repeats a trace file")
    available_by_name = {path.name: path for path in available}
    missing = [name for name in names if name not in available_by_name]
    extra = sorted(set(available_by_name) - set(names))
    if missing or extra:
        raise NVBitV5TraceError(
            f"kernel list/file mismatch: missing={missing}, unlisted={extra}"
        )
    return [available_by_name[name] for name in names], {
        "source": (
            "observed Accel-Sim kernelslist kernel-file order; recognized "
            "non-kernel entries preserved separately"
        ),
        "kernel_lists": list_descriptors,
        "non_kernel_entries": non_kernel_entries,
    }

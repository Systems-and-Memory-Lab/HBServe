#!/usr/bin/env python3
"""Check release inputs and byte identity without extracting untrusted archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def check(sdist: Path, wheel: Path) -> dict:
    expected = {"README.md", "LICENSE", "NOTICE.md", "MANIFEST.in", "pyproject.toml",
                "native/reference_cache.cpp", "tests/smoke_trace_install.py",
                "tests/check_trace_distribution.py"}
    for folder, suffixes in (("configs", {".cfg", ".json"}), ("models", {".json"}),
                             ("examples", {".py", ".json"}), ("docs", {".md"}),
                             ("reference_templates", {".json", ".gz", ".md"})):
        expected.update(str(path.relative_to(ROOT)) for path in (ROOT / folder).rglob("*")
                        if path.is_file() and path.suffix in suffixes)
    runtime = {str(path.relative_to(ROOT)) for name in ("hbserve", "hbfsim_client")
               for path in (ROOT / name).rglob("*.py")}
    data = {"hbserve/traces/SOURCES.json", "hbserve/traces/LICENSE.HBFSim"}
    data.update(str(path.relative_to(ROOT)) for path in (ROOT / "hbserve/traces/data").rglob("*.json"))
    with tarfile.open(sdist, "r:gz") as source:
        members = {}
        roots = set()
        for member in source.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe source member: {member.name}")
            roots.add(name.parts[0])
            if member.isfile():
                key = str(PurePosixPath(*name.parts[1:]))
                if key in members:
                    raise ValueError(f"duplicate source member: {key}")
                members[key] = member
        if len(roots) != 1:
            raise ValueError("source distribution must have one root")
        for name in sorted(expected | runtime | data):
            if name not in members:
                raise ValueError(f"source distribution missing {name}")
            with source.extractfile(members[name]) as stream:
                if stream.read() != (ROOT / name).read_bytes():
                    raise ValueError(f"stale source distribution member: {name}")
        if any(name.startswith(("experiments/", ".git/", "out/")) for name in members):
            raise ValueError("private work/results leaked into source distribution")
    with zipfile.ZipFile(wheel) as binary:
        names = binary.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel member")
        for name in sorted(runtime | data):
            if name not in names:
                raise ValueError(f"wheel missing {name}")
            if binary.read(name) != (ROOT / name).read_bytes():
                raise ValueError(f"stale wheel runtime member: {name}")
        unexpected = {name for name in names if name.startswith(("hbserve/", "hbfsim_client/"))} - runtime - data
        if unexpected:
            raise ValueError(f"untracked runtime files in wheel: {sorted(unexpected)}")
    return {"status": "PASS_DISTRIBUTION_CONTENTS", "source_inputs_checked": len(expected | runtime | data),
            "wheel_runtime_files_checked": len(runtime | data), "hardware_fidelity_validated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(check(arguments.sdist, arguments.wheel), sort_keys=True))

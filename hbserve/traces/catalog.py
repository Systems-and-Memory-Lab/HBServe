"""List or export exact reference sources from the checkout's template catalog.

No network connection, GPU capture, model download or fine expansion is used.
The catalog ships with the source checkout/sdist, not the small runtime wheel.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .artifacts import SCHEMA as MAP_SCHEMA, contained_path
from .common import load_json, new_output, require, save, sha256_file

SCHEMA = {"name": "hbserve.reference_source_catalog", "version": 1}


def read_catalog(path: Path) -> dict[str, Any]:
    document = load_json(path)
    require(document.get("schema") == SCHEMA, "unsupported source catalog schema")
    require(isinstance(document.get("artifacts"), dict), "catalog artifacts are missing")
    for key in ("recipes", "templates"):
        require(isinstance(document.get(key), list), f"catalog {key} are missing")
        ids = [row["id"] for row in document[key]]
        require(len(ids) == len(set(ids)), f"duplicate catalog {key} IDs")
    return document


def list_entries(catalog: Path, *, kind: str = "recipes", contains: str = "") -> dict[str, Any]:
    document = read_catalog(catalog)
    rows = [row for row in document[kind] if contains.lower() in json.dumps(row).lower()]
    return {"kind": kind, "count": len(rows), "entries": rows,
            "warning": "Source availability and generation checks are not new hardware-fidelity validation."}


def verify_catalog(catalog: Path) -> dict[str, Any]:
    """Check every packaged blob without expanding a model or using a GPU."""
    document = read_catalog(catalog)
    checked = {}
    for original, row in document["artifacts"].items():
        blob = contained_path(catalog.resolve().parent, row["blob"])
        expected = (row["sha256"], row["bytes"], row["compressed_bytes"])
        if blob in checked:
            require(checked[blob] == expected, "conflicting artifact identities")
            continue
        require(type(row["bytes"]) is int and row["bytes"] >= 0, "invalid artifact size")
        require(blob.stat().st_size == row["compressed_bytes"], "compressed artifact size differs")
        digest, size = hashlib.sha256(), 0
        with gzip.open(blob, "rb") as source:
            while block := source.read(1024 * 1024):
                size += len(block)
                require(size <= row["bytes"], "artifact exceeds declared size")
                digest.update(block)
        require(size == row["bytes"] and digest.hexdigest() == row["sha256"],
                f"decompressed artifact size/digest differs: {original}")
        checked[blob] = expected
    for kind in ("recipes", "templates"):
        for entry in document[kind]:
            require(entry["source"] in document["artifacts"], "entry metadata is missing")
            if kind == "recipes":
                require(set(entry["missing"]) == set(entry["artifacts"]) - document["artifacts"].keys(),
                        "recipe missing-input census differs")
            elif entry.get("binary_source") is not None:
                body = document["artifacts"][entry["binary_source"]]
                require(body["sha256"] == entry["binary_sha256"]
                        and body["bytes"] == entry["binary_bytes"], "template body census differs")
    return {"status": "PASS_PACKAGED_SOURCE_INTEGRITY_NOT_FIDELITY",
            "artifacts": len(document["artifacts"]), "unique_blobs": len(checked),
            "uncompressed_bytes": sum(row[1] for row in checked.values()),
            "compressed_bytes": sum(row[2] for row in checked.values()),
            "recipes": len(document["recipes"]),
            "complete_input_closures": sum(not row["missing"] for row in document["recipes"]),
            "templates": len(document["templates"]),
            "packaged_template_bodies": sum(row.get("binary_source") is not None for row in document["templates"]),
            "model_expanded": False, "hardware_fidelity_validated": False}


def export_entry(*, catalog: Path, entry_id: str, output: Path) -> dict[str, Any]:
    catalog = catalog.resolve()
    document = read_catalog(catalog)
    rows = [(kind, row) for kind in ("recipes", "templates") for row in document[kind]
            if row["id"] == entry_id]
    require(len(rows) == 1, f"catalog ID is missing or ambiguous: {entry_id}")
    kind, entry = rows[0]
    require(not entry.get("missing"), "recipe has missing inputs; see the catalog")
    names = entry.get("artifacts") if kind == "recipes" else [entry["source"], entry.get("binary_source")]
    require(isinstance(names, list) and names and all(name in document["artifacts"] for name in names),
            "entry has no packaged source body or has incomplete inputs")
    root = new_output(output)
    (root / "artifacts").mkdir()
    mapping, exported = {}, {}
    for name in names:
        row = document["artifacts"][name]
        identity = row["sha256"]
        require(isinstance(identity, str) and len(identity) == 64
                and all(ch in "0123456789abcdef" for ch in identity), "invalid artifact digest")
        require(type(row["bytes"]) is int and row["bytes"] >= 0, "invalid artifact size")
        require(row["encoding"] in {"json", "binary"}, "unsupported artifact encoding")
        suffix = ".json" if row["encoding"] == "json" else ".bin"
        relative = "artifacts/" + identity + suffix
        if relative not in exported:
            blob = contained_path(catalog.parent, row["blob"])
            require(blob.stat().st_size == row["compressed_bytes"], "compressed artifact size differs")
            target = root / relative
            partial = target.with_suffix(target.suffix + ".partial")
            digest, size = hashlib.sha256(), 0
            with gzip.open(blob, "rb") as source, partial.open("xb") as sink:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    require(size <= row["bytes"], "artifact exceeds declared size")
                    digest.update(block)
                    sink.write(block)
            require(size == row["bytes"] and digest.hexdigest() == identity,
                    f"decompressed artifact size/digest differs: {name}")
            partial.rename(target)
            exported[relative] = size
        require(exported[relative] == row["bytes"], "conflicting artifact sizes")
        mapping[name] = {"path": relative, "sha256": identity, "bytes": row["bytes"]}
    save(root / "artifact-map.json", {"schema": MAP_SCHEMA, "entry_id": entry_id, "artifacts": mapping})
    result = {
        "status": "EXPORTED_EXACT_SOURCE_INPUTS_NOT_NEW_FIDELITY_VALIDATION", "entry_id": entry_id,
        "kind": kind, "label": entry["label"], "catalog_sha256": sha256_file(catalog),
        "entrypoint": mapping[entry["source"]]["path"], "artifact_map": "artifact-map.json",
        "unique_files": len(exported), "uncompressed_bytes": sum(exported.values()),
        "source_bytes_rewritten": False, "source_status": entry["status"],
        "note": "Resolve these paths relative to export.json. A template alone is not a full-model plan.",
    }
    save(root / "export.json", result)
    return {**result, "output_root": str(root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hbserve trace catalog", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--kind", choices=("recipes", "templates"), default="recipes")
    listing.add_argument("--contains", default="")
    exporting = sub.add_parser("export")
    exporting.add_argument("--id", required=True)
    exporting.add_argument("--output-root", type=Path, required=True)
    verification = sub.add_parser("verify", help="size/digest-check every packaged source blob; no model expansion")
    for command in (listing, exporting, verification):
        command.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "list":
        result = list_entries(args.catalog, kind=args.kind, contains=args.contains)
    elif args.action == "export":
        result = export_entry(catalog=args.catalog, entry_id=args.id, output=args.output_root)
    else:
        result = verify_catalog(args.catalog)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

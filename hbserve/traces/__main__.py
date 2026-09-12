"""Reference tools; native coarse generation remains under hbserve run."""

from __future__ import annotations

import json
import sys
from typing import Sequence

from . import ROUTES

USAGE = """usage: hbserve trace <reference|prepare|catalog|replay|list> [options]

  reference  capture-bound generation through an explicit GPU cache
  prepare    inspect source/layout objects or bind a native reference window
  catalog    list/export packaged, digest-bound template sources
  replay     standalone reference replay through the current HBFSim client
  list       print the reference evidence boundary

Native coarse experiments use hbserve run. Historical simple is not included.
Use hbserve run --experiment for native fixed-window placement/execution.
"""


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0
    command, rest = arguments[0], arguments[1:]
    if command == "list" and not rest:
        print(json.dumps(ROUTES, indent=2, sort_keys=True))
        return 0
    if command not in {"reference", "prepare", "catalog", "replay"}:
        print(f"unknown reference command: {command}\n{USAGE}", file=sys.stderr)
        return 2
    try:
        if command == "catalog":
            from .catalog import main as run
        elif command == "prepare":
            from .prepare import main as run
        elif command == "reference":
            from .reference import main as run
        else:
            from .replay import main as run
        return run(rest)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"{command} failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

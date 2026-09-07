#!/usr/bin/env python3
"""Command-line entry point for HBServe."""

from __future__ import annotations

import json
import sys
from typing import Sequence


USAGE = """usage: hbserve <command> [options]

commands:
  run       run a serving workload through one external HBFSim session
  model     convert a catalog descriptor into hbserve.model JSON
  generate  generate a deterministic Poisson/lognormal request trace
  capabilities
            print the machine-readable fidelity boundary

Run `hbserve <command> --help` for the command's options.
"""


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0
    if arguments[0] == "--version":
        from hbserve import __version__

        print(__version__)
        return 0
    command, rest = arguments[0], arguments[1:]
    if command == "run":
        from hbserve.run import main as run_main

        return run_main(rest)
    if command == "model":
        from hbserve.catalog import main as model_main

        return model_main(rest)
    if command == "generate":
        from hbserve.generate import main as generate_main

        return generate_main(rest)
    if command == "capabilities":
        if rest:
            print("hbserve capabilities takes no options", file=sys.stderr)
            return 2
        from hbserve.capabilities import current_capabilities

        print(json.dumps(current_capabilities(), indent=2, sort_keys=True))
        return 0
    print(f"unknown hbserve command: {command}\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
